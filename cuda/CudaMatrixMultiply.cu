#include "CudaMatrixMultiply.h"
#include "CudaUtils.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

__global__ void matrixMultiplyKernel(
    const float* A,
    const float* B,
    float* C,
    int A_rows,
    int A_cols,
    int B_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols)
    {
        float sum = 0.0f;

        for (int k = 0; k < A_cols; ++k)
        {
            sum += A[row * A_cols + k]
                 * B[k * B_cols + col];
        }

        C[row * B_cols + col] = sum;
    }
}

Matrix cudaMatrixMultiply(const Matrix& A,
                          const Matrix& B,
                          float* kernelTimeMs
                        )
{
    if (A.cols() != B.rows())
    {
        throw std::invalid_argument(
            "Matrix dimensions are incompatible."
        );
    }

    const int A_rows = static_cast<int>(A.rows());
    const int A_cols = static_cast<int>(A.cols());
    const int B_cols = static_cast<int>(B.cols());

    const size_t A_size =
        A.rows() * A.cols() * sizeof(float);

    const size_t B_size =
        B.rows() * B.cols() * sizeof(float);

    const size_t C_size =
        A.rows() * B.cols() * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    cudaError_t error;

    // Allocate GPU memory
    error = cudaMalloc(&d_A, A_size);

    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("cudaMalloc A failed: ")
            + cudaGetErrorString(error)
        );
    }

    error = cudaMalloc(&d_B, B_size);

    if (error != cudaSuccess)
    {
        cudaFree(d_A);

        throw std::runtime_error(
            std::string("cudaMalloc B failed: ")
            + cudaGetErrorString(error)
        );
    }

    error = cudaMalloc(&d_C, C_size);

    if (error != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);

        throw std::runtime_error(
            std::string("cudaMalloc C failed: ")
            + cudaGetErrorString(error)
        );
    }

    // Copy matrices from CPU → GPU
    error = cudaMemcpy(
        d_A,
        A.data(),
        A_size,
        cudaMemcpyHostToDevice
    );

    if (error != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        throw std::runtime_error(
            std::string("Copy A to GPU failed: ")
            + cudaGetErrorString(error)
        );
    }

    error = cudaMemcpy(
        d_B,
        B.data(),
        B_size,
        cudaMemcpyHostToDevice
    );

    if (error != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        throw std::runtime_error(
            std::string("Copy B to GPU failed: ")
            + cudaGetErrorString(error)
        );
    }

    // 16 × 16 = 256 threads per CUDA block
    dim3 threadsPerBlock(16, 16);

    dim3 blocksPerGrid(
        (B_cols + threadsPerBlock.x - 1)
            / threadsPerBlock.x,

        (A_rows + threadsPerBlock.y - 1)
            / threadsPerBlock.y
    );

    // Create CUDA events for kernel timing
cudaEvent_t startEvent;
cudaEvent_t stopEvent;

cudaCheck(
    cudaEventCreate(&startEvent),
    "cudaEventCreate(start)"
);

cudaCheck(
    cudaEventCreate(&stopEvent),
    "cudaEventCreate(stop)"
);

// Start kernel timer
cudaCheck(
    cudaEventRecord(startEvent),
    "cudaEventRecord(start)"
);

// Launch CUDA kernel
matrixMultiplyKernel<<<blocksPerGrid,
                        threadsPerBlock>>>(
    d_A,
    d_B,
    d_C,
    A_rows,
    A_cols,
    B_cols
);

// Check kernel launch
cudaCheck(
    cudaGetLastError(),
    "matrixMultiplyKernel launch"
);

// Stop kernel timer
cudaCheck(
    cudaEventRecord(stopEvent),
    "cudaEventRecord(stop)"
);

cudaCheck(
    cudaEventSynchronize(stopEvent),
    "cudaEventSynchronize(stop)"
);

// Get kernel execution time
if (kernelTimeMs != nullptr)
{
    cudaCheck(
        cudaEventElapsedTime(
            kernelTimeMs,
            startEvent,
            stopEvent
        ),
        "cudaEventElapsedTime"
    );
}

// Destroy events
cudaEventDestroy(startEvent);
cudaEventDestroy(stopEvent);

// Make sure the kernel completed
cudaCheck(
    cudaDeviceSynchronize(),
    "cudaDeviceSynchronize"
);

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        throw std::runtime_error(
            std::string("Kernel launch failed: ")
            + cudaGetErrorString(error)
        );
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        throw std::runtime_error(
            std::string("Kernel execution failed: ")
            + cudaGetErrorString(error)
        );
    }

    // Create CPU result matrix
    Matrix result(A.rows(), B.cols());

    // Copy result GPU → CPU
    error = cudaMemcpy(
        result.data(),
        d_C,
        C_size,
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        throw std::runtime_error(
            std::string("Copy result to CPU failed: ")
            + cudaGetErrorString(error)
        );
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}