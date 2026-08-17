#include "CudaReLU.h"
#include "CudaUtils.h"

#include <cuda_runtime.h>

#include <stdexcept>

__global__ void reluKernel(
    const float* input,
    float* output,
    int totalElements)
{
    int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index < totalElements)
    {
        float value = input[index];

        output[index] =
            value > 0.0f ? value : 0.0f;
    }
}


// ============================================================
// ReLU backward
//
// dX = dY when X > 0
// dX = 0  when X <= 0
// ============================================================

__global__ void reluBackwardKernel(
    const float* input,
    const float* gradient,
    float* output,
    int totalElements)
{
    int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index < totalElements)
    {
        if (input[index] > 0.0f)
        {
            output[index] =
                gradient[index];
        }
        else
        {
            output[index] = 0.0f;
        }
    }
}


Matrix cudaRelu(
    const Matrix& input,
    float* kernelTimeMs)
{
    const std::size_t totalElements =
        input.rows() * input.cols();

    const std::size_t size =
        totalElements * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaCheck(
        cudaMalloc(&d_input, size),
        "cudaMalloc input"
    );

    cudaCheck(
        cudaMalloc(&d_output, size),
        "cudaMalloc output"
    );

    cudaCheck(
        cudaMemcpy(
            d_input,
            input.data(),
            size,
            cudaMemcpyHostToDevice
        ),
        "Copy input to GPU"
    );

    constexpr int threadsPerBlock = 256;

    int blocksPerGrid =
        static_cast<int>(
            (totalElements + threadsPerBlock - 1)
            / threadsPerBlock
        );

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    cudaCheck(
        cudaEventCreate(&startEvent),
        "cudaEventCreate start"
    );

    cudaCheck(
        cudaEventCreate(&stopEvent),
        "cudaEventCreate stop"
    );

    cudaCheck(
        cudaEventRecord(startEvent),
        "cudaEventRecord start"
    );

    reluKernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        d_input,
        d_output,
        static_cast<int>(totalElements)
    );

    cudaCheck(
        cudaGetLastError(),
        "ReLU kernel launch"
    );

    cudaCheck(
        cudaEventRecord(stopEvent),
        "cudaEventRecord stop"
    );

    cudaCheck(
        cudaEventSynchronize(stopEvent),
        "cudaEventSynchronize"
    );

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

    Matrix result(
        input.rows(),
        input.cols()
    );

    cudaCheck(
        cudaMemcpy(
            result.data(),
            d_output,
            size,
            cudaMemcpyDeviceToHost
        ),
        "Copy ReLU result to CPU"
    );

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}


CudaMatrix cudaReluGpu(
    const CudaMatrix& input,
    float* kernelTimeMs)
{
    CudaMatrix output(
        input.rows(),
        input.cols()
    );

    const int totalElements =
        static_cast<int>(
            input.rows() * input.cols()
        );

    constexpr int threadsPerBlock = 256;

    const int blocksPerGrid =
        (totalElements + threadsPerBlock - 1)
        / threadsPerBlock;

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    cudaCheck(
        cudaEventCreate(&startEvent),
        "cudaEventCreate start"
    );

    cudaCheck(
        cudaEventCreate(&stopEvent),
        "cudaEventCreate stop"
    );

    cudaCheck(
        cudaEventRecord(startEvent),
        "cudaEventRecord start"
    );

    reluKernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        input.data(),
        output.data(),
        totalElements
    );

    cudaCheck(
        cudaGetLastError(),
        "GPU ReLU kernel"
    );

    cudaCheck(
        cudaEventRecord(stopEvent),
        "cudaEventRecord stop"
    );

    cudaCheck(
        cudaEventSynchronize(stopEvent),
        "cudaEventSynchronize"
    );

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

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return output;
}


// ============================================================
// GPU ReLU backward
// ============================================================

CudaMatrix cudaReluBackwardGpu(
    const CudaMatrix& input,
    const CudaMatrix& gradient)
{
    if (input.rows() != gradient.rows() ||
        input.cols() != gradient.cols())
    {
        throw std::invalid_argument(
            "ReLU input and gradient dimensions "
            "must match."
        );
    }

    CudaMatrix output(
        input.rows(),
        input.cols()
    );

    const int totalElements =
        static_cast<int>(
            input.rows() * input.cols()
        );

    constexpr int threadsPerBlock = 256;

    const int blocksPerGrid =
        (totalElements + threadsPerBlock - 1)
        / threadsPerBlock;

    reluBackwardKernel<<<
        blocksPerGrid,
        threadsPerBlock
    >>>(
        input.data(),
        gradient.data(),
        output.data(),
        totalElements
    );

    cudaCheck(
        cudaGetLastError(),
        "GPU ReLU backward kernel"
    );

    cudaCheck(
        cudaDeviceSynchronize(),
        "GPU ReLU backward synchronization"
    );

    return output;
}