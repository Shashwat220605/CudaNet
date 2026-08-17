#include <cuda_runtime.h>

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