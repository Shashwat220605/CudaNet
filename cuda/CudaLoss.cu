#include "CudaLoss.h"
#include "CudaUtils.h"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

__global__ void crossEntropyKernel(
    const float* probabilities,
    const float* targets,
    float* losses,
    int total)
{
    int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= total)
    {
        return;
    }

    float probability =
        probabilities[index];

    // Prevent log(0)
    probability =
        fmaxf(probability, 1e-7f);

    float target =
        targets[index];

    losses[index] =
        -target * logf(probability);
}


float cudaCrossEntropyLoss(
    const CudaMatrix& probabilities,
    const CudaMatrix& targets)
{
    // Check dimensions
    if (probabilities.rows() != targets.rows() ||
        probabilities.cols() != targets.cols())
    {
        throw std::invalid_argument(
            "Probability and target dimensions must match."
        );
    }

    int rows =
        static_cast<int>(
            probabilities.rows()
        );

    int cols =
        static_cast<int>(
            probabilities.cols()
        );

    int total = rows * cols;

    // GPU matrix containing individual loss values
    CudaMatrix lossMatrix(
        rows,
        cols
    );

    constexpr int threads = 256;

    int blocks =
        (total + threads - 1) / threads;

    // Launch CUDA kernel
    crossEntropyKernel<<<
        blocks,
        threads
    >>>(
        probabilities.data(),
        targets.data(),
        lossMatrix.data(),
        total
    );

    cudaCheck(
        cudaGetLastError(),
        "Cross entropy kernel"
    );

    cudaCheck(
        cudaDeviceSynchronize(),
        "Cross entropy synchronization"
    );

    // Copy loss values to CPU
    Matrix hostLoss =
        lossMatrix.copyToHost();

    // Sum all loss values
    float totalLoss = 0.0f;

    for (std::size_t i = 0;
         i < hostLoss.rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < hostLoss.cols();
             ++j)
        {
            totalLoss +=
                hostLoss(i, j);
        }
    }

    // Average over batch
    return totalLoss /
           static_cast<float>(rows);
}