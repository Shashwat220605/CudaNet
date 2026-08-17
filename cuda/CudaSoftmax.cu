#include "CudaSoftmax.h"
#include "CudaUtils.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <stdexcept>


// ============================================================
// GPU Softmax
//
// Each CUDA block handles one row.
//
// input:
//     [rows x cols]
//
// output:
//     [rows x cols]
//
// For each row:
//
//     softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
// ============================================================

__global__ void softmaxKernel(
    const float* input,
    float* output,
    int rows,
    int cols)
{
    int row = blockIdx.x;

    if (row >= rows)
    {
        return;
    }

    // For the current MNIST output layer we only
    // have 10 classes, so one thread can handle
    // the entire row safely and simply.
    if (threadIdx.x != 0)
    {
        return;
    }

    // --------------------------------------------------------
    // 1. Find maximum value
    // --------------------------------------------------------

    float maxValue = -FLT_MAX;

    for (int i = 0;
         i < cols;
         ++i)
    {
        float value =
            input[row * cols + i];

        if (value > maxValue)
        {
            maxValue = value;
        }
    }

    // --------------------------------------------------------
    // 2. Calculate exponentials
    //
    // Subtracting maxValue prevents numerical overflow.
    // --------------------------------------------------------

    float sum = 0.0f;

    for (int i = 0;
         i < cols;
         ++i)
    {
        float value =
            expf(
                input[row * cols + i]
                - maxValue
            );

        output[row * cols + i] =
            value;

        sum += value;
    }

    // --------------------------------------------------------
    // 3. Normalize
    // --------------------------------------------------------

    for (int i = 0;
         i < cols;
         ++i)
    {
        output[row * cols + i] /=
            sum;
    }
}


// ============================================================
// GPU Softmax
// ============================================================

CudaMatrix cudaSoftmaxGpu(
    const CudaMatrix& input)
{
    CudaMatrix output(
        input.rows(),
        input.cols()
    );

    int rows =
        static_cast<int>(
            input.rows()
        );

    int cols =
        static_cast<int>(
            input.cols()
        );

    // One block per row.
    int blocks = rows;

    // We only need one active thread per row
    // for this simple implementation.
    constexpr int threads = 32;

    softmaxKernel<<<
        blocks,
        threads
    >>>(
        input.data(),
        output.data(),
        rows,
        cols
    );

    cudaCheck(
        cudaGetLastError(),
        "GPU Softmax kernel"
    );

    cudaCheck(
        cudaDeviceSynchronize(),
        "GPU Softmax synchronization"
    );

    return output;
}


// ============================================================
// Softmax + Cross-Entropy gradient
//
// For:
//
//     L = CrossEntropy(Softmax(logits), targets)
//
// the gradient simplifies to:
//
//     dLogits = probabilities - targets
//
// probabilities:
//     [rows x cols]
//
// targets:
//     [rows x cols]
//
// output:
//     [rows x cols]
// ============================================================

__global__ void softmaxCrossEntropyGradientKernel(
    const float* probabilities,
    const float* targets,
    float* gradient,
    int total)
{
    int index =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (index >= total)
    {
        return;
    }

    gradient[index] =
        probabilities[index]
        - targets[index];
}


// ============================================================
// GPU Softmax + Cross-Entropy gradient
// ============================================================

CudaMatrix cudaSoftmaxCrossEntropyGradientGpu(
    const CudaMatrix& probabilities,
    const CudaMatrix& targets)
{
    // --------------------------------------------------------
    // Check dimensions
    // --------------------------------------------------------

    if (probabilities.rows() != targets.rows() ||
        probabilities.cols() != targets.cols())
    {
        throw std::invalid_argument(
            "Probabilities and targets dimensions "
            "must match."
        );
    }

    // --------------------------------------------------------
    // Allocate output gradient on GPU
    // --------------------------------------------------------

    CudaMatrix gradient(
        probabilities.rows(),
        probabilities.cols()
    );

    int total =
        static_cast<int>(
            probabilities.rows()
            * probabilities.cols()
        );

    // --------------------------------------------------------
    // Launch kernel
    // --------------------------------------------------------

    constexpr int threads = 256;

    int blocks =
        (total + threads - 1)
        / threads;

    softmaxCrossEntropyGradientKernel<<<
        blocks,
        threads
    >>>(
        probabilities.data(),
        targets.data(),
        gradient.data(),
        total
    );

    // --------------------------------------------------------
    // Check kernel
    // --------------------------------------------------------

    cudaCheck(
        cudaGetLastError(),
        "Softmax cross entropy gradient kernel"
    );

    cudaCheck(
        cudaDeviceSynchronize(),
        "Softmax cross entropy gradient synchronization"
    );

    return gradient;
}