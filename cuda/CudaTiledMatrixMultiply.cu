#include "CudaDenseLayer.h"
#include "CudaUtils.h"

#include <cuda_runtime.h>

#include <random>
#include <stdexcept>
#include <vector>

constexpr int TILE_SIZE = 16;


// ============================================================
// Forward matrix multiplication
//
// output = input * weights
//
// input:   [rows x inputSize]
// weights: [inputSize x outputSize]
// output:  [rows x outputSize]
// ============================================================

__global__ void denseMatMulKernel(
    const float* input,
    const float* weights,
    float* output,
    int rows,
    int inputSize,
    int outputSize)
{
    __shared__ float sharedInput[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedWeights[TILE_SIZE][TILE_SIZE];

    int row =
        blockIdx.y * TILE_SIZE + threadIdx.y;

    int col =
        blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int numTiles =
        (inputSize + TILE_SIZE - 1)
        / TILE_SIZE;

    for (int tile = 0;
         tile < numTiles;
         ++tile)
    {
        int inputCol =
            tile * TILE_SIZE + threadIdx.x;

        int weightRow =
            tile * TILE_SIZE + threadIdx.y;

        if (row < rows &&
            inputCol < inputSize)
        {
            sharedInput[
                threadIdx.y
            ][
                threadIdx.x
            ] =
                input[
                    row * inputSize
                    + inputCol
                ];
        }
        else
        {
            sharedInput[
                threadIdx.y
            ][
                threadIdx.x
            ] = 0.0f;
        }

        if (weightRow < inputSize &&
            col < outputSize)
        {
            sharedWeights[
                threadIdx.y
            ][
                threadIdx.x
            ] =
                weights[
                    weightRow * outputSize
                    + col
                ];
        }
        else
        {
            sharedWeights[
                threadIdx.y
            ][
                threadIdx.x
            ] = 0.0f;
        }

        __syncthreads();

        for (int k = 0;
             k < TILE_SIZE;
             ++k)
        {
            sum +=
                sharedInput[
                    threadIdx.y
                ][k]
                *
                sharedWeights[
                    k
                ][
                    threadIdx.x
                ];
        }

        __syncthreads();
    }

    if (row < rows &&
        col < outputSize)
    {
        output[
            row * outputSize + col
        ] = sum;
    }
}


// ============================================================
// Add bias
//
// output[row][col] += bias[col]
// ============================================================

__global__ void addBiasKernel(
    float* output,
    const float* bias,
    int rows,
    int outputSize)
{
    int index =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int totalElements =
        rows * outputSize;

    if (index < totalElements)
    {
        int column =
            index % outputSize;

        output[index] += bias[column];
    }
}


// ============================================================
// dW kernel
//
// dW[i][j] = sum over rows:
//             input[row][i] * gradient[row][j]
//
// input:    [rows x inputSize]
// gradient: [rows x outputSize]
// dW:       [inputSize x outputSize]
// ============================================================

__global__ void denseWeightGradientKernel(
    const float* input,
    const float* gradient,
    float* weightGradient,
    int rows,
    int inputSize,
    int outputSize)
{
    int index =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int total =
        inputSize * outputSize;

    if (index >= total)
    {
        return;
    }

    int inputIndex =
        index / outputSize;

    int outputIndex =
        index % outputSize;

    float sum = 0.0f;

    for (int row = 0;
         row < rows;
         ++row)
    {
        sum +=
            input[
                row * inputSize
                + inputIndex
            ]
            *
            gradient[
                row * outputSize
                + outputIndex
            ];
    }

    weightGradient[index] = sum;
}


// ============================================================
// dB kernel
//
// dB[j] = sum over rows gradient[row][j]
// ============================================================

__global__ void denseBiasGradientKernel(
    const float* gradient,
    float* biasGradient,
    int rows,
    int outputSize)
{
    int column =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (column >= outputSize)
    {
        return;
    }

    float sum = 0.0f;

    for (int row = 0;
         row < rows;
         ++row)
    {
        sum +=
            gradient[
                row * outputSize
                + column
            ];
    }

    biasGradient[column] = sum;
}


// ============================================================
// dX kernel
//
// dX[row][i] = sum over j:
//               gradient[row][j] * weights[i][j]
//
// gradient: [rows x outputSize]
// weights:  [inputSize x outputSize]
// dX:       [rows x inputSize]
// ============================================================

__global__ void denseInputGradientKernel(
    const float* gradient,
    const float* weights,
    float* inputGradient,
    int rows,
    int inputSize,
    int outputSize)
{
    int index =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int total =
        rows * inputSize;

    if (index >= total)
    {
        return;
    }

    int row =
        index / inputSize;

    int inputIndex =
        index % inputSize;

    float sum = 0.0f;

    for (int j = 0;
         j < outputSize;
         ++j)
    {
        sum +=
            gradient[
                row * outputSize + j
            ]
            *
            weights[
                inputIndex * outputSize + j
            ];
    }

    inputGradient[index] = sum;
}


// ============================================================
// Weight update
//
// W -= learningRate * dW
// ============================================================

__global__ void updateWeightsKernel(
    float* weights,
    const float* gradients,
    float learningRate,
    int total)
{
    int index =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (index < total)
    {
        weights[index] -=
            learningRate
            * gradients[index];
    }
}


// ============================================================
// Bias update
//
// B -= learningRate * dB
// ============================================================

__global__ void updateBiasKernel(
    float* bias,
    const float* gradients,
    float learningRate,
    int outputSize)
{
    int index =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (index < outputSize)
    {
        bias[index] -=
            learningRate
            * gradients[index];
    }
}


// ============================================================
// Constructor
// ============================================================

CudaDenseLayer::CudaDenseLayer(
    std::size_t inputSize,
    std::size_t outputSize)
    : inputSize_(inputSize),
      outputSize_(outputSize),
      d_weights_(nullptr),
      d_bias_(nullptr),
      d_weight_grad_(nullptr),
      d_bias_grad_(nullptr)
{
    std::size_t weightBytes =
        inputSize_
        * outputSize_
        * sizeof(float);

    std::size_t biasBytes =
        outputSize_
        * sizeof(float);

    cudaCheck(
        cudaMalloc(
            &d_weights_,
            weightBytes
        ),
        "cudaMalloc weights"
    );

    cudaCheck(
        cudaMalloc(
            &d_bias_,
            biasBytes
        ),
        "cudaMalloc bias"
    );

    cudaCheck(
        cudaMalloc(
            &d_weight_grad_,
            weightBytes
        ),
        "cudaMalloc weight gradients"
    );

    cudaCheck(
        cudaMalloc(
            &d_bias_grad_,
            biasBytes
        ),
        "cudaMalloc bias gradients"
    );

    zeroBias();

    randomizeWeights();
}


// ============================================================
// Destructor
// ============================================================

CudaDenseLayer::~CudaDenseLayer()
{
    if (d_weights_)
    {
        cudaFree(d_weights_);
    }

    if (d_bias_)
    {
        cudaFree(d_bias_);
    }

    if (d_weight_grad_)
    {
        cudaFree(d_weight_grad_);
    }

    if (d_bias_grad_)
    {
        cudaFree(d_bias_grad_);
    }
}


// ============================================================
// Move constructor
// ============================================================

CudaDenseLayer::CudaDenseLayer(
    CudaDenseLayer&& other) noexcept
    : inputSize_(other.inputSize_),
      outputSize_(other.outputSize_),
      d_weights_(other.d_weights_),
      d_bias_(other.d_bias_),
      d_weight_grad_(other.d_weight_grad_),
      d_bias_grad_(other.d_bias_grad_)
{
    other.d_weights_ = nullptr;
    other.d_bias_ = nullptr;
    other.d_weight_grad_ = nullptr;
    other.d_bias_grad_ = nullptr;
}


// ============================================================
// Move assignment
// ============================================================

CudaDenseLayer& CudaDenseLayer::operator=(
    CudaDenseLayer&& other) noexcept
{
    if (this != &other)
    {
        if (d_weights_)
        {
            cudaFree(d_weights_);
        }

        if (d_bias_)
        {
            cudaFree(d_bias_);
        }

        if (d_weight_grad_)
        {
            cudaFree(d_weight_grad_);
        }

        if (d_bias_grad_)
        {
            cudaFree(d_bias_grad_);
        }

        inputSize_ = other.inputSize_;
        outputSize_ = other.outputSize_;

        d_weights_ = other.d_weights_;
        d_bias_ = other.d_bias_;
        d_weight_grad_ = other.d_weight_grad_;
        d_bias_grad_ = other.d_bias_grad_;

        other.d_weights_ = nullptr;
        other.d_bias_ = nullptr;
        other.d_weight_grad_ = nullptr;
        other.d_bias_grad_ = nullptr;
    }

    return *this;
}


// ============================================================
// Randomize weights
// ============================================================

void CudaDenseLayer::randomizeWeights(
    float min,
    float max)
{
    std::size_t count =
        inputSize_
        * outputSize_;

    std::vector<float> weights(count);

    std::mt19937 generator(42);

    std::uniform_real_distribution<float>
        distribution(min, max);

    for (float& value : weights)
    {
        value =
            distribution(generator);
    }

    cudaCheck(
        cudaMemcpy(
            d_weights_,
            weights.data(),
            count * sizeof(float),
            cudaMemcpyHostToDevice
        ),
        "Copy weights to GPU"
    );
}


// ============================================================
// Zero bias
// ============================================================

void CudaDenseLayer::zeroBias()
{
    cudaCheck(
        cudaMemset(
            d_bias_,
            0,
            outputSize_ * sizeof(float)
        ),
        "Initialize bias"
    );
}


// ============================================================
// CPU forward
// ============================================================

Matrix CudaDenseLayer::forward(
    const Matrix& input)
{
    if (input.cols() != inputSize_)
    {
        throw std::invalid_argument(
            "Input size does not match CudaDenseLayer."
        );
    }

    const int rows =
        static_cast<int>(
            input.rows()
        );

    const std::size_t inputBytes =
        input.rows()
        * input.cols()
        * sizeof(float);

    const std::size_t outputBytes =
        input.rows()
        * outputSize_
        * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaCheck(
        cudaMalloc(
            &d_input,
            inputBytes
        ),
        "cudaMalloc input"
    );

    cudaCheck(
        cudaMalloc(
            &d_output,
            outputBytes
        ),
        "cudaMalloc output"
    );

    cudaCheck(
        cudaMemcpy(
            d_input,
            input.data(),
            inputBytes,
            cudaMemcpyHostToDevice
        ),
        "Copy input to GPU"
    );

    dim3 threads(
        TILE_SIZE,
        TILE_SIZE
    );

    dim3 blocks(
        (outputSize_ + TILE_SIZE - 1)
            / TILE_SIZE,

        (rows + TILE_SIZE - 1)
            / TILE_SIZE
    );

    denseMatMulKernel<<<
        blocks,
        threads
    >>>(
        d_input,
        d_weights_,
        d_output,
        rows,
        static_cast<int>(inputSize_),
        static_cast<int>(outputSize_)
    );

    cudaCheck(
        cudaGetLastError(),
        "Dense matrix multiplication"
    );

    int totalElements =
        rows
        * static_cast<int>(outputSize_);

    constexpr int biasThreads = 256;

    int biasBlocks =
        (totalElements + biasThreads - 1)
        / biasThreads;

    addBiasKernel<<<
        biasBlocks,
        biasThreads
    >>>(
        d_output,
        d_bias_,
        rows,
        static_cast<int>(outputSize_)
    );

    cudaCheck(
        cudaGetLastError(),
        "Bias kernel"
    );

    cudaCheck(
        cudaDeviceSynchronize(),
        "CudaDenseLayer synchronization"
    );

    Matrix result(
        input.rows(),
        outputSize_
    );

    cudaCheck(
        cudaMemcpy(
            result.data(),
            d_output,
            outputBytes,
            cudaMemcpyDeviceToHost
        ),
        "Copy Dense output to CPU"
    );

    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}


// ============================================================
// GPU forward
// ============================================================

CudaMatrix CudaDenseLayer::forwardGpu(
    const CudaMatrix& input)
{
    if (input.cols() != inputSize_)
    {
        throw std::invalid_argument(
            "Input size does not match CudaDenseLayer."
        );
    }

    CudaMatrix output(
        input.rows(),
        outputSize_
    );

    dim3 threads(
        TILE_SIZE,
        TILE_SIZE
    );

    dim3 blocks(
        (outputSize_ + TILE_SIZE - 1)
            / TILE_SIZE,

        (input.rows() + TILE_SIZE - 1)
            / TILE_SIZE
    );

    denseMatMulKernel<<<
        blocks,
        threads
    >>>(
        input.data(),
        d_weights_,
        output.data(),
        static_cast<int>(input.rows()),
        static_cast<int>(inputSize_),
        static_cast<int>(outputSize_)
    );

    cudaCheck(
        cudaGetLastError(),
        "GPU Dense matrix multiplication"
    );

    int totalElements =
        static_cast<int>(
            input.rows()
            * outputSize_
        );

    constexpr int biasThreads = 256;

    int biasBlocks =
        (totalElements + biasThreads - 1)
        / biasThreads;

    addBiasKernel<<<
        biasBlocks,
        biasThreads
    >>>(
        output.data(),
        d_bias_,
        static_cast<int>(input.rows()),
        static_cast<int>(outputSize_)
    );

    cudaCheck(
        cudaGetLastError(),
        "GPU Dense bias"
    );

    cudaCheck(
        cudaDeviceSynchronize(),
        "GPU Dense synchronization"
    );

    return output;
}


// ============================================================
// GPU backward
// ============================================================

CudaMatrix CudaDenseLayer::backwardGpu(
    const CudaMatrix& input,
    const CudaMatrix& gradient,
    float learningRate)
{
    if (input.cols() != inputSize_)
    {
        throw std::invalid_argument(
            "Backward input dimensions do not match."
        );
    }

    if (gradient.rows() != input.rows() ||
        gradient.cols() != outputSize_)
    {
        throw std::invalid_argument(
            "Backward gradient dimensions do not match."
        );
    }

    int rows =
        static_cast<int>(
            input.rows()
        );

    int inputSize =
        static_cast<int>(
            inputSize_
        );

    int outputSize =
        static_cast<int>(
            outputSize_
        );

    // --------------------------------------------------------
    // 1. Calculate dW
    //
    // dW = X^T * dZ
    // --------------------------------------------------------

    int weightCount =
        inputSize * outputSize;

    constexpr int threads = 256;

    int weightBlocks =
        (weightCount + threads - 1)
        / threads;

    denseWeightGradientKernel<<<
        weightBlocks,
        threads
    >>>(
        input.data(),
        gradient.data(),
        d_weight_grad_,
        rows,
        inputSize,
        outputSize
    );

    cudaCheck(
        cudaGetLastError(),
        "GPU Dense weight gradient"
    );


    // --------------------------------------------------------
    // 2. Calculate dB
    //
    // dB = sum(dZ)
    // --------------------------------------------------------

    int biasBlocks =
        (outputSize + threads - 1)
        / threads;

    denseBiasGradientKernel<<<
        biasBlocks,
        threads
    >>>(
        gradient.data(),
        d_bias_grad_,
        rows,
        outputSize
    );

    cudaCheck(
        cudaGetLastError(),
        "GPU Dense bias gradient"
    );


    // --------------------------------------------------------
    // 3. Calculate dX
    //
    // dX = dZ * W^T
    // --------------------------------------------------------

    CudaMatrix inputGradient(
        input.rows(),
        inputSize_
    );

    int inputGradientCount =
        rows * inputSize;

    int inputGradientBlocks =
        (inputGradientCount + threads - 1)
        / threads;

    denseInputGradientKernel<<<
        inputGradientBlocks,
        threads
    >>>(
        gradient.data(),
        d_weights_,
        inputGradient.data(),
        rows,
        inputSize,
        outputSize
    );

    cudaCheck(
        cudaGetLastError(),
        "GPU Dense input gradient"
    );


    // --------------------------------------------------------
    // 4. Update weights
    //
    // W -= learningRate * dW
    // --------------------------------------------------------

    if (learningRate != 0.0f)
    {
        updateWeightsKernel<<<
            weightBlocks,
            threads
        >>>(
            d_weights_,
            d_weight_grad_,
            learningRate,
            weightCount
        );

        cudaCheck(
            cudaGetLastError(),
            "GPU Dense weight update"
        );


        // ----------------------------------------------------
        // 5. Update bias
        //
        // B -= learningRate * dB
        // ----------------------------------------------------

        updateBiasKernel<<<
            biasBlocks,
            threads
        >>>(
            d_bias_,
            d_bias_grad_,
            learningRate,
            outputSize
        );

        cudaCheck(
            cudaGetLastError(),
            "GPU Dense bias update"
        );
    }

    cudaCheck(
        cudaDeviceSynchronize(),
        "GPU Dense backward synchronization"
    );

    return inputGradient;
}


// ============================================================
// Get dimensions
// ============================================================

std::size_t CudaDenseLayer::inputSize() const
{
    return inputSize_;
}


std::size_t CudaDenseLayer::outputSize() const
{
    return outputSize_;
}


// ============================================================
// Copy weight gradient to CPU
// ============================================================

Matrix CudaDenseLayer::weightGradientToHost() const
{
    Matrix result(
        inputSize_,
        outputSize_
    );

    std::size_t bytes =
        inputSize_
        * outputSize_
        * sizeof(float);

    cudaCheck(
        cudaMemcpy(
            result.data(),
            d_weight_grad_,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "Copy weight gradient to CPU"
    );

    return result;
}


// ============================================================
// Copy bias gradient to CPU
// ============================================================

Matrix CudaDenseLayer::biasGradientToHost() const
{
    Matrix result(
        1,
        outputSize_
    );

    std::size_t bytes =
        outputSize_
        * sizeof(float);

    cudaCheck(
        cudaMemcpy(
            result.data(),
            d_bias_grad_,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "Copy bias gradient to CPU"
    );

    return result;
}