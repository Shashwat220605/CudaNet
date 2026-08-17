#pragma once

#include "CudaMatrix.h"
#include "Matrix.h"

#include <cstddef>

class CudaDenseLayer
{
private:
    std::size_t inputSize_;
    std::size_t outputSize_;

    float* d_weights_;
    float* d_bias_;

    // GPU gradient buffers
    float* d_weight_grad_;
    float* d_bias_grad_;

public:
    CudaDenseLayer(
        std::size_t inputSize,
        std::size_t outputSize
    );

    ~CudaDenseLayer();

    CudaDenseLayer(
        const CudaDenseLayer&
    ) = delete;

    CudaDenseLayer& operator=(
        const CudaDenseLayer&
    ) = delete;

    CudaDenseLayer(
        CudaDenseLayer&& other
    ) noexcept;

    CudaDenseLayer& operator=(
        CudaDenseLayer&& other
    ) noexcept;

    // CPU input -> GPU computation -> CPU output
    Matrix forward(
        const Matrix& input
    );

    // GPU input -> GPU computation -> GPU output
    CudaMatrix forwardGpu(
        const CudaMatrix& input
    );

    // GPU backward pass
    //
    // input:
    //     input that was used during forward
    //
    // gradient:
    //     dL/dZ from the next layer
    //
    // learningRate:
    //     weight update amount
    //
    // returns:
    //     dL/dX
    CudaMatrix backwardGpu(
        const CudaMatrix& input,
        const CudaMatrix& gradient,
        float learningRate
    );

    void randomizeWeights(
        float min = -0.05f,
        float max = 0.05f
    );

    void zeroBias();

    std::size_t inputSize() const;
    std::size_t outputSize() const;

    // Copy gradients to CPU for debugging/checking
    Matrix weightGradientToHost() const;
    Matrix biasGradientToHost() const;
};