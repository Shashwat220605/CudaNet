#include "DenseLayer.h"

#include <stdexcept>

DenseLayer::DenseLayer(
    std::size_t inputSize,
    std::size_t outputSize)
    : weights_(inputSize, outputSize),
      bias_(1, outputSize),
      input_(1, inputSize),
      output_(1, outputSize),
      inputGradient_(1, inputSize),
      weightGradient_(inputSize, outputSize),
      biasGradient_(1, outputSize)
{
    // Small random initialization
    weights_.randomize(-0.05f, 0.05f);

    // Start biases at zero
    bias_.fill(0.0f);

    // Initialize gradient matrices
    inputGradient_.fill(0.0f);
    weightGradient_.fill(0.0f);
    biasGradient_.fill(0.0f);
}

Matrix DenseLayer::forward(const Matrix& input)
{
    if (input.cols() != weights_.rows())
    {
        throw std::invalid_argument(
            "Input size does not match DenseLayer."
        );
    }

    // Save input for backpropagation
    input_ = input;

    // Z = XW
    Matrix result = input * weights_;

    // Z = XW + B
    for (std::size_t i = 0;
         i < result.rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < result.cols();
             ++j)
        {
            result(i, j) += bias_(0, j);
        }
    }

    output_ = result;

    return output_;
}

void DenseLayer::backward(
    const Matrix& gradient,
    float learningRate)
{
    if (gradient.rows() != output_.rows() ||
        gradient.cols() != output_.cols())
    {
        throw std::invalid_argument(
            "Gradient dimensions do not match layer output."
        );
    }

    // =====================================================
    // 1. Calculate input gradient
    //
    // dX = dZ * W^T
    // =====================================================

    inputGradient_.fill(0.0f);

    for (std::size_t i = 0;
         i < input_.rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < input_.cols();
             ++j)
        {
            float sum = 0.0f;

            for (std::size_t k = 0;
                 k < gradient.cols();
                 ++k)
            {
                sum +=
                    gradient(i, k)
                    * weights_(j, k);
            }

            inputGradient_(i, j) = sum;
        }
    }

    // =====================================================
    // 2. Calculate weight gradient
    //
    // dW = X^T * dZ
    // =====================================================

    weightGradient_.fill(0.0f);

    for (std::size_t i = 0;
         i < weights_.rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < weights_.cols();
             ++j)
        {
            float sum = 0.0f;

            for (std::size_t k = 0;
                 k < input_.rows();
                 ++k)
            {
                sum +=
                    input_(k, i)
                    * gradient(k, j);
            }

            weightGradient_(i, j) = sum;
        }
    }

    // =====================================================
    // 3. Calculate bias gradient
    //
    // dB = sum(dZ)
    // =====================================================

    biasGradient_.fill(0.0f);

    for (std::size_t j = 0;
         j < bias_.cols();
         ++j)
    {
        for (std::size_t i = 0;
             i < gradient.rows();
             ++i)
        {
            biasGradient_(0, j) +=
                gradient(i, j);
        }
    }

    // =====================================================
    // 4. Update weights
    //
    // W = W - learningRate * dW
    // =====================================================

    for (std::size_t i = 0;
         i < weights_.rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < weights_.cols();
             ++j)
        {
            weights_(i, j) -=
                learningRate
                * weightGradient_(i, j);
        }
    }

    // =====================================================
    // 5. Update biases
    //
    // B = B - learningRate * dB
    // =====================================================

    for (std::size_t j = 0;
         j < bias_.cols();
         ++j)
    {
        bias_(0, j) -=
            learningRate
            * biasGradient_(0, j);
    }
}

const Matrix& DenseLayer::weights() const
{
    return weights_;
}

const Matrix& DenseLayer::bias() const
{
    return bias_;
}

const Matrix& DenseLayer::output() const
{
    return output_;
}

const Matrix& DenseLayer::inputGradient() const
{
    return inputGradient_;
}

const Matrix& DenseLayer::weightGradient() const
{
    return weightGradient_;
}

const Matrix& DenseLayer::biasGradient() const
{
    return biasGradient_;
}

Matrix& DenseLayer::mutableWeights()
{
    return weights_;
}