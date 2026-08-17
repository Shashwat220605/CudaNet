#pragma once

#include "Matrix.h"

class DenseLayer
{
private:
    Matrix weights_;
    Matrix bias_;

    Matrix input_;
    Matrix output_;

    // Gradient with respect to the input
    Matrix inputGradient_;

    // Gradients calculated during backpropagation
    Matrix weightGradient_;
    Matrix biasGradient_;

public:
    DenseLayer(
        std::size_t inputSize,
        std::size_t outputSize
    );

    // Forward pass
    Matrix forward(const Matrix& input);

    // Backward pass
    void backward(
        const Matrix& gradient,
        float learningRate
    );

    // Accessors
    const Matrix& weights() const;
    Matrix& mutableWeights();
    const Matrix& bias() const;
    const Matrix& output() const;

    // Gradient accessors
    const Matrix& inputGradient() const;
    const Matrix& weightGradient() const;
    const Matrix& biasGradient() const;
};