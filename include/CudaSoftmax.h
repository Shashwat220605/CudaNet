#pragma once

#include "CudaMatrix.h"

CudaMatrix cudaSoftmaxGpu(
    const CudaMatrix& input
);

// Gradient of Softmax + Cross Entropy.
//
// For:
//     loss = CrossEntropy(Softmax(logits), targets)
//
// the gradient simplifies to:
//
//     dLogits = probabilities - targets
//
CudaMatrix cudaSoftmaxCrossEntropyGradientGpu(
    const CudaMatrix& probabilities,
    const CudaMatrix& targets
);