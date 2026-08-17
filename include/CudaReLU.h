#pragma once

#include "Matrix.h"
#include "CudaMatrix.h"

Matrix cudaRelu(
    const Matrix& input,
    float* kernelTimeMs = nullptr
);

CudaMatrix cudaReluGpu(
    const CudaMatrix& input,
    float* kernelTimeMs = nullptr
);

// GPU ReLU backward
CudaMatrix cudaReluBackwardGpu(
    const CudaMatrix& input,
    const CudaMatrix& gradient
);