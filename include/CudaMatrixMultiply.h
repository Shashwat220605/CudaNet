#pragma once

#include "Matrix.h"

Matrix cudaMatrixMultiply(
    const Matrix& A,
    const Matrix& B,
    float* kernelTimeMs = nullptr
);