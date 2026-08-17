#pragma once

#include "Matrix.h"

Matrix cudaTiledMatrixMultiply(
    const Matrix& A,
    const Matrix& B,
    float* kernelTimeMs = nullptr
);