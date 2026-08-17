#pragma once

#include "Matrix.h"

Matrix relu(const Matrix& input);

Matrix reluBackward(
    const Matrix& input,
    const Matrix& gradient
);