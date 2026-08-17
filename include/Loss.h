#pragma once

#include "Matrix.h"

float crossEntropyLoss(
    const Matrix& probabilities,
    const Matrix& targets
);