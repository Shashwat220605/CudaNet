#pragma once

#include "CudaMatrix.h"

float cudaCrossEntropyLoss(
    const CudaMatrix& probabilities,
    const CudaMatrix& targets
);