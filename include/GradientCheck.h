#pragma once

#include "DenseLayer.h"

bool checkDenseLayerGradient(
    DenseLayer& layer,
    const Matrix& input,
    const Matrix& gradient,
    float epsilon = 0.0001f,
    float tolerance = 0.001f
);