#pragma once

#include "DenseLayer.h"
#include "Matrix.h"

class Trainer
{
public:
    static float trainStep(
        DenseLayer& layer1,
        DenseLayer& layer2,
        const Matrix& input,
        const Matrix& target,
        float learningRate
    );
};