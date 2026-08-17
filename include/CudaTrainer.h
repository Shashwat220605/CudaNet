#pragma once

#include "CudaDenseLayer.h"
#include "CudaMatrix.h"

class CudaTrainer
{
public:

    static float trainStep(
        CudaDenseLayer& layer1,
        CudaDenseLayer& layer2,
        const CudaMatrix& input,
        const CudaMatrix& target,
        float learningRate
    );
};