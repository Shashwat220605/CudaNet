#pragma once

#include "CudaMatrix.h"
#include "Dataset.h"

#include <cstddef>
#include <vector>

class CudaBatch
{
public:
    static CudaMatrix createInput(
        const std::vector<Sample>& samples,
        std::size_t start,
        std::size_t batchSize
    );

    static CudaMatrix createTarget(
        const std::vector<Sample>& samples,
        std::size_t start,
        std::size_t batchSize
    );
};