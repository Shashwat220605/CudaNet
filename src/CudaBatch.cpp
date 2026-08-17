#include "CudaBatch.h"

#include "Matrix.h"

CudaMatrix CudaBatch::createInput(
    const std::vector<Sample>& samples,
    std::size_t start,
    std::size_t batchSize)
{
    Matrix hostInput(
        batchSize,
        3
    );

    for (std::size_t i = 0;
         i < batchSize;
         ++i)
    {
        const Sample& sample =
            samples[start + i];

        hostInput(i, 0) = sample.x;
        hostInput(i, 1) = sample.y;
        hostInput(i, 2) = sample.z;
    }

    CudaMatrix gpuInput(
        batchSize,
        3
    );

    gpuInput.copyFromHost(hostInput);

    return gpuInput;
}


CudaMatrix CudaBatch::createTarget(
    const std::vector<Sample>& samples,
    std::size_t start,
    std::size_t batchSize)
{
    Matrix hostTarget(
        batchSize,
        2
    );

    for (std::size_t i = 0;
         i < batchSize;
         ++i)
    {
        int label =
            samples[start + i].label;

        hostTarget(i, 0) =
            label == 0 ? 1.0f : 0.0f;

        hostTarget(i, 1) =
            label == 1 ? 1.0f : 0.0f;
    }

    CudaMatrix gpuTarget(
        batchSize,
        2
    );

    gpuTarget.copyFromHost(hostTarget);

    return gpuTarget;
}
