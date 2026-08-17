#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

inline void cudaCheck(
    cudaError_t error,
    const char* operation)
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation)
            + " failed: "
            + cudaGetErrorString(error)
        );
    }
}