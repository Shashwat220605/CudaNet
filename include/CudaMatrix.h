#pragma once

#include "Matrix.h"

#include <cstddef>

class CudaMatrix
{
private:
    float* deviceData_;
    std::size_t rows_;
    std::size_t cols_;

public:
    CudaMatrix(
        std::size_t rows,
        std::size_t cols
    );

    ~CudaMatrix();

    CudaMatrix(const CudaMatrix&) = delete;
    CudaMatrix& operator=(const CudaMatrix&) = delete;

    CudaMatrix(CudaMatrix&& other) noexcept;
    CudaMatrix& operator=(CudaMatrix&& other) noexcept;

    std::size_t rows() const;
    std::size_t cols() const;

    float* data();
    const float* data() const;

    void copyFromHost(const Matrix& matrix);

    Matrix copyToHost() const;

    void fill(float value);
};