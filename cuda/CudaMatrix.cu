#include "CudaMatrix.h"
#include "CudaUtils.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <utility>

CudaMatrix::CudaMatrix(
    std::size_t rows,
    std::size_t cols)
    : deviceData_(nullptr),
      rows_(rows),
      cols_(cols)
{
    std::size_t bytes =
        rows_ * cols_ * sizeof(float);

    cudaCheck(
        cudaMalloc(
            &deviceData_,
            bytes
        ),
        "CudaMatrix cudaMalloc"
    );
}

CudaMatrix::~CudaMatrix()
{
    if (deviceData_ != nullptr)
    {
        cudaFree(deviceData_);
    }
}

CudaMatrix::CudaMatrix(
    CudaMatrix&& other) noexcept
    : deviceData_(other.deviceData_),
      rows_(other.rows_),
      cols_(other.cols_)
{
    other.deviceData_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
}

CudaMatrix& CudaMatrix::operator=(
    CudaMatrix&& other) noexcept
{
    if (this != &other)
    {
        if (deviceData_ != nullptr)
        {
            cudaFree(deviceData_);
        }

        deviceData_ = other.deviceData_;
        rows_ = other.rows_;
        cols_ = other.cols_;

        other.deviceData_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }

    return *this;
}

std::size_t CudaMatrix::rows() const
{
    return rows_;
}

std::size_t CudaMatrix::cols() const
{
    return cols_;
}

float* CudaMatrix::data()
{
    return deviceData_;
}

const float* CudaMatrix::data() const
{
    return deviceData_;
}

void CudaMatrix::copyFromHost(
    const Matrix& matrix)
{
    if (matrix.rows() != rows_ ||
        matrix.cols() != cols_)
    {
        throw std::invalid_argument(
            "CudaMatrix dimensions do not match."
        );
    }

    std::size_t bytes =
        rows_ * cols_ * sizeof(float);

    cudaCheck(
        cudaMemcpy(
            deviceData_,
            matrix.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "CudaMatrix Host to Device copy"
    );
}

Matrix CudaMatrix::copyToHost() const
{
    Matrix result(rows_, cols_);

    std::size_t bytes =
        rows_ * cols_ * sizeof(float);

    cudaCheck(
        cudaMemcpy(
            result.data(),
            deviceData_,
            bytes,
            cudaMemcpyDeviceToHost
        ),
        "CudaMatrix Device to Host copy"
    );

    return result;
}

__global__ void fillKernel(
    float* data,
    int totalElements,
    float value)
{
    int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (index < totalElements)
    {
        data[index] = value;
    }
}

void CudaMatrix::fill(float value)
{
    int totalElements =
        static_cast<int>(rows_ * cols_);

    constexpr int threads = 256;

    int blocks =
        (totalElements + threads - 1)
        / threads;

    fillKernel<<<blocks, threads>>>(
        deviceData_,
        totalElements,
        value
    );

    cudaCheck(
        cudaGetLastError(),
        "CudaMatrix fill kernel"
    );

    cudaCheck(
        cudaDeviceSynchronize(),
        "CudaMatrix fill synchronization"
    );
}