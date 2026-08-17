#include "Matrix.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows),
      cols_(cols),
      data_(rows * cols, 0.0f)
{
}

Matrix::Matrix(std::size_t rows,
               std::size_t cols,
               float value)
    : rows_(rows),
      cols_(cols),
      data_(rows * cols, value)
{
}

float& Matrix::operator()(std::size_t row,
                          std::size_t col)
{
    return data_[row * cols_ + col];
}

const float& Matrix::operator()(std::size_t row,
                                std::size_t col) const
{
    return data_[row * cols_ + col];
}

std::size_t Matrix::rows() const
{
    return rows_;
}

std::size_t Matrix::cols() const
{
    return cols_;
}

void Matrix::fill(float value)
{
    std::fill(data_.begin(), data_.end(), value);
}

void Matrix::randomize(float min, float max)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(min, max);

    for (float& value : data_)
    {
        value = distribution(generator);
    }
}

Matrix Matrix::operator+(const Matrix& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument(
            "Matrix dimensions must match for addition."
        );
    }

    Matrix result(rows_, cols_);

    for (std::size_t i = 0; i < rows_; ++i)
    {
        for (std::size_t j = 0; j < cols_; ++j)
        {
            result(i, j) = (*this)(i, j) + other(i, j);
        }
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument(
            "Matrix dimensions must match for subtraction."
        );
    }

    Matrix result(rows_, cols_);

    for (std::size_t i = 0; i < rows_; ++i)
    {
        for (std::size_t j = 0; j < cols_; ++j)
        {
            result(i, j) = (*this)(i, j) - other(i, j);
        }
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (cols_ != other.rows_)
    {
        throw std::invalid_argument(
            "Matrix dimensions are incompatible for multiplication."
        );
    }

    Matrix result(rows_, other.cols_);

    for (std::size_t i = 0; i < rows_; ++i)
    {
        for (std::size_t j = 0; j < other.cols_; ++j)
        {
            float sum = 0.0f;

            for (std::size_t k = 0; k < cols_; ++k)
            {
                sum += (*this)(i, k) * other(k, j);
            }

            result(i, j) = sum;
        }
    }

    return result;
}

void Matrix::print() const
{
    for (std::size_t i = 0; i < rows_; ++i)
    {
        for (std::size_t j = 0; j < cols_; ++j)
        {
            std::cout << (*this)(i, j) << " ";
        }

        std::cout << '\n';
    }
}

float* Matrix::data()
{
    return data_.data();
}

const float* Matrix::data() const
{
    return data_.data();
}