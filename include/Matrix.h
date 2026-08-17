#pragma once

#include <cstddef>
#include <vector>

class Matrix
{
private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<float> data_;

public:

float* data();

const float* data() const;

    Matrix(std::size_t rows, std::size_t cols);

    Matrix(std::size_t rows,
           std::size_t cols,
           float value);

    float& operator()(std::size_t row,
                      std::size_t col);

    const float& operator()(std::size_t row,
                            std::size_t col) const;

    std::size_t rows() const;
    std::size_t cols() const;

    void fill(float value);

    void randomize(float min = -1.0f,
                   float max = 1.0f);

    Matrix operator+(const Matrix& other) const;

    Matrix operator-(const Matrix& other) const;

    Matrix operator*(const Matrix& other) const;

    void print() const;
};