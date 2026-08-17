#include "Softmax.h"

#include <cmath>
#include <stdexcept>

Matrix softmax(const Matrix& input)
{
    Matrix result(input.rows(), input.cols());

    for (std::size_t row = 0; row < input.rows(); ++row)
    {
        float maxValue = input(row, 0);

        for (std::size_t col = 1; col < input.cols(); ++col)
        {
            if (input(row, col) > maxValue)
            {
                maxValue = input(row, col);
            }
        }

        float sum = 0.0f;

        for (std::size_t col = 0; col < input.cols(); ++col)
        {
            float value =
                std::exp(input(row, col) - maxValue);

            result(row, col) = value;
            sum += value;
        }

        for (std::size_t col = 0; col < input.cols(); ++col)
        {
            result(row, col) /= sum;
        }
    }

    return result;
}