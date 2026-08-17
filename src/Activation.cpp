#include "Activation.h"

#include <stdexcept>

Matrix relu(const Matrix& input)
{
    Matrix result(
        input.rows(),
        input.cols()
    );

    for (std::size_t i = 0;
         i < input.rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < input.cols();
             ++j)
        {
            float value = input(i, j);

            result(i, j) =
                value > 0.0f
                ? value
                : 0.0f;
        }
    }

    return result;
}

Matrix reluBackward(
    const Matrix& input,
    const Matrix& gradient)
{
    if (input.rows() != gradient.rows() ||
        input.cols() != gradient.cols())
    {
        throw std::invalid_argument(
            "ReLU input and gradient dimensions "
            "must match."
        );
    }

    Matrix result(
        input.rows(),
        input.cols()
    );

    for (std::size_t i = 0;
         i < input.rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < input.cols();
             ++j)
        {
            if (input(i, j) > 0.0f)
            {
                result(i, j) =
                    gradient(i, j);
            }
            else
            {
                result(i, j) = 0.0f;
            }
        }
    }

    return result;
}