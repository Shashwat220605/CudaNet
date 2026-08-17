#include "Loss.h"

#include <cmath>
#include <stdexcept>

float crossEntropyLoss(
    const Matrix& probabilities,
    const Matrix& targets)
{
    if (probabilities.rows() != targets.rows() ||
        probabilities.cols() != targets.cols())
    {
        throw std::invalid_argument(
            "Probability and target dimensions must match."
        );
    }

    constexpr float epsilon = 1e-7f;

    float loss = 0.0f;

    for (std::size_t i = 0;
         i < probabilities.rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < probabilities.cols();
             ++j)
        {
            float probability =
                probabilities(i, j);

            probability =
                std::max(
                    probability,
                    epsilon
                );

            loss -=
                targets(i, j)
                * std::log(probability);
        }
    }

    return loss /
           static_cast<float>(
               probabilities.rows()
           );
}