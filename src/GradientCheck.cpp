#include "GradientCheck.h"

#include <cmath>
#include <iostream>

bool checkDenseLayerGradient(
    DenseLayer& layer,
    const Matrix& input,
    const Matrix& gradient,
    float epsilon,
    float tolerance)
{
    // Run forward once so the layer stores its input/output.
    layer.forward(input);

    // Calculate analytical gradients.
    // Learning rate = 0 so weights do not change.
    layer.backward(gradient, 0.0f);

    const Matrix& analytical =
        layer.weightGradient();

    bool passed = true;

    float maxDifference = 0.0f;

    for (std::size_t i = 0;
         i < layer.weights().rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < layer.weights().cols();
             ++j)
        {
            // Save original weight
            float original =
                layer.mutableWeights()(i, j);

            // W + epsilon
            layer.mutableWeights()(i, j) =
                original + epsilon;

            Matrix plusOutput =
                layer.forward(input);

            float plusValue = 0.0f;

            for (std::size_t r = 0;
                 r < plusOutput.rows();
                 ++r)
            {
                for (std::size_t c = 0;
                     c < plusOutput.cols();
                     ++c)
                {
                    plusValue +=
                        plusOutput(r, c)
                        * gradient(r, c);
                }
            }

            // W - epsilon
            layer.mutableWeights()(i, j) =
                original - epsilon;

            Matrix minusOutput =
                layer.forward(input);

            float minusValue = 0.0f;

            for (std::size_t r = 0;
                 r < minusOutput.rows();
                 ++r)
            {
                for (std::size_t c = 0;
                     c < minusOutput.cols();
                     ++c)
                {
                    minusValue +=
                        minusOutput(r, c)
                        * gradient(r, c);
                }
            }

            // Restore original weight
            layer.mutableWeights()(i, j) =
                original;

            // Numerical derivative
            float numerical =
                (plusValue - minusValue)
                / (2.0f * epsilon);

            float analyticalValue =
                analytical(i, j);

            float difference =
                std::abs(
                    numerical
                    - analyticalValue
                );

            if (difference > maxDifference)
            {
                maxDifference = difference;
            }

            if (difference > tolerance)
            {
                passed = false;

                std::cout
                    << "Gradient mismatch at ["
                    << i << "][" << j << "]\n";

                std::cout
                    << "Analytical: "
                    << analyticalValue
                    << "\n";

                std::cout
                    << "Numerical:   "
                    << numerical
                    << "\n";

                std::cout
                    << "Difference:  "
                    << difference
                    << "\n\n";
            }
        }
    }

    std::cout
        << "Maximum gradient difference: "
        << maxDifference
        << "\n";

    return passed;
}