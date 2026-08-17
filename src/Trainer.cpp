#include "Trainer.h"

#include "Activation.h"
#include "Loss.h"
#include "Softmax.h"

#include <stdexcept>

float Trainer::trainStep(
    DenseLayer& layer1,
    DenseLayer& layer2,
    const Matrix& input,
    const Matrix& target,
    float learningRate)
{
    // =========================
    // Forward pass
    // =========================

    Matrix z1 = layer1.forward(input);

    Matrix a1 = relu(z1);

    Matrix z2 = layer2.forward(a1);

    Matrix probabilities = softmax(z2);

    // =========================
    // Calculate loss
    // =========================

    float loss =
        crossEntropyLoss(
            probabilities,
            target
        );

    // =========================
    // Softmax + Cross Entropy
    //
    // dZ2 = probabilities - target
    // =========================

    Matrix gradient2(
        probabilities.rows(),
        probabilities.cols()
    );

    for (std::size_t i = 0;
         i < probabilities.rows();
         ++i)
    {
        for (std::size_t j = 0;
             j < probabilities.cols();
             ++j)
        {
            gradient2(i, j) =
                probabilities(i, j)
                - target(i, j);
        }
    }

    // =========================
    // Backprop layer 2
    // =========================

    layer2.backward(
        gradient2,
        learningRate
    );

    // Gradient flowing into ReLU
    Matrix reluGradient =
        layer2.inputGradient();

    // =========================
    // Backprop through ReLU
    // =========================

    Matrix gradient1 =
        reluBackward(
            z1,
            reluGradient
        );

    // =========================
    // Backprop layer 1
    // =========================

    layer1.backward(
        gradient1,
        learningRate
    );

    return loss;
}