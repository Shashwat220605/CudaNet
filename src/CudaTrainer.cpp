#include "CudaTrainer.h"

#include "CudaLoss.h"
#include "CudaReLU.h"
#include "CudaSoftmax.h"


float CudaTrainer::trainStep(
    CudaDenseLayer& layer1,
    CudaDenseLayer& layer2,
    const CudaMatrix& input,
    const CudaMatrix& target,
    float learningRate)
{
    // ========================================================
    // Forward pass
    // ========================================================

    // Dense layer 1
    CudaMatrix z1 =
        layer1.forwardGpu(input);

    // ReLU
    CudaMatrix a1 =
        cudaReluGpu(z1);

    // Dense layer 2
    CudaMatrix z2 =
        layer2.forwardGpu(a1);

    // Softmax
    CudaMatrix probabilities =
        cudaSoftmaxGpu(z2);


    // ========================================================
    // Calculate loss
    // ========================================================

    float loss =
        cudaCrossEntropyLoss(
            probabilities,
            target
        );


    // ========================================================
    // Softmax + Cross Entropy gradient
    //
    // dZ2 = probabilities - target
    // ========================================================

    CudaMatrix gradient2 =
        cudaSoftmaxCrossEntropyGradientGpu(
            probabilities,
            target
        );


    // ========================================================
    // Backprop layer 2
    // ========================================================

    CudaMatrix reluGradient =
        layer2.backwardGpu(
            a1,
            gradient2,
            learningRate
        );


    // ========================================================
    // Backprop through ReLU
    //
    // Input to ReLU = z1
    // Gradient      = gradient from layer2
    // ========================================================

    CudaMatrix gradient1 =
        cudaReluBackwardGpu(
            z1,
            reluGradient
        );


    // ========================================================
    // Backprop layer 1
    // ========================================================

    layer1.backwardGpu(
        input,
        gradient1,
        learningRate
    );


    return loss;
}