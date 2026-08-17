#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "CudaBatch.h"
#include "CudaDenseLayer.h"
#include "CudaReLU.h"
#include "CudaSoftmax.h"
#include "CudaTrainer.h"

#include "DenseLayer.h"
#include "Dataset.h"
#include "Trainer.h"


// ============================================================
// Evaluate GPU classification accuracy
// ============================================================

float evaluateAccuracy(
    CudaDenseLayer& layer1,
    CudaDenseLayer& layer2,
    const std::vector<Sample>& dataset)
{
    std::size_t correct = 0;

    for (const Sample& sample : dataset)
    {
        Matrix hostInput(1, 3);

        hostInput(0, 0) = sample.x;
        hostInput(0, 1) = sample.y;
        hostInput(0, 2) = sample.z;

        CudaMatrix input(1, 3);

        input.copyFromHost(hostInput);

        // Forward pass

        CudaMatrix z1 =
            layer1.forwardGpu(input);

        CudaMatrix a1 =
            cudaReluGpu(z1);

        CudaMatrix z2 =
            layer2.forwardGpu(a1);

        CudaMatrix probabilities =
            cudaSoftmaxGpu(z2);

        Matrix result =
            probabilities.copyToHost();

        int predicted =
            result(0, 0) > result(0, 1)
                ? 0
                : 1;

        if (predicted == sample.label)
        {
            ++correct;
        }
    }

    return static_cast<float>(correct)
         / static_cast<float>(dataset.size());
}


// ============================================================
// CPU benchmark
//
// Network:
//     3 -> 128 -> 2
//
// Same dataset and training configuration as GPU benchmark.
// ============================================================

double benchmarkCpu(
    const std::vector<Sample>& trainSet,
    std::size_t batchSize,
    int epochs,
    float learningRate)
{
    DenseLayer layer1(3, 128);
    DenseLayer layer2(128, 2);

    auto start =
        std::chrono::high_resolution_clock::now();

    for (int epoch = 0;
         epoch < epochs;
         ++epoch)
    {
        for (std::size_t batchStart = 0;
             batchStart < trainSet.size();
             batchStart += batchSize)
        {
            std::size_t currentBatchSize =
                std::min(
                    batchSize,
                    trainSet.size() - batchStart
                );

            Matrix input(
                currentBatchSize,
                3
            );

            Matrix target(
                currentBatchSize,
                2
            );

            for (std::size_t i = 0;
                 i < currentBatchSize;
                 ++i)
            {
                const Sample& sample =
                    trainSet[batchStart + i];

                input(i, 0) = sample.x;
                input(i, 1) = sample.y;
                input(i, 2) = sample.z;

                target(i, 0) =
                    sample.label == 0
                        ? 1.0f
                        : 0.0f;

                target(i, 1) =
                    sample.label == 1
                        ? 1.0f
                        : 0.0f;
            }

            Trainer::trainStep(
                layer1,
                layer2,
                input,
                target,
                learningRate
            );
        }
    }

    auto end =
        std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}


// ============================================================
// GPU benchmark
//
// Network:
//     3 -> 128 -> 2
//
// Same dataset and training configuration as CPU benchmark.
// ============================================================

double benchmarkGpu(
    const std::vector<Sample>& trainSet,
    std::size_t batchSize,
    int epochs,
    float learningRate)
{
    CudaDenseLayer layer1(3, 128);
    CudaDenseLayer layer2(128, 2);

    // Synchronize before starting timer so previous
    // CUDA work is not included in the measurement.
    cudaDeviceSynchronize();

    auto start =
        std::chrono::high_resolution_clock::now();

    for (int epoch = 0;
         epoch < epochs;
         ++epoch)
    {
        for (std::size_t batchStart = 0;
             batchStart < trainSet.size();
             batchStart += batchSize)
        {
            std::size_t currentBatchSize =
                std::min(
                    batchSize,
                    trainSet.size() - batchStart
                );

            CudaMatrix input =
                CudaBatch::createInput(
                    trainSet,
                    batchStart,
                    currentBatchSize
                );

            CudaMatrix target =
                CudaBatch::createTarget(
                    trainSet,
                    batchStart,
                    currentBatchSize
                );

            CudaTrainer::trainStep(
                layer1,
                layer2,
                input,
                target,
                learningRate
            );
        }
    }

    // CUDA kernels can execute asynchronously.
    // Synchronize before stopping the timer.
    cudaDeviceSynchronize();

    auto end =
        std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}


// ============================================================
// Main
// ============================================================

int main()
{
    std::cout
        << "========================================\n"
        << "              CudaNet\n"
        << "========================================\n\n";


    // ========================================================
    // 1. Generate synthetic dataset
    // ========================================================

    constexpr std::size_t totalSamples = 1000;

    SyntheticDataset dataset(
        totalSamples,
        42
    );

    std::vector<Sample> trainSet =
        dataset.trainingSet(0.8f);

    std::vector<Sample> testSet =
        dataset.testSet(0.8f);


    std::cout
        << "Dataset created\n"
        << "Total samples : "
        << totalSamples
        << "\n"
        << "Training      : "
        << trainSet.size()
        << "\n"
        << "Testing       : "
        << testSet.size()
        << "\n\n";


    // ========================================================
    // 2. GPU network
    //
    // 3 -> 8 -> 2
    //
    // Dense -> ReLU -> Dense -> Softmax
    // ========================================================

    CudaDenseLayer layer1(
        3,
        8
    );

    CudaDenseLayer layer2(
        8,
        2
    );


    // ========================================================
    // 3. Training configuration
    // ========================================================

    constexpr std::size_t batchSize = 32;

    constexpr int epochs = 50;

    constexpr float learningRate = 0.03f;


    std::cout
        << "Network       : 3 -> 8 -> 2\n"
        << "Batch size    : "
        << batchSize
        << "\n"
        << "Epochs        : "
        << epochs
        << "\n"
        << "Learning rate : "
        << learningRate
        << "\n\n";


    // ========================================================
    // 4. Initial accuracy
    // ========================================================

    float initialAccuracy =
        evaluateAccuracy(
            layer1,
            layer2,
            testSet
        );

    std::cout
        << std::fixed
        << std::setprecision(2);

    std::cout
        << "Initial accuracy: "
        << initialAccuracy * 100.0f
        << "%\n\n";


    // ========================================================
    // 5. GPU training
    // ========================================================

    std::cout
        << "========== GPU TRAINING ==========\n";

    for (int epoch = 0;
         epoch < epochs;
         ++epoch)
    {
        float epochLoss = 0.0f;

        std::size_t batchCount = 0;


        for (std::size_t start = 0;
             start < trainSet.size();
             start += batchSize)
        {
            std::size_t currentBatchSize =
                std::min(
                    batchSize,
                    trainSet.size() - start
                );


            CudaMatrix input =
                CudaBatch::createInput(
                    trainSet,
                    start,
                    currentBatchSize
                );


            CudaMatrix target =
                CudaBatch::createTarget(
                    trainSet,
                    start,
                    currentBatchSize
                );


            float loss =
                CudaTrainer::trainStep(
                    layer1,
                    layer2,
                    input,
                    target,
                    learningRate
                );


            epochLoss += loss;

            ++batchCount;
        }


        epochLoss /=
            static_cast<float>(batchCount);


        if ((epoch + 1) % 5 == 0)
        {
            float accuracy =
                evaluateAccuracy(
                    layer1,
                    layer2,
                    testSet
                );

            std::cout
                << "Epoch "
                << std::setw(2)
                << (epoch + 1)
                << " | Loss: "
                << std::setprecision(6)
                << epochLoss
                << " | Accuracy: "
                << std::setprecision(2)
                << accuracy * 100.0f
                << "%\n";
        }
    }


    // ========================================================
    // 6. Final GPU accuracy
    // ========================================================

    float finalAccuracy =
        evaluateAccuracy(
            layer1,
            layer2,
            testSet
        );


    std::cout
        << "\n========== RESULTS ==========\n";

    std::cout
        << "Initial accuracy: "
        << finalAccuracy * 0.0f
        + initialAccuracy * 100.0f
        << "%\n";

    std::cout
        << "Final accuracy:   "
        << finalAccuracy * 100.0f
        << "%\n";


    if (finalAccuracy > initialAccuracy)
    {
        std::cout
            << "\nCUDA TRAINING TEST: PASSED\n";
    }
    else
    {
        std::cout
            << "\nCUDA TRAINING TEST: FAILED\n";
    }


    // ========================================================
    // 7. Larger CPU vs GPU benchmark
    //
    // Benchmark network:
    //
    //     3 -> 128 -> 2
    //
    // Larger dataset and more work than the demo above.
    // ========================================================

    std::cout
        << "\n========== CPU vs GPU BENCHMARK ==========\n";

    constexpr std::size_t benchmarkSamples =
        10000;

    constexpr std::size_t benchmarkBatchSize =
        64;

    constexpr int benchmarkEpochs =
        50;


    SyntheticDataset benchmarkDataset(
        benchmarkSamples,
        123
    );


    std::vector<Sample> benchmarkTrainSet =
        benchmarkDataset.trainingSet(0.8f);


    std::cout
        << "Benchmark samples : "
        << benchmarkSamples
        << "\n";

    std::cout
        << "Benchmark network : 3 -> 128 -> 2\n";

    std::cout
        << "Benchmark batch   : "
        << benchmarkBatchSize
        << "\n";

    std::cout
        << "Benchmark epochs  : "
        << benchmarkEpochs
        << "\n\n";


    std::cout
        << "Running CPU benchmark...\n";


    double cpuTime =
        benchmarkCpu(
            benchmarkTrainSet,
            benchmarkBatchSize,
            benchmarkEpochs,
            learningRate
        );


    std::cout
        << "Running GPU benchmark...\n";


    double gpuTime =
        benchmarkGpu(
            benchmarkTrainSet,
            benchmarkBatchSize,
            benchmarkEpochs,
            learningRate
        );


    double speedup = 0.0;

    if (gpuTime > 0.0)
    {
        speedup =
            cpuTime / gpuTime;
    }


    std::cout
        << std::fixed
        << std::setprecision(3);


    std::cout
        << "\nCPU training time : "
        << cpuTime
        << " ms\n";


    std::cout
        << "GPU training time : "
        << gpuTime
        << " ms\n";


    std::cout
        << "GPU speedup       : "
        << speedup
        << "x\n";


    std::cout
        << "==========================================\n";


    // ========================================================
    // 8. Final message
    // ========================================================

    std::cout
        << "\n========================================\n"
        << "             CudaNet\n"
        << "          DEMONSTRATION END\n"
        << "========================================\n";


    return 0;
}