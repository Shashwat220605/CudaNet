# CudaNet

CudaNet is a CUDA-accelerated neural network implementation written in C++ and NVIDIA CUDA.

The project implements the fundamental components of a neural network from scratch, including matrix operations, dense layers, ReLU activation, Softmax, cross-entropy loss, forward propagation, backpropagation, and GPU-based training.

---

## Project Architecture

```text
                    CudaNet
                       │
          ┌────────────┴────────────┐
          │                         │
       CPU Path                  CUDA Path
          │                         │
      Matrix                    CudaMatrix
          │                         │
    DenseLayer              CUDA Matrix Multiply
          │                         │
       ReLU                 Tiled Matrix Multiply
          │                         │
      Softmax                 CudaDenseLayer
          │                         │
   Cross Entropy                 CudaReLU
          │                         │
       Trainer                 CudaSoftmax
                                    │
                                 CudaLoss
                                    │
                               CudaTrainer
```

---

## Neural Network

The demonstration network is:

```text
Input
  │
  │ 3 features
  ▼
Dense Layer
3 → 8
  │
  ▼
ReLU
  │
  ▼
Dense Layer
8 → 2
  │
  ▼
Softmax
  │
  ▼
Class probabilities
```

The network is trained using cross-entropy loss.

---

## Forward Propagation

For the first dense layer:

```text
Z1 = XW1 + B1
```

ReLU activation:

```text
A1 = ReLU(Z1)
```

Second dense layer:

```text
Z2 = A1W2 + B2
```

Softmax converts the output logits into probabilities:

```text
P_i = exp(Z_i) / Σ exp(Z_j)
```

The resulting probabilities are used by cross-entropy loss.

---

## Backpropagation

For Softmax combined with cross-entropy:

```text
dZ2 = P - Y
```

where:

```text
P = predicted probabilities
Y = target one-hot vector
```

The gradient is propagated through the second Dense layer, then through ReLU, and finally through the first Dense layer.

Weights are updated using:

```text
W = W - learningRate × dW
```

Biases are updated using:

```text
B = B - learningRate × dB
```

---

## CUDA Implementation

CudaNet contains GPU implementations of the major neural-network operations.

### CUDA Matrix Operations

The project contains CUDA matrix multiplication implementations, including a tiled matrix multiplication implementation.

Tiled matrix multiplication allows CUDA threads to cooperate when loading matrix data and improves reuse through shared memory.

### GPU Dense Layer

The CUDA Dense layer performs:

```text
Y = XW + B
```

on the GPU.

It also calculates gradients required for backpropagation.

### GPU ReLU

The ReLU operation is:

```text
ReLU(x) = max(0, x)
```

The CUDA implementation performs this operation in parallel across matrix elements.

### GPU Softmax

The CUDA Softmax implementation converts logits into probabilities.

Numerical stability is important because exponential values can become extremely large or small.

### GPU Cross Entropy

Cross-entropy measures the difference between the predicted probability distribution and the target distribution.

The GPU loss implementation is used during training.

---

## Training Pipeline

The GPU training process is:

```text
Input Batch
     │
     ▼
CudaDenseLayer
     │
     ▼
CudaReLU
     │
     ▼
CudaDenseLayer
     │
     ▼
CudaSoftmax
     │
     ▼
Cross Entropy
     │
     ▼
Loss
     │
     ▼
Backpropagation
     │
     ▼
Gradient Calculation
     │
     ▼
Weight Updates
```

Training is performed using mini-batches.

---

## Dataset

The project currently uses a generated synthetic classification dataset.

The dataset contains:

```text
3 input features
2 classes
```

The data is generated deterministically using a fixed random seed.

The dataset is divided into:

```text
Training Set
    │
    └── 80%

Testing Set
    │
    └── 20%
```

This allows the training process and classification accuracy to be evaluated without requiring an external dataset.

---

## Batch Processing

Training data is divided into mini-batches.

The demonstration uses:

```text
Batch size = 32
```

Each batch is converted into GPU matrices before being passed to the CUDA trainer.

Target labels are represented using one-hot encoding:

```text
Class 0 → [1, 0]

Class 1 → [0, 1]
```

---

## GPU Trainer

`CudaTrainer` coordinates the complete training step.

Its responsibilities include:

1. Forward propagation
2. Softmax calculation
3. Cross-entropy loss
4. Softmax + cross-entropy gradient
5. Dense layer backpropagation
6. ReLU backpropagation
7. Weight updates

This keeps the training logic separate from the individual CUDA kernels.

---

## Testing

The project was developed and tested incrementally.

The following components were tested:

```text
CPU Matrix operations              ✓
CPU Dense layer                    ✓
CPU activation functions           ✓
CPU Softmax                        ✓
CPU Cross Entropy                  ✓
CPU backpropagation                ✓

CUDA Matrix operations             ✓
CUDA matrix multiplication         ✓
CUDA tiled multiplication          ✓
CUDA Dense forward                 ✓
CUDA Dense backward                ✓
CUDA weight gradients              ✓
CUDA bias gradients                ✓
CUDA ReLU                          ✓
CUDA Softmax                       ✓
CUDA Cross Entropy                 ✓
Softmax + CE gradient              ✓
Complete GPU training step         ✓
```

A complete GPU training test successfully demonstrated that the training loss decreases during optimization.

---

## CPU vs GPU Benchmark

CudaNet contains CPU and GPU benchmark implementations.

The benchmark uses the same general training workload for both implementations.

The benchmark network is:

```text
3 → 128 → 2
```

The benchmark uses a larger synthetic dataset than the demonstration training run.

The benchmark measures:

```text
CPU training time
GPU training time
GPU speedup
```

CUDA synchronization is used around GPU timing so asynchronous CUDA work is included in the measured GPU execution time.

### Benchmark limitation

The final benchmark executable could not be executed on the development Windows installation because Windows Smart App Control / Code Integrity blocked the locally compiled unsigned executable.

The CUDA project itself successfully compiles, and the GPU training pipeline was previously executed successfully.

Therefore, no final CPU-vs-GPU speedup number is claimed in this README.

---

## Build Requirements

The project requires:

- Windows
- CMake
- Visual Studio / MSVC
- NVIDIA CUDA Toolkit
- NVIDIA GPU supporting the configured CUDA architecture

---

## Building the Project

From the project directory:

```bash
cmake -S . -B build
```

Build the Release configuration:

```bash
cmake --build build --config Release
```

The executable is generated at:

```text
build/Release/CudaNet.exe
```

---

## Project Structure

```text
CudaNet/
│
├── CMakeLists.txt
├── README.md
│
├── include/
│   ├── Activation.h
│   ├── CudaBatch.h
│   ├── CudaDenseLayer.h
│   ├── CudaLoss.h
│   ├── CudaMatrix.h
│   ├── CudaMatrixMultiply.h
│   ├── CudaReLU.h
│   ├── CudaSoftmax.h
│   ├── CudaTiledMatrixMultiply.h
│   ├── CudaTrainer.h
│   ├── CudaUtils.h
│   ├── Dataset.h
│   ├── DenseLayer.h
│   ├── GradientCheck.h
│   ├── Loss.h
│   ├── Matrix.h
│   ├── Softmax.h
│   └── Trainer.h
│
├── src/
│   ├── Activation.cpp
│   ├── CudaBatch.cpp
│   ├── CudaTrainer.cpp
│   ├── Dataset.cpp
│   ├── DenseLayer.cpp
│   ├── GradientCheck.cpp
│   ├── Loss.cpp
│   ├── main.cu
│   ├── Matrix.cpp
│   ├── Softmax.cpp
│   └── Trainer.cpp
│
└── cuda/
    ├── CudaLoss.cu
    ├── CudaMatrix.cu
    ├── CudaMatrixMultiply.cu
    ├── CudaReLU.cu
    ├── CudaSoftmax.cu
    └── CudaTiledMatrixMultiply.cu
```

---

## CUDA Architecture

The project is configured for CUDA architecture 86:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)
```

This targets NVIDIA GPUs using compute capability 8.6.

---

## Future Improvements

Possible extensions include:

- MNIST dataset support
- Larger neural networks
- Multiple hidden layers
- CUDA streams
- Pinned host memory
- Better GPU memory reuse
- Batch processing directly on GPU
- CUDA event-based benchmarking
- Mixed precision training
- cuBLAS integration
- Model saving/loading
- GPU inference mode

---

## ⚠️ Windows Security Notice

The prebuilt executable may be blocked from running on some Windows systems due to Windows Security or antivirus protection. This is an environment/security restriction and does **not** indicate a problem with the CudaNet implementation itself.

The complete source code, CUDA kernels, headers, CMake configuration, and project structure are included in this repository. The project can be inspected, compiled, and tested in a suitable CUDA-enabled development environment.

For security reasons, users should **not disable Windows Security** simply to run the executable. Instead, building the project from source in a trusted development environment is recommended.

## 🚀 Benefits of CudaNet

CudaNet demonstrates how neural-network operations can be implemented and accelerated using **NVIDIA CUDA and GPU parallel processing**.

Key benefits include:

* **GPU Acceleration:** Computationally intensive operations can be distributed across thousands of GPU threads.
* **Parallel Computation:** Matrix operations, activation functions, loss calculations, and other workloads can take advantage of massive parallelism.
* **Efficient Neural Network Operations:** CUDA kernels allow important neural-network computations to execute directly on the GPU.
* **Learning and Research:** The project provides a practical implementation for understanding how neural networks interact with GPU hardware at a lower level.
* **C++ and CUDA Integration:** Combines the performance and control of C++ with NVIDIA's CUDA programming model.
* **Modular Architecture:** Components such as matrices, layers, activation functions, loss functions, datasets, and training utilities are separated into reusable modules.
* **Reduced CPU Workload:** Suitable computations can be moved from the CPU to the GPU, allowing the CPU to focus on other tasks.

## 🧠 Theory Behind the Project

CudaNet is based on the idea of **parallel computing for neural networks**.

Neural networks perform a large number of mathematical operations, particularly matrix multiplication. A CPU typically executes a relatively small number of operations simultaneously, while a GPU is designed to perform a very large number of similar operations in parallel.

CudaNet uses CUDA kernels to take advantage of this architecture.

A simplified workflow is:

**Input Data → Matrix Operations → Neural Network Layers → Activation → Loss Calculation → Backpropagation → Weight Updates**

During these stages, computationally expensive operations can be executed on the GPU.

### Why CUDA?

CUDA provides a programming model that allows developers to write functions called **kernels** that execute across many GPU threads.

For example, a matrix multiplication operation can be divided into many smaller calculations:

```text
GPU
 ├── Thread 1 → Calculate one part
 ├── Thread 2 → Calculate another part
 ├── Thread 3 → Calculate another part
 ├── Thread 4 → Calculate another part
 ├── ...
 └── Thousands of threads → Work in parallel
```

This makes GPUs particularly well suited for workloads commonly found in machine learning.

## ⭐ Advantages

CudaNet provides several advantages as a learning and experimental neural-network framework:

1. **Low-level GPU understanding**
   Demonstrates what happens underneath high-level machine-learning frameworks.

2. **Performance-oriented design**
   Uses GPU parallelism for computationally intensive operations.

3. **Modular implementation**
   Neural-network components are separated into reusable classes and modules.

4. **Hands-on CUDA experience**
   Provides practical experience with CUDA kernels, GPU memory, threads, and parallel computation.

5. **Foundation for further development**
   The architecture can be extended with additional layers, optimizers, activation functions, datasets, and GPU optimizations.

6. **Educational value**
   Helps bridge the gap between neural-network theory and the underlying hardware-level computation.

> **Note:** Actual performance improvements depend on the GPU, workload size, memory transfers, kernel implementation, and optimization level. CudaNet is primarily intended as a practical CUDA-based neural-network implementation and learning project.


## Conclusion

CudaNet demonstrates how a basic neural network can be implemented from scratch and accelerated using NVIDIA CUDA.

The project covers both the mathematical foundations of neural-network training and the practical implementation of GPU parallelism.

The complete GPU training pipeline is:

```text
Data
 ↓
GPU Forward Pass
 ↓
Loss
 ↓
GPU Backpropagation
 ↓
Gradient Updates
 ↓
Learning
```

CudaNet serves as both a neural-network implementation and a practical CUDA learning project.
