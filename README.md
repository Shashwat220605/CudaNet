# CudaNet

CudaNet is a CUDA-accelerated neural network implementation written in C++ and NVIDIA CUDA.

The project implements neural-network components from scratch, including matrix operations, dense layers, ReLU activation, Softmax, cross-entropy loss, forward propagation, backpropagation, and GPU-based training.

## Project Architecture

```text
CudaNet
├── CPU Path
│   ├── Matrix
│   ├── DenseLayer
│   ├── ReLU
│   ├── Softmax
│   ├── Cross Entropy
│   └── Trainer
│
└── CUDA Path
    ├── CudaMatrix
    ├── CUDA Matrix Multiply
    ├── Tiled Matrix Multiply
    ├── CudaDenseLayer
    ├── CudaReLU
    ├── CudaSoftmax
    ├── CudaLoss
    └── CudaTrainer
```

## Neural Network

The demonstration network is:

```text
Input (3 features)
        │
        ▼
Dense 3 → 8
        │
        ▼
      ReLU
        │
        ▼
Dense 8 → 2
        │
        ▼
    Softmax
        │
        ▼
Class probabilities
```

The network is trained using cross-entropy loss.

## Training Pipeline

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
Backpropagation
     │
     ▼
Gradient Calculation
     │
     ▼
Weight Updates
```

Training uses mini-batches with a demonstration batch size of 32.

## CUDA Implementation

CudaNet contains GPU implementations of the major neural-network operations.

### Matrix Operations

The project includes CUDA matrix multiplication, including a tiled implementation that uses shared memory to improve data reuse.

### Dense Layer

The CUDA Dense layer performs:

```text
Y = XW + B
```

on the GPU and calculates the gradients required for backpropagation.

### ReLU

```text
ReLU(x) = max(0, x)
```

The CUDA implementation evaluates the operation in parallel across matrix elements.

### Softmax

CUDA Softmax converts logits into probabilities while accounting for numerical stability during exponentiation.

### Cross Entropy

The GPU loss implementation measures the difference between predicted probabilities and target distributions.

## Dataset

The demonstration uses a deterministic synthetic classification dataset with:

```text
3 input features
2 classes
80% training data
20% testing data
```

Target labels use one-hot encoding:

```text
Class 0 → [1, 0]
Class 1 → [0, 1]
```

## Testing

The project was developed and tested incrementally. Components tested include:

- CPU matrix operations
- CPU dense layers and activations
- CPU Softmax and cross-entropy
- CPU backpropagation
- CUDA matrix operations
- CUDA matrix multiplication and tiled multiplication
- CUDA Dense forward/backward passes
- CUDA ReLU
- CUDA Softmax
- CUDA cross-entropy
- Softmax + cross-entropy gradient
- Complete GPU training step

The complete GPU training test demonstrated that training loss decreases during optimization.

## CPU vs GPU Benchmark

CudaNet includes CPU and GPU benchmark implementations using the same general training workload.

The benchmark network is:

```text
3 → 128 → 2
```

It measures:

```text
CPU training time
GPU training time
GPU speedup
```

CUDA synchronization is used around GPU timing so asynchronous CUDA work is included in the measurement.

### Benchmark limitation

A final benchmark executable could not be executed on the development Windows installation because Windows Smart App Control / Code Integrity blocked the locally compiled unsigned executable.

The project compiles successfully, and the GPU training pipeline was previously executed successfully. Therefore, this README does not claim a final CPU-vs-GPU speedup number.

## Build Requirements

- Windows
- CMake
- Visual Studio / MSVC
- NVIDIA CUDA Toolkit
- NVIDIA GPU supporting the configured CUDA architecture

## Building

From the project directory:

```bash
cmake -S . -B build
cmake --build build --config Release
```

The executable is generated at:

```text
build/Release/CudaNet.exe
```

## Project Structure

```text
CudaNet/
├── CMakeLists.txt
├── README.md
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
└── cuda/
    ├── CudaLoss.cu
    ├── CudaMatrix.cu
    ├── CudaMatrixMultiply.cu
    ├── CudaReLU.cu
    ├── CudaSoftmax.cu
    └── CudaTiledMatrixMultiply.cu
```

## CUDA Architecture

The project is configured for CUDA architecture 86:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)
```

This targets NVIDIA GPUs using compute capability 8.6.

## Future Improvements

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

## Windows Security Notice

A prebuilt executable may be blocked on some Windows systems by Windows Security or antivirus protection. This is an environment/security restriction and does not indicate a problem with the CudaNet implementation itself.

For security reasons, users should **not disable Windows Security** simply to run the executable. Building the project from source in a trusted CUDA-enabled development environment is recommended.

## Why CudaNet?

CudaNet demonstrates how neural-network operations can be implemented from scratch and accelerated using NVIDIA CUDA and GPU parallel processing.

The project provides hands-on experience with:

- GPU parallel computation
- CUDA kernels and threads
- GPU memory and shared memory
- Matrix operations
- Neural-network forward and backward passes
- C++ and CUDA integration
- Modular neural-network architecture

Actual performance depends on the GPU, workload size, memory transfers, kernel implementation, and optimization level. CudaNet is primarily intended as a practical CUDA-based neural-network implementation and learning project.

## Conclusion

CudaNet demonstrates how a basic neural network can be implemented from scratch and accelerated using NVIDIA CUDA. It combines the mathematical foundations of neural-network training with practical GPU parallelism and provides a foundation for further experimentation with CUDA-based machine learning.
