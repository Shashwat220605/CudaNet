cmake_minimum_required(VERSION 3.24)

project(CudaNet
    VERSION 1.0
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(CudaNet
    src/main.cu
    src/Matrix.cpp
    src/DenseLayer.cpp
    src/Activation.cpp
    src/Softmax.cpp
    src/Loss.cpp
    src/GradientCheck.cpp
    src/Trainer.cpp
    src/CudaTrainer.cpp
    src/Dataset.cpp
    src/CudaBatch.cpp
    cuda/CudaMatrixMultiply.cu
    cuda/CudaTiledMatrixMultiply.cu
    cuda/CudaReLU.cu    
    cuda/CudaMatrix.cu
    cuda/CudaSoftmax.cu
    cuda/CudaLoss.cu
)

target_include_directories(CudaNet
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)