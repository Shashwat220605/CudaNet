# Contributing to CudaNet

CudaNet is a C++/CUDA neural-network learning project.

## Development workflow

1. Create a feature branch from `master`.
2. Keep changes focused and explain the purpose of the change.
3. Build the project with CMake and test CUDA changes on a compatible NVIDIA environment.
4. Open a pull request describing the implementation and validation performed.

## CUDA changes

When modifying CUDA kernels, document relevant assumptions about matrix dimensions, memory access, synchronization, and the targeted CUDA architecture.

## Security

Do not commit credentials, private keys, generated secrets, or machine-specific sensitive files.
