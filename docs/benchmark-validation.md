# Benchmark validation

Before comparing CudaNet performance across revisions, keep the workload and runtime environment consistent.

Record the input dimensions, CUDA architecture, driver/runtime version, and number of repetitions. For correctness-sensitive optimizations, compare the output against a known reference before treating a faster result as an improvement.