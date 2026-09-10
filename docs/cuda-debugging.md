# CUDA debugging notes

When a CUDA change behaves differently from the CPU reference, isolate the smallest workload that reproduces the issue.

## Checklist

- Check kernel launch dimensions and bounds.
- Validate device allocations and copy directions.
- Synchronize during debugging so asynchronous failures are surfaced at the relevant operation.
- Check return codes and CUDA error state after suspicious operations.
- Compare a small deterministic input against a known-correct reference result.

Remove temporary synchronization and verbose diagnostics when the investigation is complete if they are not required by the final implementation.