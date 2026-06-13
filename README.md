# Mini-autodiff

## Summary
**Mini-autodiff** is my first implementation of an automatic differentiation engine. It currently supports computational graphs, backpropagation, memory pooling, broadcasting, optimization, and loss functions. A small `main` example is included to test the system.

The goal of this project is to keep everything minimal, simple, and educational. It is not yet optimized for large workloads (e.g., CPU/GPU parallelization), but I plan to explore these optimizations in the future and document the process.


## Project structure
The source is split the way larger engines (e.g. PyTorch, TensorFlow, JAX) separate their low-level runtime from the user-facing API:

```
core/   - the autodiff engine itself: Tensor, the compute graph (ComputeNode),
          the memory pool, and the math/broadcasting ops that build the graph.
nn/     - higher-level API built on top of core: optimizers (SGD, Adam) and
          loss functions (MSE, BCE).
main.cc - example programs (linear regression, MLP on XOR).
tests/  - gradient-checking test harness.
build/  - generated object files and binaries (not checked in).
```

Build with `make` (binary at `build/run`), or run the tests with `make test` (see below).


## For learners
The repository also includes a [blog.md](blog.md) that explains how the engine works and introduces the core ideas behind automatic differentiation. It follows my journey building the system and aims to make the concepts intuitive for newcomers.


## Testing
The `tests/` directory contains a gradient-checking harness that compares the engine's analytic gradients (via `Tensor::backward()`) against numerical gradients computed with central differences. It covers every op (including broadcasting and matmul edge cases), both losses, and matmul's shape validation.

Run it with:

```bash
make test
```

This builds the `build/test_gradcheck` binary and runs it, printing a `[PASS]`/`[FAIL]` line per case.


---
Published under the MIT License.