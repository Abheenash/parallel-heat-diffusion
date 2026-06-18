# Parallel Heat Diffusion Simulation (std::thread vs OpenMP)

A 2D heat-diffusion simulation solved with an explicit finite-difference
stencil on a double-buffered grid. The interior is parallelized two ways so the
parallelization strategies can be benchmarked head-to-head on identical inputs:

- **`heat_threads.cpp`** — baseline. Spawns and joins `std::thread` objects every
  time step, with the interior rows partitioned into per-thread bands.
- **`heat_omp.cpp`** — OpenMP version. Replaces the manual spawn/join with a
  single `#pragma omp parallel for`, reusing a persistent thread team across all
  time steps.

Both produce bit-for-bit identical results (see *Correctness* below).

## The physics

Each cell is updated from its four neighbors using the discretized 2D heat
equation:

```
next[i][j] = curr[i][j] + alpha * ( curr[i-1][j] + curr[i+1][j]
                                  + curr[i][j-1] + curr[i][j+1] - 4*curr[i][j] )
```

The top row is held at 100.0 (a fixed hot boundary); everything else starts at
0.0 and diffuses over `STEPS` iterations. Writes go to a separate `next` buffer
and the buffers are swapped each step, so no cell is read and written in the
same pass.

## Build

With CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# binaries: build/heat_threads  build/heat_omp
```

Or directly:

```bash
g++ -O2 -pthread heat_threads.cpp -o heat_threads
g++ -O2 -fopenmp heat_omp.cpp     -o heat_omp
```

On macOS (Apple clang), OpenMP needs the Homebrew `libomp` runtime — install it
with `brew install libomp`, then build the OpenMP version with:

```bash
clang++ -std=c++17 -O2 -Xpreprocessor -fopenmp \
  -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
  heat_omp.cpp -o heat_omp
```

## Run

```bash
./heat_omp [N] [STEPS] [THREADS]      # defaults: 1000 500 1
./heat_omp 2000 500 8
```

Output line:

```
[omp]     N=2000  steps=500  threads=8  time=0.41 s  checksum=...
```

## Benchmark

OpenMP version across thread counts on a 2000x2000 grid, 500 time steps, on a
MacBook Air (Apple M4, 2025, 16 GB):

| Threads | Time (s) | Speedup |
|--------:|---------:|--------:|
| 1       | 0.84     | 1.00x   |
| 2       | 0.48     | 1.74x   |
| 4       | 0.36     | 2.31x   |
| 8       | 0.41     | 2.05x   |

Speedup peaks at 4 threads and dips slightly at 8 — consistent with the
memory-bandwidth limit discussed below and the M4's mix of 4 performance and 4
efficiency cores (the extra threads land on the slower efficiency cores).

Reproduce with:

```bash
for t in 1 2 4 8; do ./heat_omp 2000 500 $t; done
```

## Correctness

The simulation is deterministic, so the final grid — and the printed
`checksum` (the sum of all cells) — must be **identical for any thread count and
identical between the two implementations**. To verify:

```bash
./heat_threads 600 200 1
./heat_omp     600 200 1
./heat_omp     600 200 8
```

All three print the same checksum. That single number is the proof that the
parallelization introduced no data races and that the OpenMP refactor preserved
the original behavior exactly.

## What changed, and why it matters

The algorithm and data layout are unchanged between the two versions — the only
difference is the parallelization mechanism:

- The baseline constructs `THREADS` new `std::thread` objects and joins them on
  **every** time step. For a 500-step run that is hundreds of rounds of thread
  creation and teardown.
- The OpenMP version enters the runtime's thread team once and reuses it across
  all steps. The `schedule(static)` clause hands each thread one contiguous band
  of rows — the same partition the baseline built by hand — so the work split is
  equivalent and the comparison is apples-to-apples.

The update is safe to parallelize because it is embarrassingly parallel: every
`next[i][j]` reads only from the `curr` buffer, so distinct rows never share a
write target.

## Scaling notes

This stencil is **memory-bandwidth bound**, not compute bound — each cell does a
handful of flops but touches five doubles, so beyond a few threads the limit is
how fast memory can feed the cores, not the cores themselves. Two levers that
matter more than thread count:

1. **Data layout.** `std::vector<std::vector<double>>` stores each row in a
   separate heap allocation, so traversing the grid chases pointers and hurts
   cache locality and vectorization. A flat `std::vector<double>` of size `N*N`
   indexed as `i*N + j` is contiguous and is the standard next optimization.
2. **One parallel region instead of one per step.** Hoisting the `parallel`
   region out of the time loop removes per-step fork/join overhead entirely:

   ```cpp
   #pragma omp parallel
   {
       for (int step = 0; step < STEPS; ++step) {
           #pragma omp for schedule(static)
           for (int i = 1; i < N - 1; ++i)
               for (int j = 1; j < N - 1; ++j)
                   next[i][j] = curr[i][j] + ALPHA * (
                       curr[i-1][j] + curr[i+1][j] +
                       curr[i][j-1] + curr[i][j+1] - 4.0 * curr[i][j]);
           // implicit barrier: all next writes finish before the swap

           #pragma omp single
           {
               for (int j = 0; j < N; ++j) next[0][j] = 100.0;
               curr.swap(next);
           }
           // implicit barrier: swap finishes before the next iteration reads curr
       }
   }
   ```

   Benchmarking all three variants (per-step std::thread, per-step omp for,
   single-region omp for) makes the fork/join cost visible.
