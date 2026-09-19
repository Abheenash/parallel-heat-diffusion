# parallel-heat-diffusion

A 2-D heat-diffusion solver (5-point finite-difference stencil) parallelised **four ways** —
serial, `std::thread` spawned per step, persistent `std::thread` workers with a spin barrier,
and OpenMP — benchmarked head-to-head on identical inputs, **proven bitwise-identical** across
every backend and thread count, and analysed against the machine's *measured* memory bandwidth
so the scaling limits are explained, not guessed.

[![ci](https://github.com/Abheenash/parallel-heat-diffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/Abheenash/parallel-heat-diffusion/actions/workflows/ci.yml)

<p align="center"><img src="docs/heat-2000.png" width="360" alt="Temperature field after 500 steps: a hot top edge diffusing downward into a cold plate"><br>
<sub>2000×2000 grid after 500 steps, hot top edge — written by the solver as a PGM.</sub></p>

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release        # add -DOpenMP_ROOT=$(brew --prefix libomp) on macOS
cmake --build build -j && ctest --test-dir build
./build/heat --all --n 2000 --steps 500 --threads 1,2,4,8 --pgm out.pgm --csv out.csv
./build/heat --bandwidth                              # STREAM-style ceiling for this machine
```

## The two results that matter

**1. On a DRAM-resident grid, every backend hits the same wall — and the wall is the memory bus.**

2000×2000 doubles (32 MB per buffer), 500 steps, Apple M4 (4P + 6E cores):

| backend | 1 thr | 2 | 4 | 8 | 10 |
| --- | --- | --- | --- | --- | --- |
| serial | 0.629 s | | | | |
| threads-spawn (per step) | 0.99× | 1.60× | 1.80× | 1.78× | 1.72× |
| threads-persistent + spin barrier | 1.02× | 1.62× | **2.03×** | 1.49× | 1.44× |
| openmp | 1.02× | 1.64× | 1.97× | 1.78× | 1.76× |

The stencil moves 16 bytes per cell-update (one read, one write; the three input rows stream through
cache). At 4 threads that's **~100 GB/s** — and `heat --bandwidth` measures this machine's STREAM
copy ceiling at **98 GB/s** (75 GB/s from a single core). So the best possible speedup over the
52 GB/s serial run is ≈ 1.9×, and that's exactly where all three backends land. Adding threads
beyond four can't help; nothing about the code is the bottleneck. The 4000×4000 run
([`docs/scaling-n4000-m4.txt`](docs/scaling-n4000-m4.txt)) tells the same story at 1.9×.

**2. On a cache-resident grid, synchronisation cost is the whole story.**

512×512 (2 MB per buffer — fits in L2), 8,000 steps, so each step is ~55 µs:

| backend | 1 thr | 2 | 4 | 8 | 10 |
| --- | --- | --- | --- | --- | --- |
| serial | 0.452 s | | | | |
| threads-spawn (per step) | **0.81×** | 1.19× | 1.23× | 0.94× | 0.84× |
| threads-persistent, condvar barrier (v2.0) | 0.99× | 1.54× | 2.14× | 1.73× | 1.43× |
| threads-persistent, **spin barrier** | 1.00× | 1.95× | 2.70× | 2.57× | **2.89×** |
| openmp | 0.96× | 1.38× | 1.52× | 0.96× | 0.85× |

- Spawning threads every step is *slower than serial* at every count: creating and joining a thread
  costs more than 55 µs of work.
- Persistent workers with a mutex/condvar barrier get 2.14×; swapping it for a spinning
  sense-reversal barrier (parking is a syscall each way) lifts that to **2.89×** — the effective
  213 GB/s only makes sense because the data is served from cache, not DRAM.
- OpenMP's implicit per-step barrier behaves like the condvar version and collapses past four threads
  when work lands on the efficiency cores.

Same code, same kernel, opposite conclusions depending on whether the working set fits in cache — which
is the actual lesson of the project.

## Correctness — bitwise, not "close enough"

`tests/test_heat.cpp` runs every backend at thread counts 1, 2, 3, 4 and 7 on grids of 3, 7, 64 and 257
and step counts 1, 2, 7 and 50, and `memcmp`s the entire field against the serial reference:
**320 combinations, identical bits.** That works because the parallel decomposition only changes
*which thread* computes a cell, never the arithmetic — and the checksum is a fixed-order sum, not a
parallel reduction.

It then checks physical invariants the field must satisfy: every value in `[0, 100]`, the hot row
stays at 100, the cold walls stay at 0, row means are non-increasing away from the heat source, and
the field is **exactly** left/right mirror-symmetric. That last one failed the first time: with
`(up + dn + left) + right` the mirrored cell computes `(up + dn + right) + left`, which is a
different floating-point result. Summing the vertical and horizontal neighbour pairs separately —
each pair is commutative — makes the symmetry exact at no extra cost, and the test pins it.

CI runs the suite on Linux and macOS and again under ThreadSanitizer and AddressSanitizer (with the
OpenMP backend off in the sanitizer job — TSAN can't see libomp's internal synchronisation and
reports false positives unless libomp is built with TSAN support).

## Design

- **One contiguous allocation** (`Grid::data`, row-major) — the previous version used
  `vector<vector<double>>`, which puts each row in its own heap block and defeats the prefetcher.
  The inner loop is written on `__restrict` row pointers so the compiler vectorises it.
- **Both buffers start with the hot boundary**, so a swap is a pointer swap and nothing re-heats row 0.
- **Persistent backend:** each worker owns a fixed row band for the whole run (its rows stay warm in
  its core's cache) and swaps its *own* view of the two buffers after the barrier, so one barrier per
  step suffices and nothing shared is mutated.
- **OpenMP backend:** one `parallel` region hoisted out of the time loop so the team is created once;
  `omp for` supplies the per-step barrier and `omp single` does the swap.
- **`--bandwidth`** runs a STREAM-style copy and triad at 1…N threads on 512 MB arrays, so the
  stencil's achieved GB/s in the same table can be read against the machine's real ceiling.

## Reproduce

`scripts/scaling.sh` regenerates every table above (`docs/*.txt`, `results/*.csv`). Numbers in this
README are from an Apple M4, clang, `-O3 -march=native`, best of 3 runs.

## What I'd do next

Temporal blocking (compute several time steps per cache-resident tile before writing back) is the
standard way past the bandwidth wall for this kernel — it raises arithmetic intensity so the roofline
moves. A NUMA-aware first-touch placement would matter on a multi-socket machine; it doesn't on an M4.
