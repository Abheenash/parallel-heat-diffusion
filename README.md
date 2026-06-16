# Parallel Heat Diffusion Simulation

A 2D heat-diffusion simulation written in C++ that models how heat spreads across a flat plate over time. It includes a serial version and a multithreaded version, and benchmarks how the parallel version scales across CPU cores. Built and tested on Apple Silicon.

## How It Works

The simulation represents a metal plate as a 2D grid of temperatures. The top edge is held hot (100°) and the rest of the plate starts cold (0°). At each time step, every interior cell is updated toward the average of its four neighbors — the finite-difference form of the 2D heat equation. Over many steps, heat spreads from the hot edge across the plate and settles into a smooth gradient.

Two grids are used at all times: the program reads from the current grid and writes results into a second grid, then swaps them ("double buffering"). This ensures every cell is updated from one consistent snapshot, which also makes the update safe to parallelize.

The parallel version splits the grid's rows into equal bands and gives one band to each thread. Because every thread writes to a different set of cells and only reads the shared input grid, no two threads ever touch the same memory — so there are no data races.

## Build and Run

**Serial version** (prints a text heatmap of the final state):
```bash
clang++ heat.cpp -o heat
./heat
```

**Parallel version** (the three numbers are grid size, time steps, and thread count):
```bash
clang++ -O2 -std=c++17 heat2.cpp -o heat2
./heat2 1000 500 4
```

## Results

Benchmark on a MacBook Air (Apple M4, 2025, 16 GB), 1000×1000 grid, 500 time steps:

| Threads | Time (s) | Speedup |
|--------:|---------:|--------:|
| 1       | 0.154    | 1.0×    |
| 2       | 0.076    | 2.0×    |
| 4       | 0.057    | 2.7×    |
| 8       | 0.059    | 2.6×    |

Speedup is near-linear up to 2 threads, peaks at about 2.7× with 4 threads, and then flattens out — 8 threads is no faster than 4.

## Scaling Analysis

The speedup does not keep doubling as threads are added, for three reasons:

1. **The workload is memory-bound.** Each cell update performs only a few arithmetic operations but reads five values from memory. The cores spend most of their time waiting on memory rather than computing, so once a couple of threads saturate the available memory bandwidth, adding more threads mostly adds more cores waiting in line. This is the dominant reason stencil-style simulations rarely scale linearly.

2. **Apple Silicon uses two kinds of cores.** The M4 combines fast "performance" cores with slower "efficiency" cores. The first few threads run on the performance cores and scale well; additional threads spill onto the efficiency cores, which contribute much less. This is why moving from 4 to 8 threads gives no improvement.

3. **Threads are recreated every time step.** This implementation spawns a fresh set of threads on each of the 500 steps, and that setup cost grows with the thread count, eating into the gains at higher thread counts.

## Future Improvements
- Reuse threads with a thread pool instead of creating them every step
- Add an OpenMP version and compare the two approaches
- Scale across multiple machines with MPI, or onto the GPU with CUDA
- Support larger grids and generate a rendered heatmap image

## Example Output

A 40×40 run, showing heat spreading from the hot top edge (`@`) down into the cooler plate:

```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
=*###%%%%%%%%%%%%%%%%%%%%%%%%%%%%###*=
:=+**#########%%%%%%%%%%#########**+=:
.-=++****####################****++=-.
.:-==++*******##########*******++==-:.
.:--==++++******************++++==--:.
  .:--===+++++************+++++===--:.
  .::--===++++++++++++++++++++===--::.
  ..::--=====++++++++++++++=====--::..
  ..::----======++++++++======----::..
  ...::----==================----::...
   ..:::-----==============-----:::..
   ...:::--------======--------:::...
   ...:::::------------------:::::...
    ...:::::----------------:::::...
    ....:::::::----------:::::::....
    .....::::::::::::::::::::::.....
     .....::::::::::::::::::::.....
     ......::::::::::::::::::......
      .......::::::::::::::.......
      .........::::::::::.........
       ..........................
        ........................
         ......................
           ..................
            ................
               ..........
```
