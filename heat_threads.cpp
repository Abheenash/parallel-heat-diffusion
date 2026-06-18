// Parallel 2D heat-diffusion simulation — std::thread version (baseline).
//
// This is the original implementation: it partitions the interior rows into
// per-thread bands and spawns/joins std::thread objects EVERY time step. It is
// kept here so you can benchmark it head-to-head against the OpenMP version
// (heat_omp.cpp) on the exact same machine, grid size, and step count.
//
// Build:  see CMakeLists.txt, or:
//         g++ -O2 -pthread heat_threads.cpp -o heat_threads
// Run:    ./heat_threads [N] [STEPS] [THREADS]   (defaults: 1000 500 1)

#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <string>

using Grid = std::vector<std::vector<double>>;

// One thread runs this: update interior rows [rowStart, rowEnd).
void updateRows(const Grid& curr, Grid& next, int N, double alpha,
                int rowStart, int rowEnd) {
    for (int i = rowStart; i < rowEnd; ++i)
        for (int j = 1; j < N - 1; ++j)
            next[i][j] = curr[i][j] + alpha * (
                curr[i-1][j] + curr[i+1][j] +
                curr[i][j-1] + curr[i][j+1] - 4.0 * curr[i][j]);
}

int main(int argc, char** argv) {
    int N        = (argc > 1) ? std::stoi(argv[1]) : 1000;  // grid size
    int STEPS    = (argc > 2) ? std::stoi(argv[2]) : 500;   // time steps
    int nThreads = (argc > 3) ? std::stoi(argv[3]) : 1;     // number of threads
    const double ALPHA = 0.2;

    Grid curr(N, std::vector<double>(N, 0.0));
    Grid next = curr;
    for (int j = 0; j < N; ++j) curr[0][j] = 100.0;   // hot top row

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < STEPS; ++step) {
        std::vector<std::thread> pool;
        int chunk = (N - 2 + nThreads - 1) / nThreads;   // rows per thread
        for (int t = 0; t < nThreads; ++t) {
            int rowStart = 1 + t * chunk;
            int rowEnd   = std::min(rowStart + chunk, N - 1);
            if (rowStart < rowEnd)
                pool.emplace_back([&, rowStart, rowEnd]() {
                    updateRows(curr, next, N, ALPHA, rowStart, rowEnd);
                });
        }
        for (auto& th : pool) th.join();                 // wait for all threads
        for (int j = 0; j < N; ++j) next[0][j] = 100.0;  // keep top hot
        curr.swap(next);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();

    double checksum = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            checksum += curr[i][j];

    std::cout << "[threads] N=" << N << "  steps=" << STEPS
              << "  threads=" << nThreads
              << "  time=" << secs << " s"
              << "  checksum=" << checksum << "\n";
    return 0;
}
