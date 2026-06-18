// Parallel 2D heat-diffusion simulation — OpenMP version.
//
// Same finite-difference stencil and double-buffered update as the std::thread
// version (heat_threads.cpp). The only change is HOW the interior rows are
// parallelized: instead of manually spawning and joining std::thread objects
// every time step, this version uses one OpenMP `parallel for`. OpenMP keeps a
// persistent thread team alive across iterations, so we pay no per-step
// thread-creation cost.
//
// Build:  see CMakeLists.txt, or:
//         g++ -O2 -fopenmp heat_omp.cpp -o heat_omp
// Run:    ./heat_omp [N] [STEPS] [THREADS]   (defaults: 1000 500 1)

#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <omp.h>

using Grid = std::vector<std::vector<double>>;

int main(int argc, char** argv) {
    int N        = (argc > 1) ? std::stoi(argv[1]) : 1000;  // grid size
    int STEPS    = (argc > 2) ? std::stoi(argv[2]) : 500;   // time steps
    int nThreads = (argc > 3) ? std::stoi(argv[3]) : 1;     // number of threads
    const double ALPHA = 0.2;

    omp_set_num_threads(nThreads);

    Grid curr(N, std::vector<double>(N, 0.0));
    Grid next = curr;
    for (int j = 0; j < N; ++j) curr[0][j] = 100.0;   // hot top row

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < STEPS; ++step) {
        // Embarrassingly parallel: each next[i][j] reads only from curr, so
        // splitting the interior rows across threads has no shared writes and
        // is race-free by design. schedule(static) gives each thread one
        // contiguous band of rows — the same partition the std::thread version
        // built by hand.
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                next[i][j] = curr[i][j] + ALPHA * (
                    curr[i-1][j] + curr[i+1][j] +
                    curr[i][j-1] + curr[i][j+1] - 4.0 * curr[i][j]);
            }
        }

        for (int j = 0; j < N; ++j) next[0][j] = 100.0;  // keep top hot
        curr.swap(next);                                 // O(1) buffer swap
    }

    auto end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();

    // The simulation is fully deterministic, so the final grid — and this
    // checksum — must be identical for any thread count and must match the
    // std::thread version exactly. Run with 1 and N threads and confirm the
    // checksums match: that's your correctness proof for the parallelization.
    double checksum = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            checksum += curr[i][j];

    std::cout << "[omp]     N=" << N << "  steps=" << STEPS
              << "  threads=" << omp_get_max_threads()
              << "  time=" << secs << " s"
              << "  checksum=" << checksum << "\n";
    return 0;
}
