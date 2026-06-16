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
                curr[i][j-1] + curr[i][j+1] - 4 * curr[i][j]);
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

    std::cout << "N=" << N << "  steps=" << STEPS
              << "  threads=" << nThreads
              << "  time=" << secs << " s\n";
    return 0;
}
