// STREAM-style memory bandwidth probe. The stencil's speed ceiling is the memory
// system, not the ALUs — this measures that ceiling on the same machine so the
// stencil's achieved GB/s can be read against it.
#pragma once

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace heat {

inline void run_bandwidth_benchmark() {
    using clk = std::chrono::steady_clock;
    const std::size_t N = 64u << 20;  // 64M doubles = 512 MB per array
    std::vector<double> a(N, 1.0), b(N, 2.0), c(N, 0.0);
    const int hw = static_cast<int>(std::thread::hardware_concurrency());

    auto time_copy = [&](int threads) {
        double best = 1e300;
        for (int rep = 0; rep < 3; ++rep) {
            auto t0 = clk::now();
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; ++t) {
                const std::size_t lo = N * t / threads, hi = N * (t + 1) / threads;
                pool.emplace_back([&, lo, hi] { for (std::size_t i = lo; i < hi; ++i) c[i] = a[i]; });
            }
            for (auto& th : pool) th.join();
            double s = std::chrono::duration<double>(clk::now() - t0).count();
            if (s < best) best = s;
        }
        return best;
    };
    auto time_triad = [&](int threads) {
        double best = 1e300;
        for (int rep = 0; rep < 3; ++rep) {
            auto t0 = clk::now();
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; ++t) {
                const std::size_t lo = N * t / threads, hi = N * (t + 1) / threads;
                pool.emplace_back([&, lo, hi] { for (std::size_t i = lo; i < hi; ++i) a[i] = b[i] + 3.0 * c[i]; });
            }
            for (auto& th : pool) th.join();
            double s = std::chrono::duration<double>(clk::now() - t0).count();
            if (s < best) best = s;
        }
        return best;
    };

    std::printf("memory bandwidth (STREAM-style, %zu MB arrays)\n", N * sizeof(double) >> 20);
    std::printf("%3s %12s %12s\n", "thr", "copy GB/s", "triad GB/s");
    for (int t : {1, 2, 4, 8, hw}) {
        if (t > hw) continue;
        double copy = time_copy(t), triad = time_triad(t);
        std::printf("%3d %12.1f %12.1f\n", t, 2.0 * N * sizeof(double) / copy / 1e9,
                    3.0 * N * sizeof(double) / triad / 1e9);
    }
    std::printf("\n");
}

}  // namespace heat
