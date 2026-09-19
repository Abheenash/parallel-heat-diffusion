#include "backends.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

#ifdef HEAT_HAVE_OPENMP
#include <omp.h>
#endif

namespace heat {

const char* backend_name(Backend b) {
    switch (b) {
        case Backend::Serial: return "serial";
        case Backend::ThreadsSpawn: return "threads-spawn";
        case Backend::ThreadsPersistent: return "threads-persistent";
        case Backend::OpenMP: return "openmp";
    }
    return "?";
}

bool parse_backend(const std::string& s, Backend& out) {
    for (Backend b : {Backend::Serial, Backend::ThreadsSpawn, Backend::ThreadsPersistent, Backend::OpenMP}) {
        if (s == backend_name(b)) { out = b; return true; }
    }
    return false;
}

bool backend_available(Backend b) {
#ifdef HEAT_HAVE_OPENMP
    (void)b;
    return true;
#else
    return b != Backend::OpenMP;
#endif
}

std::vector<Backend> all_backends() {
    std::vector<Backend> v;
    for (Backend b : {Backend::Serial, Backend::ThreadsSpawn, Backend::ThreadsPersistent, Backend::OpenMP}) {
        if (backend_available(b)) v.push_back(b);
    }
    return v;
}

// Split interior rows [1, n-1) into `threads` near-equal contiguous bands.
static std::pair<std::size_t, std::size_t> band(std::size_t n, int threads, int t) {
    const std::size_t rows = n - 2;
    const std::size_t lo = 1 + rows * static_cast<std::size_t>(t) / static_cast<std::size_t>(threads);
    const std::size_t hi = 1 + rows * static_cast<std::size_t>(t + 1) / static_cast<std::size_t>(threads);
    return {lo, hi};
}

// ---------------------------------------------------------------- serial
static void run_serial(Grid& curr, Grid& next, int steps) {
    for (int s = 0; s < steps; ++s) {
        update_rows(curr, next, 1, curr.n - 1);
        std::swap(curr.data, next.data);
    }
}

// ---------------------------------------------------------------- threads: spawn per step
// The obvious first version: create `threads` std::threads every time step, join them,
// swap. Correct, and it measures exactly what per-step thread creation costs.
static void run_threads_spawn(Grid& curr, Grid& next, int steps, int threads) {
    for (int s = 0; s < steps; ++s) {
        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(threads));
        for (int t = 0; t < threads; ++t) {
            const auto range = band(curr.n, threads, t);
            const std::size_t lo = range.first, hi = range.second;
            pool.emplace_back([&curr, &next, lo, hi] { update_rows(curr, next, lo, hi); });
        }
        for (auto& th : pool) th.join();
        std::swap(curr.data, next.data);
    }
}

// ---------------------------------------------------------------- threads: persistent + barrier
// A reusable barrier. A stencil step on a cache-resident grid is tens of
// microseconds, so parking on a condition variable (a syscall each way) would cost
// as much as the work. Threads spin on the generation counter instead, yielding
// after a while so an oversubscribed machine still makes progress.
class Barrier {
public:
    explicit Barrier(int parties) : parties_(parties) {}

    void arrive_and_wait() {
        const unsigned gen = generation_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == parties_) {
            arrived_.store(0, std::memory_order_relaxed);
            generation_.store(gen + 1, std::memory_order_release);  // releases everyone
            return;
        }
        unsigned spins = 0;
        while (generation_.load(std::memory_order_acquire) == gen) {
            if (++spins > 2000) std::this_thread::yield();
        }
    }

private:
    const int parties_;
    std::atomic<int> arrived_{0};
    std::atomic<unsigned> generation_{0};
};

// Create the workers once. Each owns a fixed row band for the whole simulation (so
// its rows stay warm in its core's cache across steps) and swaps its own view of
// the two buffers after every barrier — no second barrier and no shared swap.
static void run_threads_persistent(Grid& curr, Grid& next, int steps, int threads) {
    Barrier barrier(threads);
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t) {
        const auto range = band(curr.n, threads, t);
        const std::size_t lo = range.first, hi = range.second;
        pool.emplace_back([&, lo, hi] {
            Grid* a = &curr;
            Grid* b = &next;
            for (int s = 0; s < steps; ++s) {
                update_rows(*a, *b, lo, hi);
                barrier.arrive_and_wait();
                std::swap(a, b);
            }
        });
    }
    for (auto& th : pool) th.join();
    // After an odd number of steps the newest field is in `next`; make `curr` the result.
    if (steps % 2 == 1) std::swap(curr.data, next.data);
}

// ---------------------------------------------------------------- OpenMP
// One parallel region for the whole simulation (hoisted out of the time loop) so the
// team is created once; `omp for` inside carries an implicit barrier per step, and the
// swap is done by a single thread with the `single` construct's implicit barrier.
static void run_openmp(Grid& curr, Grid& next, int steps, int threads) {
#ifdef HEAT_HAVE_OPENMP
    Grid* a = &curr;
    Grid* b = &next;
    const std::size_t n = curr.n;
    omp_set_num_threads(threads);
#pragma omp parallel default(none) shared(a, b, steps, n)
    {
        for (int s = 0; s < steps; ++s) {
#pragma omp for schedule(static)
            for (std::size_t i = 1; i < n - 1; ++i) {
                update_rows(*a, *b, i, i + 1);
            }
#pragma omp single
            std::swap(a, b);
        }
    }
    if (steps % 2 == 1) std::swap(curr.data, next.data);
#else
    (void)curr; (void)next; (void)steps; (void)threads;
#endif
}

void run(Backend b, Grid& curr, Grid& next, int steps, int threads) {
    if (threads < 1) threads = 1;
    switch (b) {
        case Backend::Serial: run_serial(curr, next, steps); break;
        case Backend::ThreadsSpawn: run_threads_spawn(curr, next, steps, threads); break;
        case Backend::ThreadsPersistent: run_threads_persistent(curr, next, steps, threads); break;
        case Backend::OpenMP: run_openmp(curr, next, steps, threads); break;
    }
}

}  // namespace heat
