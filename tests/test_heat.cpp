// Every backend, at every thread count, must produce the SAME BITS as the serial
// reference — not "close", identical. Plus physical invariants the field must obey.
#include "../src/backends.hpp"
#include "../src/grid.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #c, __FILE__, __LINE__); std::exit(1); } } while (0)

static heat::Grid solve(heat::Backend b, std::size_t n, int steps, int threads) {
    heat::Grid curr(n), next(n);
    heat::init(curr); heat::init(next);
    heat::run(b, curr, next, steps, threads);
    return curr;
}

int main() {
    int tests = 0;
    for (std::size_t n : {3u, 7u, 64u, 257u}) {
        for (int steps : {1, 2, 7, 50}) {
            heat::Grid ref = solve(heat::Backend::Serial, n, steps, 1);
            for (heat::Backend b : heat::all_backends()) {
                for (int t : {1, 2, 3, 4, 7}) {
                    heat::Grid g = solve(b, n, steps, t);
                    CHECK(g.data.size() == ref.data.size());
                    CHECK(std::memcmp(g.data.data(), ref.data.data(), g.data.size() * sizeof(double)) == 0);
                    ++tests;
                }
            }
            // Physical invariants on the reference field.
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    const double v = ref.at(i, j);
                    CHECK(v >= 0.0 && v <= heat::kHot);                 // bounded by the boundary values
                    CHECK(v == ref.at(i, n - 1 - j));                    // left/right mirror symmetry, exact
                }
            }
            for (std::size_t j = 0; j < n; ++j) CHECK(ref.at(0, j) == heat::kHot);   // top row stays hot
            for (std::size_t i = 1; i + 1 < n; ++i) {
                CHECK(ref.at(i, 0) == 0.0 && ref.at(i, n - 1) == 0.0);            // cold side walls untouched
            }
            // Heat flows down: row means are non-increasing away from the hot row.
            double prev = heat::kHot;
            for (std::size_t i = 1; i + 1 < n; ++i) {
                double m = 0; for (std::size_t j = 0; j < n; ++j) m += ref.at(i, j);
                m /= static_cast<double>(n);
                CHECK(m <= prev + 1e-12);
                prev = m;
            }
        }
    }
    std::printf("%d backend/thread-count combinations bitwise-identical to serial; invariants hold\n", tests);
    return 0;
}
