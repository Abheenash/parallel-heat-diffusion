// Flat, contiguous 2-D grid and the shared stencil kernel used by every backend.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace heat {

// One contiguous allocation, row-major: cell (i, j) lives at data[i * n + j].
// This is the single most important performance decision in the project — a
// vector<vector<double>> puts every row in its own heap block, so a stencil that
// touches rows i-1, i, i+1 walks three unrelated allocations and defeats the
// hardware prefetcher.
struct Grid {
    std::size_t n = 0;
    std::vector<double> data;

    explicit Grid(std::size_t n_, double fill = 0.0) : n(n_), data(n_ * n_, fill) {}
    double* row(std::size_t i) noexcept { return data.data() + i * n; }
    const double* row(std::size_t i) const noexcept { return data.data() + i * n; }
    double& at(std::size_t i, std::size_t j) noexcept { return data[i * n + j]; }
    double at(std::size_t i, std::size_t j) const noexcept { return data[i * n + j]; }
};

constexpr double kHot = 100.0;   // temperature held on the top row
constexpr double kAlpha = 0.2;   // diffusion coefficient (stable for the 5-point stencil when <= 0.25)

// Hot top row, everything else cold. Both buffers get the same boundary so a
// swap never needs to "re-heat" row 0.
inline void init(Grid& g) {
    for (std::size_t j = 0; j < g.n; ++j) g.at(0, j) = kHot;
}

// Update rows [i0, i1) of `next` from `curr`. Interior only: rows 1..n-2, cols 1..n-2.
// Written with raw restrict pointers so the compiler vectorises the inner loop.
inline void update_rows(const Grid& curr, Grid& next, std::size_t i0, std::size_t i1) noexcept {
    const std::size_t n = curr.n;
    for (std::size_t i = (i0 < 1 ? 1 : i0); i < i1 && i < n - 1; ++i) {
        const double* __restrict up = curr.row(i - 1);
        const double* __restrict mid = curr.row(i);
        const double* __restrict dn = curr.row(i + 1);
        double* __restrict out = next.row(i);
        // The vertical and horizontal neighbour pairs are summed separately on purpose:
        // each pair is commutative, so the field stays *bitwise* mirror-symmetric
        // (the test suite checks that). (a+b)+c+d would not be.
        for (std::size_t j = 1; j < n - 1; ++j) {
            out[j] = mid[j] + kAlpha * ((up[j] + dn[j]) + (mid[j - 1] + mid[j + 1]) - 4.0 * mid[j]);
        }
    }
}

// Deterministic, order-fixed checksum so every backend and thread count can be
// compared bitwise. (A parallel reduction would reorder the adds and break that.)
inline double checksum(const Grid& g) noexcept {
    double s = 0.0;
    for (double v : g.data) s += v;
    return s;
}

// Write the temperature field as a binary PGM (0 = cold, 255 = hot) — viewable anywhere.
inline bool write_pgm(const Grid& g, const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P5\n%zu %zu\n255\n", g.n, g.n);
    std::vector<unsigned char> row(g.n);
    for (std::size_t i = 0; i < g.n; ++i) {
        for (std::size_t j = 0; j < g.n; ++j) {
            double v = g.at(i, j) / kHot;
            if (v < 0) v = 0;
            if (v > 1) v = 1;
            row[j] = static_cast<unsigned char>(v * 255.0 + 0.5);
        }
        std::fwrite(row.data(), 1, g.n, f);
    }
    std::fclose(f);
    return true;
}

}  // namespace heat
