// heat — 2-D heat diffusion, four parallel backends, one binary.
//
//   heat --backend openmp --n 2000 --steps 500 --threads 8 [--pgm out.pgm] [--csv results.csv]
//   heat --all --n 2000 --steps 500 --threads 1,2,4,8      run every backend at every count
//   heat --bandwidth                                        STREAM-style copy/triad bandwidth
//
// Prints time, cells/s, the effective memory bandwidth the stencil achieved, and a
// bitwise checksum that must match across backends and thread counts.
#include "backends.hpp"
#include "bandwidth.hpp"
#include "grid.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

namespace {

struct Args {
    std::size_t n = 2000;
    int steps = 500;
    std::vector<int> threads = {1};
    std::vector<heat::Backend> backends;
    std::string pgm, csv;
    bool bandwidth = false;
    int reps = 3;
};

void usage() {
    std::fprintf(stderr,
        "usage: heat [--backend serial|threads-spawn|threads-persistent|openmp | --all]\n"
        "            [--n N] [--steps S] [--threads 1,2,4,8] [--reps R]\n"
        "            [--pgm out.pgm] [--csv results.csv] [--bandwidth]\n");
}

bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto val = [&](std::string& out) { if (i + 1 >= argc) return false; out = argv[++i]; return true; };
        std::string v;
        if (k == "--backend") {
            if (!val(v)) return false;
            heat::Backend b;
            if (!heat::parse_backend(v, b)) { std::fprintf(stderr, "unknown backend %s\n", v.c_str()); return false; }
            a.backends.push_back(b);
        } else if (k == "--all") {
            a.backends = heat::all_backends();
        } else if (k == "--n") {
            if (!val(v)) return false; a.n = std::strtoull(v.c_str(), nullptr, 10);
        } else if (k == "--steps") {
            if (!val(v)) return false; a.steps = std::atoi(v.c_str());
        } else if (k == "--reps") {
            if (!val(v)) return false; a.reps = std::atoi(v.c_str());
        } else if (k == "--threads") {
            if (!val(v)) return false;
            a.threads.clear();
            std::stringstream ss(v);
            std::string tok;
            while (std::getline(ss, tok, ',')) a.threads.push_back(std::atoi(tok.c_str()));
        } else if (k == "--pgm") {
            if (!val(a.pgm)) return false;
        } else if (k == "--csv") {
            if (!val(a.csv)) return false;
        } else if (k == "--bandwidth") {
            a.bandwidth = true;
        } else if (k == "--help" || k == "-h") {
            return false;
        } else {
            std::fprintf(stderr, "unknown option %s\n", k.c_str());
            return false;
        }
    }
    if (a.backends.empty() && !a.bandwidth) a.backends.push_back(heat::Backend::Serial);
    if (a.n < 3 || a.steps < 1) return false;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) { usage(); return 2; }

    if (a.bandwidth) {
        heat::run_bandwidth_benchmark();
        if (a.backends.empty()) return 0;
    }

    std::ofstream csv;
    if (!a.csv.empty()) {
        csv.open(a.csv);
        csv << "backend,n,steps,threads,seconds,mcells_per_s,gbytes_per_s,checksum\n";
    }

    std::printf("%-19s %5s %5s %3s %9s %11s %9s  %s\n", "backend", "n", "steps", "thr", "time", "Mcells/s", "GB/s", "checksum");
    double serial_time = 0.0;
    for (heat::Backend b : a.backends) {
        const std::vector<int> counts = (b == heat::Backend::Serial) ? std::vector<int>{1} : a.threads;
        for (int t : counts) {
            double best = 1e300;
            double sum = 0.0;
            heat::Grid curr(a.n), next(a.n);
            for (int r = 0; r < a.reps; ++r) {
                curr = heat::Grid(a.n); next = heat::Grid(a.n);
                heat::init(curr); heat::init(next);
                auto t0 = clk::now();
                heat::run(b, curr, next, a.steps, t);
                double secs = std::chrono::duration<double>(clk::now() - t0).count();
                if (secs < best) best = secs;
                sum = heat::checksum(curr);
            }
            // Traffic model: each interior cell is read once and written once per step
            // (the 3 input rows stream through cache), i.e. 16 bytes per cell-update.
            const double cells = static_cast<double>((a.n - 2) * (a.n - 2)) * a.steps;
            const double mcells = cells / best / 1e6;
            const double gbs = cells * 16.0 / best / 1e9;
            if (b == heat::Backend::Serial) serial_time = best;
            char speed[24] = "";
            if (serial_time > 0 && b != heat::Backend::Serial) std::snprintf(speed, sizeof speed, "  %.2fx", serial_time / best);
            std::printf("%-19s %5zu %5d %3d %8.4fs %11.1f %9.1f  %.6f%s\n", heat::backend_name(b), a.n, a.steps, t, best,
                        mcells, gbs, sum, speed);
            if (csv.is_open()) {
                csv << heat::backend_name(b) << ',' << a.n << ',' << a.steps << ',' << t << ',' << best << ',' << mcells
                    << ',' << gbs << ',' << std::fixed << sum << '\n';
            }
            if (!a.pgm.empty() && b == a.backends.back() && t == counts.back()) {
                if (heat::write_pgm(curr, a.pgm)) std::printf("wrote %s\n", a.pgm.c_str());
            }
        }
    }
    return 0;
}
