// The four ways the same stencil gets parallelised. Each returns the number of
// time steps run; the caller owns the grids and measures time around the call.
#pragma once

#include "grid.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace heat {

enum class Backend { Serial, ThreadsSpawn, ThreadsPersistent, OpenMP };

const char* backend_name(Backend b);
bool parse_backend(const std::string& s, Backend& out);
bool backend_available(Backend b);
std::vector<Backend> all_backends();

// Runs `steps` iterations, leaving the final field in `curr`.
void run(Backend b, Grid& curr, Grid& next, int steps, int threads);

}  // namespace heat
