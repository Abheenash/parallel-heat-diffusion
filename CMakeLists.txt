cmake_minimum_required(VERSION 3.12)
project(parallel_heat_diffusion CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to an optimized build; -O2 matters a lot for a tight stencil loop.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Baseline: std::thread version
add_executable(heat_threads heat_threads.cpp)
target_link_libraries(heat_threads PRIVATE Threads::Threads)

# OpenMP version
add_executable(heat_omp heat_omp.cpp)
target_link_libraries(heat_omp PRIVATE OpenMP::OpenMP_CXX)
