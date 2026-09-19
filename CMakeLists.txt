cmake_minimum_required(VERSION 3.14)
project(parallel_heat_diffusion LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(HEAT_NATIVE "Compile with -march=native" ON)
option(HEAT_SANITIZE "Build with -fsanitize=<thread|address>" "")

find_package(Threads REQUIRED)
# OpenMP is optional so the project builds on stock Apple clang; install libomp
# (brew install libomp) and pass -DOpenMP_ROOT=/opt/homebrew/opt/libomp to enable it.
find_package(OpenMP)

add_library(heat_core STATIC src/backends.cpp)
target_include_directories(heat_core PUBLIC src)
target_link_libraries(heat_core PUBLIC Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(heat_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(heat_core PUBLIC HEAT_HAVE_OPENMP=1)
  message(STATUS "OpenMP backend: enabled")
else()
  message(STATUS "OpenMP backend: disabled (OpenMP not found)")
endif()

function(heat_flags tgt)
  target_compile_options(${tgt} PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wshadow -O3>)
  if(HEAT_NATIVE AND NOT HEAT_SANITIZE)
    target_compile_options(${tgt} PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native>)
  endif()
  if(HEAT_SANITIZE)
    target_compile_options(${tgt} PRIVATE -fsanitize=${HEAT_SANITIZE} -g -O1 -fno-omit-frame-pointer)
    target_link_options(${tgt} PRIVATE -fsanitize=${HEAT_SANITIZE})
  endif()
endfunction()
heat_flags(heat_core)

add_executable(heat src/main.cpp)
target_link_libraries(heat PRIVATE heat_core)
heat_flags(heat)

enable_testing()
add_executable(test_heat tests/test_heat.cpp)
target_link_libraries(test_heat PRIVATE heat_core)
heat_flags(test_heat)
add_test(NAME heat_bitwise_and_invariants COMMAND test_heat)
