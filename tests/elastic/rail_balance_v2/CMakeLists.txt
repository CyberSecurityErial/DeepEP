cmake_minimum_required(VERSION 3.24)
project(standalone_rail_balance LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(BALANCE_CUDA_ARCHITECTURES "89;90" CACHE STRING
    "CUDA architectures; include 90 to build the TMA path")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(rail_balance STATIC src/rail_balance.cu)
target_include_directories(
  rail_balance PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rail_balance PUBLIC CUDA::cudart)
target_compile_options(
  rail_balance PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
set_target_properties(
  rail_balance PROPERTIES
    CUDA_ARCHITECTURES "${BALANCE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)

option(BALANCE_BUILD_TESTS "Build the eight-GPU correctness test" ON)
if(BALANCE_BUILD_TESTS)
  enable_testing()
  add_executable(rail_balance_test tests/rail_balance_test.cu)
  target_link_libraries(rail_balance_test PRIVATE rail_balance CUDA::cudart)
  set_target_properties(
    rail_balance_test PROPERTIES
      CUDA_ARCHITECTURES "${BALANCE_CUDA_ARCHITECTURES}")
  add_test(NAME rail_balance_test COMMAND rail_balance_test)
endif()

option(BALANCE_BUILD_BENCHMARK "Build the eight-GPU microbenchmark" ON)
if(BALANCE_BUILD_BENCHMARK)
  add_executable(
    rail_balance_bench benchmarks/rail_balance_bench.cu)
  target_link_libraries(
    rail_balance_bench PRIVATE rail_balance CUDA::cudart Threads::Threads)
  set_target_properties(
    rail_balance_bench PROPERTIES
      CUDA_ARCHITECTURES "${BALANCE_CUDA_ARCHITECTURES}")
endif()
