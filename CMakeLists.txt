cmake_minimum_required(VERSION 3.12...3.18)
project(kepler_jax LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/src/kepler_jax/src)

# CPU op library
pybind11_add_module(
    cpu_ops
    ${CMAKE_CURRENT_LIST_DIR}/src/kepler_jax/src/cpu_ops.cc)
install(TARGETS cpu_ops DESTINATION kepler_jax)

if (KEPLER_JAX_CUDA)
    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    pybind11_add_module(
        gpu_ops
        ${CMAKE_CURRENT_LIST_DIR}/src/kepler_jax/src/kernels.cc.cu
        ${CMAKE_CURRENT_LIST_DIR}/src/kepler_jax/src/cpu_ops.cc)
    install(TARGETS gpu_ops DESTINATION kepler_jax)
else()
    message(STATUS "Building without CUDA")
endif()
