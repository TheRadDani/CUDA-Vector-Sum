# Specify the minimum version for CMake
cmake_minimum_required(VERSION 3.10)

# Set policy for CUDA architectures (CMP0104)
cmake_policy(SET CMP0104 NEW)

# Project name and enable CUDA
project(VecAdd CUDA)

# Run nvidia-smi command to get compute capability
execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE GPU_COMPUTE_CAPABILITY
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Construct the architecture flag for CUDA
string(REGEX REPLACE "\\." "" COMPUTE_CAPABILITY_SHORT "${GPU_COMPUTE_CAPABILITY}")

# Set the CUDA architectures property dynamically
set(CMAKE_CUDA_ARCHITECTURES ${COMPUTE_CAPABILITY_SHORT})

# Add executable target with CUDA source files
add_executable(vecAdd main.cu vecAdd.cu)

# Set the C++ standard (optional, depending on your code)
set_target_properties(vecAdd PROPERTIES
    CXX_STANDARD 11
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories (optional if you have additional include directories)
include_directories(${CMAKE_SOURCE_DIR})
