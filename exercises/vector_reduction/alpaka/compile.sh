#!/bin/bash
# Compile script for Vector Reduction exercise

set -e

# Paths
ALPAKA_ROOT="../../../alpaka"
ALPAKA_INCLUDE="${ALPAKA_ROOT}/include"

# Compiler settings
CXX=${CXX:-g++}
CXXFLAGS="-std=c++20 -O3 -march=native"
INCLUDES="-I${ALPAKA_INCLUDE}"

# Define which backend to use (controls config.h)
# Options:
#   ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED     (CPU Serial - default)
#   ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED    (OpenMP Blocks - add -fopenmp)
#   ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED     (TBB Blocks - add -ltbb)
#   ALPAKA_ACC_GPU_CUDA_ENABLED            (CUDA - use nvcc)
#   ALPAKA_ACC_GPU_HIP_ENABLED             (HIP/ROCm - use hipcc)
BACKEND="-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED"

# Additional flags for specific backends
# OMPFLAGS="-fopenmp"              # For OpenMP backend
# TBBFLAGS="-ltbb"                 # For TBB backend

echo "Backend: ${BACKEND}"
echo ""

echo "Compiling vector_reduce_template..."
${CXX} ${CXXFLAGS} ${INCLUDES} ${BACKEND} \
    vector_reduce_template.cpp \
    -o vector_reduce_template

echo "Compiling vector_reduce_solution..."
${CXX} ${CXXFLAGS} ${INCLUDES} ${BACKEND} \
    vector_reduce_solution.cpp \
    -o vector_reduce_solution

echo ""
echo "✓ Compilation successful!"
echo ""
echo "Run with:"
echo "  ./vector_reduce_template"
echo "  ./vector_reduce_solution"
