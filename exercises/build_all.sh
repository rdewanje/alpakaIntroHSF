#!/bin/bash
# Script to build all Alpaka exercises

set -e  # Exit on error

echo "========================================="
echo "Building Alpaka Tutorial Exercises"
echo "========================================="
echo ""

# Build Exercise 1: Vector Addition
echo "Building Exercise 1: Vector Addition..."
cd vector_addition/alpaka
./compile.sh
cd - > /dev/null
echo ""

# Build Exercise 2: Vector Reduction
echo "Building Exercise 2: Vector Reduction..."
cd vector_reduction/alpaka
./compile.sh
cd - > /dev/null
echo ""

echo "========================================="
echo "All exercises built successfully!"
echo "========================================="
echo ""
echo "To run the exercises:"
echo "  Exercise 1 Template: ./vector_addition/alpaka/vector_add_template"
echo "  Exercise 1 Solution: ./vector_addition/alpaka/vector_add_solution"
echo "  Exercise 2 Template: ./vector_reduction/alpaka/vector_reduce_template"
echo "  Exercise 2 Solution: ./vector_reduction/alpaka/vector_reduce_solution"
echo ""
echo "To build CUDA versions (requires nvcc):"
echo "  cd vector_addition/cuda && make"
echo "  cd vector_reduction/cuda && make"
