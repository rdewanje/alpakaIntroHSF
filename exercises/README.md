# Alpaka Programming Exercises

Welcome to the Alpaka tutorial exercises! These hands-on exercises will teach you how to write portable accelerator code using the Alpaka C++ abstraction library.

## What is Alpaka?

Alpaka (Abstraction Library for Parallel Kernel Acceleration) is a C++ library that provides a single-source abstraction for writing accelerator code that can run on:
- CPUs (serial, OpenMP, TBB, std::thread)
- NVIDIA GPUs (CUDA)
- AMD GPUs (HIP)
- Intel GPUs (SYCL)

**Write once, run everywhere!**

## Prerequisites

- C++17 compiler (g++, clang++)
- Alpaka library (included in parent directory)
- Optional: CUDA toolkit (nvcc) for comparing with CUDA versions
- Optional: OpenMP support for parallel CPU backend
- Optional: ROCm for AMD GPUs, oneAPI for Intel GPUs

## Exercise Structure

Each exercise contains:
1. **CUDA reference implementation** (`cuda/`) - To understand the problem in familiar terms
2. **Alpaka template** (`alpaka/*_template.cpp`) - Where you'll implement the solution
3. **Alpaka solution** (`alpaka/*_solution.cpp`) - Reference solution for checking your work
4. **README** - Detailed instructions and learning objectives

## Exercises

### Exercise 1: Vector Addition
**Difficulty**: Beginner  
**Time**: 30-45 minutes  
**Topics**: Basic kernel structure, memory management, kernel launch

Learn the fundamentals of Alpaka by implementing simple vector addition (C = A + B).

**Key concepts**:
- Alpaka kernel structure
- Buffer allocation and management
- Memory transfers
- Work division configuration
- `alpaka::uniformElements` for portable indexing

👉 [Start Exercise 1](vector_addition/README.md)

---

### Exercise 2: Block-wise Vector Sum Reduction
**Difficulty**: Intermediate  
**Time**: 1-2 hours  
**Topics**: Shared memory, synchronization, multi-kernel algorithms

Implement a parallel sum reduction using block-level communication and tree-based reduction.

**Key concepts**:
- Shared memory allocation
- Block-level synchronization
- Tree reduction algorithm
- Sequential addressing for efficiency
- Multi-pass kernel strategies

👉 [Start Exercise 2](vector_reduction/README.md)

---

## Learning Path

We recommend completing the exercises in order:

1. **Start with Exercise 1** to learn Alpaka basics
   - Understand the kernel structure
   - Learn about buffer management
   - Get comfortable with the API

2. **Progress to Exercise 2** for advanced topics
   - Master shared memory usage
   - Understand synchronization
   - Learn multi-kernel patterns

## Quick Start

```bash
# Build all exercises at once
cd exercises
./build_all.sh

# Or build individual exercises
cd vector_addition/alpaka
./compile.sh

# Run the template (your work)
./vector_add_template

# Run the solution (to check)
./vector_add_solution
```

## Comparing CUDA and Alpaka

Each exercise includes a CUDA reference implementation. Here's a quick comparison:

### CUDA Code
```cpp
__global__ void kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= 2.0f;
    }
}

// Launch
kernel<<<blocks, threads>>>(d_data, n);
```

### Equivalent Alpaka Code
```cpp
struct Kernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, float* data, int n) const {
        for(auto i : alpaka::uniformElements(acc, n)) {
            data[i] *= 2.0f;
        }
    }
};

// Launch
auto task = alpaka::createTaskKernel<Acc>(workDiv, kernel, d_data, n);
alpaka::enqueue(queue, task);
```

**Benefits of Alpaka**:
- ✅ Portable across CPU and GPU backends
- ✅ Type-safe with better compile-time checking
- ✅ Automatic handling of grid-stride loops
- ✅ Cleaner memory management with RAII buffers
- ✅ Easier to test (can run on CPU without GPU)

## Tips for Success

1. **Read the CUDA version first** - Understand the algorithm before porting
2. **Follow the TODO markers** - They guide you through the implementation
3. **Compile often** - Catch errors early
4. **Use the solution** - But try yourself first!
5. **Experiment** - Try different backends, sizes, and modifications

## Common Gotchas

### 1. Include the right headers
```cpp
#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>
```

### 2. Kernel must be a functor (class with operator())
```cpp
// Not a function!
struct MyKernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, ...) const { }
};
```

### 3. Always wait for queues before reading results
```cpp
alpaka::enqueue(queue, task);
alpaka::wait(queue);  // Don't forget this!
// Now safe to read results
```

### 4. Buffer types must match
```cpp
using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
using BufAcc = alpaka::Buf<DevAcc, Data, Dim, Idx>;
// Can't memcpy between incompatible buffer types!
```

## Directory Structure

```
exercises/
├── README.md                    (this file)
├── vector_addition/
│   ├── README.md
│   ├── cuda/
│   │   ├── vector_add.cu
│   │   └── Makefile
│   └── alpaka/
│       ├── vector_add_template.cpp
│       ├── vector_add_solution.cpp
│       └── compile.sh
└── vector_reduction/
    ├── README.md
    ├── cuda/
    │   ├── vector_reduce.cu
    │   └── Makefile
    └── alpaka/
        ├── vector_reduce_template.cpp
        ├── vector_reduce_solution.cpp
        └── compile.sh
```

## Build System

Each exercise uses a simple bash script (`compile.sh`) that shows all compilation flags explicitly:

```bash
#!/bin/bash
CXX=g++
CXXFLAGS="-std=c++17 -O3 -march=native"
INCLUDES="-I../../../alpaka/include"
BACKEND="-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED"

${CXX} ${CXXFLAGS} ${INCLUDES} ${BACKEND} \
    vector_add_template.cpp \
    -o vector_add_template
```

### Switching Backends

Edit the `BACKEND` variable in `compile.sh`:

```bash
# CPU Serial (default)
BACKEND="-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED"

# CPU with OpenMP (add -fopenmp to CXXFLAGS)
BACKEND="-DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED"
CXXFLAGS="-std=c++17 -O3 -march=native -fopenmp"

# CPU with TBB (link with -ltbb)
BACKEND="-DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED"
LDFLAGS="-ltbb"
```

For GPU backends (CUDA, HIP), refer to the alpaka documentation as setup is more involved.

## Testing Different Backends

To run your code on different accelerators, you can also modify the `main()` function:

```cpp
// Run on all available backends
return alpaka::executeForEachAccTag([=](auto const& tag) { 
    return example(tag); 
});

// Or run on specific backend
auto tag = alpaka::TagCpuSerial{};  // CPU serial
// auto tag = alpaka::TagGpuCudaRt{};  // NVIDIA GPU
// auto tag = alpaka::TagGpuHipRt{};   // AMD GPU
return example(tag);
```

## Troubleshooting

### Build Errors

**Problem**: `alpaka/alpaka.hpp: No such file or directory`  
**Solution**: Check that `ALPAKA_ROOT` in `compile.sh` points to the correct alpaka directory

**Problem**: `ALPAKA_FN_ACC not defined`  
**Solution**: Include `<alpaka/alpaka.hpp>` at the top of your file

**Problem**: Compilation takes very long  
**Solution**: This is normal for header-only template libraries. Consider using `ccache` or precompiled headers

### Runtime Errors

**Problem**: Wrong results  
**Solution**: Check you're synchronizing the queue with `alpaka::wait(queue)` before reading buffers

**Problem**: Segmentation fault  
**Solution**: Verify buffer sizes match the extents you're using

## Additional Resources

- [Alpaka Documentation](https://alpaka.readthedocs.io/)
- [Alpaka GitHub Repository](https://github.com/alpaka-group/alpaka)
- [Alpaka Examples](../alpaka/example/)
- [CUDA to Alpaka Migration Guide](https://alpaka.readthedocs.io/en/latest/usage/implementation/mapping.html)

## Getting Help

If you get stuck:
1. Check the solution file to see the correct implementation
2. Review the README for that exercise
3. Compare with the CUDA version to understand the algorithm
4. Look at other examples in `../alpaka/example/`

## Next Steps

After completing these exercises:
- Explore more complex examples in the alpaka repository
- Try porting your own CUDA kernels to Alpaka
- Experiment with different accelerator backends
- Learn about advanced features (cooperative groups, atomic operations, etc.)

Good luck and happy learning! 🚀
