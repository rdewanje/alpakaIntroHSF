# Exercise 1: Vector Addition

## Overview
This exercise teaches you how to implement vector addition using Alpaka, a C++ abstraction library for accelerator programming. You'll learn how to port CUDA kernels to Alpaka's portable abstraction.

## Learning Objectives
- Understand the basic structure of an Alpaka kernel
- Learn about buffer management in Alpaka
- Understand memory transfers between host and device
- Configure and launch kernels using Alpaka's work division
- Compare CUDA and Alpaka programming models

## Problem Description
Implement `C = A + B` where A, B, and C are vectors of floating-point numbers.

## Directory Structure
```
vector_addition/
├── README.md              (this file)
├── cuda/
│   ├── vector_add.cu      (CUDA reference implementation)
│   └── Makefile
└── alpaka/
    ├── vector_add_template.cpp  (YOUR WORK - fill in the TODOs)
    ├── vector_add_solution.cpp  (reference solution)
    └── compile.sh               (build script)
```

## Instructions

### Step 1: Study the CUDA Version
First, look at the CUDA implementation to understand the problem:

```bash
cd cuda
make
./vector_add
```

**Key points in CUDA version:**
- Kernel uses `blockIdx.x * blockDim.x + threadIdx.x` for indexing
- Manual grid/block configuration
- Explicit memory management with `cudaMalloc`, `cudaMemcpy`, `cudaFree`

### Step 2: Understand the Alpaka Template
Open `alpaka/vector_add_template.cpp` and review:

1. **The kernel structure** - How Alpaka kernels are class-based with `operator()`
2. **uniformElements** - Alpaka's abstraction for thread indexing
3. **Buffer types** - How Alpaka handles memory on different devices
4. **Work division** - How Alpaka configures kernel launches

### Step 3: Complete the TODOs
Work through each TODO in the template file:

**TODO 1: Implement the kernel**
```cpp
for(auto i : alpaka::uniformElements(acc, numElements))
{
    C[i] = A[i] + B[i];
}
```
The `uniformElements` iterator automatically handles the grid-stride loop pattern.

**TODO 2-3: Queue setup**
```cpp
using QueueProperty = alpaka::Blocking;
using QueueAcc = alpaka::Queue<Acc, QueueProperty>;
```

**TODO 4-5: Host buffers**
```cpp
using BufHost = alpaka::Buf<DevHost, Data, Dim, Idx>;
BufHost bufHostA(alpaka::allocBuf<Data, Idx>(devHost, extent));
```

**TODO 6: Device buffers**
Similar to host buffers but use `DevAcc` instead of `DevHost`.

**TODO 7: Memory copies**
```cpp
alpaka::memcpy(queue, bufAccA, bufHostA);  // Host -> Device
```

**TODO 8-9: Kernel launch**
```cpp
alpaka::KernelCfg<Acc> const kernelCfg = {extent, elementsPerThread};
auto const workDiv = alpaka::getValidWorkDiv(...);
auto const taskKernel = alpaka::createTaskKernel<Acc>(...);
alpaka::enqueue(queue, taskKernel);
```

**TODO 10: Copy results back**
```cpp
alpaka::memcpy(queue, bufHostC, bufAccC);  // Device -> Host
```

### Step 4: Build and Test

```bash
cd ../alpaka

# Build both template and solution
./compile.sh

# Run your template
./vector_add_template

# Compare with solution
./vector_add_solution
```

**What the compile script does:**
```bash
g++ -std=c++17 -O3 -march=native \
    -I../../../alpaka/include \
    -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED \
    vector_add_template.cpp \
    -o vector_add_template
```

You can edit `compile.sh` to:
- Use different compiler: `CXX=clang++ ./compile.sh`
- Enable OpenMP backend: Change `BACKEND` to `-DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED` and add `-fopenmp`
- Use CUDA backend: Requires more setup (see alpaka documentation)

### Step 5: Verify Your Understanding

After completing the exercise, you should be able to answer:

1. How does `alpaka::uniformElements` simplify thread indexing compared to CUDA?
2. What's the difference between `BufHost` and `BufAcc`?
3. Why does Alpaka use `alpaka::memcpy` instead of separate functions for HtoD and DtoH?
4. How does Alpaka's `getValidWorkDiv` differ from manually calculating grid/block sizes in CUDA?
5. What accelerator backends can this code run on without modification?

## CUDA vs Alpaka Comparison

| Aspect | CUDA | Alpaka |
|--------|------|--------|
| Thread Index | `blockIdx.x * blockDim.x + threadIdx.x` | `alpaka::uniformElements(acc, n)` |
| Memory Allocation | `cudaMalloc(&ptr, size)` | `alpaka::allocBuf<T, Idx>(dev, extent)` |
| Memory Copy | `cudaMemcpy(dst, src, size, direction)` | `alpaka::memcpy(queue, dst, src)` |
| Kernel Launch | `kernel<<<grid, block>>>()` | `alpaka::enqueue(queue, taskKernel)` |
| Synchronization | `cudaDeviceSynchronize()` | `alpaka::wait(queue)` |
| Portability | NVIDIA GPUs only | CPU, NVIDIA, AMD, Intel, and more |

## Hints

- If you get compilation errors about missing types, check that you've defined all the `using` declarations.
- Make sure to synchronize the queue with `alpaka::wait(queue)` before accessing results.
- The solution file is there for reference, but try to complete it yourself first!

## Expected Output

```
Using alpaka accelerator: AccCpuSerial
Execution results correct!
```

## Next Steps
Once you've completed this exercise, move on to **Exercise 2: Vector Reduction** which introduces shared memory and block-level synchronization.
