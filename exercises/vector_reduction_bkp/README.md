# Exercise 2: Block-wise Vector Sum Reduction

## Overview
This exercise teaches you how to implement a parallel reduction algorithm using Alpaka. You'll learn about shared memory, block-level synchronization, and multi-kernel reductions.

## Learning Objectives
- Understand parallel reduction algorithms
- Use shared memory in Alpaka kernels
- Implement block-level synchronization
- Launch multiple kernels in sequence
- Handle hierarchical parallelism (grid → block → thread)

## Problem Description
Compute the sum of all elements in a vector using a two-phase reduction:
1. **Phase 1**: Each block reduces its portion of the input to a partial sum
2. **Phase 2**: A single block reduces all partial sums to the final result

This is more complex than vector addition because it requires:
- Intra-block communication via shared memory
- Thread synchronization within blocks
- Multiple kernel launches

## Directory Structure
```
vector_reduction/
├── README.md              (this file)
├── cuda/
│   ├── vector_reduce.cu   (CUDA reference implementation)
│   └── Makefile
└── alpaka/
    ├── vector_reduce_template.cpp  (YOUR WORK - fill in the TODOs)
    ├── vector_reduce_solution.cpp  (reference solution)
    └── compile.sh                  (build script)
```

## Instructions

### Step 1: Study the CUDA Version
First, understand the CUDA implementation:

```bash
cd cuda
make
./vector_reduce
```

**Key concepts in CUDA version:**
- `__shared__` memory for block-level reduction
- `__syncthreads()` for synchronization
- Tree-based reduction pattern
- Sequential addressing to avoid warp divergence
- Two-phase reduction strategy

### Step 2: Understand the Algorithm

The reduction happens in stages:

```
Input: [1, 1, 1, 1, 1, 1, 1, 1]  (8 elements, 2 blocks of 4 threads)

Block 0:           Block 1:
Thread: 0 1 2 3    Thread: 0 1 2 3
Load:  [1 1 1 1]   Load:  [1 1 1 1]

After stride=2:
       [2 _ 2 _]          [2 _ 2 _]
After stride=1:
       [4 _ _ _]          [4 _ _ _]

Partial sums: [4, 4]

Final reduction (1 block):
Thread: 0 1
Load:  [4 4]
After stride=1:
       [8 _]

Result: 8
```

### Step 3: Complete the TODOs

**TODO 1: Implement the reduction kernel**

Key components:
```cpp
// Declare shared memory
auto& sdata = alpaka::declareSharedVar<SharedArray<T, TBlockSize>, __COUNTER__>(acc);

// Get indices
auto const blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
auto const threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

// Grid-stride accumulation
T sum = 0;
for(TIdx idx = globalIdx; idx < n; idx += gridStride)
{
    sum += input[idx];
}
sdata[threadIdx] = sum;

// Synchronize
alpaka::syncBlockThreads(acc);

// Tree reduction
for(uint32_t stride = TBlockSize / 2; stride > 0; stride >>= 1)
{
    if(threadIdx < stride)
    {
        sdata[threadIdx] += sdata[threadIdx + stride];
    }
    alpaka::syncBlockThreads(acc);
}

// Write result
if(threadIdx == 0)
{
    output[blockIdx] = sdata[0];
}
```

**TODO 2-3: Queue and device setup** (similar to Exercise 1)

**TODO 4-6: Buffer allocation**
You need three buffers:
- Input buffer: size `numElements`
- Partial sums buffer: size `numBlocks`
- Output buffer: size `1`

**TODO 7-8: Kernel configuration**
Define two work divisions:
```cpp
WorkDiv workDiv1(
    alpaka::Vec<Dim, Idx>(numBlocks),   // grid size
    alpaka::Vec<Dim, Idx>(blockSize),   // block size
    alpaka::Vec<Dim, Idx>(1));          // elements per thread
```

**TODO 9: Launch both kernels**
```cpp
// First kernel: input → partial sums
auto taskKernel1 = alpaka::createTaskKernel<Acc>(
    workDiv1, kernel1,
    std::data(bufAccInput),
    std::data(bufAccPartial),
    numElements);
alpaka::enqueue(queue, taskKernel1);
alpaka::wait(queue);

// Second kernel: partial sums → final result
auto taskKernel2 = alpaka::createTaskKernel<Acc>(
    workDiv2, kernel2,
    std::data(bufAccPartial),
    std::data(bufAccOutput),
    numBlocks);
alpaka::enqueue(queue, taskKernel2);
```

**TODO 10: Copy result**
```cpp
Data h_result;
auto resultView = alpaka::createView(devHost, &h_result, extentOutput);
alpaka::memcpy(queue, resultView, bufAccOutput);
```

### Step 4: Build and Test

```bash
cd ../alpaka

# Build both template and solution
./compile.sh

# Run your template
./vector_reduce_template

# Compare with solution
./vector_reduce_solution
```

**What the compile script does:**
```bash
g++ -std=c++17 -O3 -march=native \
    -I../../../alpaka/include \
    -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED \
    vector_reduce_template.cpp \
    -o vector_reduce_template
```

### Step 5: Verify Your Understanding

After completing the exercise:

1. Why do we need shared memory for reduction?
2. Why is synchronization required after each reduction step?
3. Why use sequential addressing instead of interleaved addressing?
4. Why do we need two kernel launches?
5. How would you optimize this for very large arrays?

## Algorithm Deep Dive

### Why Tree Reduction?

A sequential reduction would be:
```cpp
sum = 0;
for(int i = 0; i < n; i++)
    sum += array[i];  // O(n) time, no parallelism!
```

Tree reduction achieves O(log n) parallel time:
- Step 1: n/2 pairs add in parallel → n/2 results
- Step 2: n/4 pairs add in parallel → n/4 results
- ...
- Step log₂(n): final result

### Why Sequential Addressing?

```cpp
// BAD: Interleaved (causes divergence)
stride = 1
if (tid % 2 == 0)  // Half the threads in each warp diverge!
    sdata[tid] += sdata[tid + 1];

// GOOD: Sequential (no divergence)
stride = blockSize / 2
if (tid < stride)  // All threads in first half of warp execute together!
    sdata[tid] += sdata[tid + stride];
```

### Why Two Phases?

- **Can't synchronize across blocks** - only within a block
- **Phase 1**: Each block independently reduces its data
- **Phase 2**: One block reduces the per-block results
- Alternative: Recursively launch kernels until result fits in one block

## CUDA vs Alpaka: Shared Memory & Synchronization

| Aspect | CUDA | Alpaka |
|--------|------|--------|
| Shared Memory | `__shared__ float sdata[256];` | `alpaka::declareSharedVar<Array<T,N>, ID>(acc)` |
| Block Sync | `__syncthreads()` | `alpaka::syncBlockThreads(acc)` |
| Block Index | `blockIdx.x` | `alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0]` |
| Thread Index | `threadIdx.x` | `alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0]` |
| Grid Size | `gridDim.x` | `alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0]` |

## Common Pitfalls

1. **Forgetting synchronization** - Results in race conditions
2. **Wrong buffer sizes** - Partial sums buffer must be `numBlocks` in size
3. **Not handling partial blocks** - Last block might have fewer than `blockSize` elements
4. **Incorrect stride calculation** - Must be powers of 2 for tree reduction
5. **Missing second kernel** - Forgetting to reduce the partial sums

## Expected Output

```
Using alpaka accelerator: AccCpuSerial
Reducing 1048576 elements
Using 4096 blocks with 256 threads each
Result: 1048576
Expected: 1048576
Reduction result correct!
```

## Optimization Opportunities

Once you have a working solution, consider:

1. **Warp-level optimizations** - No sync needed within a warp (last 32 elements)
2. **Loop unrolling** - Manually unroll the last few reduction steps
3. **Multiple elements per thread** - Load more data in phase 1
4. **Atomic operations** - For very small reductions
5. **Cooperative groups** - More flexible synchronization (CUDA 9+)

## Further Reading

- [NVIDIA Parallel Reduction Presentation](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- Alpaka documentation on shared memory
- CUB library for optimized reductions

## Next Steps

Congratulations! You've completed both basic Alpaka exercises. You now understand:
- Basic kernel patterns (vector addition)
- Advanced patterns (block-wise reduction)
- Memory hierarchy (global, shared)
- Synchronization primitives
- Multi-kernel algorithms

Try experimenting with:
- Different accelerator backends (CUDA, HIP, OpenMP, etc.)
- Different data types
- Different reduction operations (max, min, product)
- Larger problem sizes
