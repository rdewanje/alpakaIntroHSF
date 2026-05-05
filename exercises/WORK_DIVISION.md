# Understanding Work Division in Alpaka

## The Key Insight: CPU vs GPU Work Distribution

### GPU Strategy
```
Many blocks × Many threads per block × Few elements per thread
Example: 512 blocks × 256 threads × 1 element = 131,072 parallel units
```

### CPU Serial Strategy  
```
Few blocks × 1 thread per block × Many elements per thread  
Example: 1024 blocks × 1 thread × 1024 elements = 1,048,576 elements processed
```

## Work Division Structure

```cpp
WorkDiv1D workDiv{
    alpaka::Vec<Dim1D, Idx>(blocksPerGrid),      // Grid dimension
    alpaka::Vec<Dim1D, Idx>(threadsPerBlock),    // Block dimension  
    alpaka::Vec<Dim1D, Idx>(elementsPerThread)}; // Elements per thread
```

### The Three Levels

1. **Grid** (`blocksPerGrid`): How many blocks to launch
2. **Block** (`threadsPerBlock`): How many threads cooperate within each block
3. **Thread** (`elementsPerThread`): How many elements each thread processes

## Backend Constraints

| Backend | `threadsPerBlock` | `elementsPerThread` | Strategy |
|---------|-------------------|---------------------|----------|
| CPU Serial | **Must be 1** | **Large** (100s-1000s) | Each "block" processes chunk sequentially |
| CPU OpenMP/TBB | 1-N | Medium | Parallel blocks, sequential within block |
| GPU (CUDA/HIP) | 64-1024 | Small (1-8) | Massive parallelism |

## Querying Device Properties

Always query the device to respect backend constraints:

```cpp
auto const devProps = alpaka::getAccDevProps<Acc1D>(device);
Idx const maxThreadsPerBlock = static_cast<Idx>(devProps.m_blockThreadExtentMax[0]);

// Adapt to backend
Idx const threadsPerBlock = std::min(Idx{256}, maxThreadsPerBlock);
```

For CPU Serial: `maxThreadsPerBlock == 1`  
For GPU: `maxThreadsPerBlock == 1024` (typically)

## What `uniformElements` Does

```cpp
for(auto i : alpaka::uniformElements(acc, numElements))
{
    C[i] = A[i] + B[i];
}
```

**Under the hood:**
1. Calculates: `globalThreadIdx = blockIdx * blockDim + threadIdx`
2. Calculates: `stride = gridDim * blockDim * elementsPerThread`
3. Iterates: `for(i = globalThreadIdx * elementsPerThread; i < numElements; i += stride)`

### CUDA Equivalent (Grid-Stride Loop)
```cuda
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for(int i = tid; i < n; i += stride) {
    C[i] = A[i] + B[i];
}
```

## Common Pitfall: Using Compile-Time vs Runtime Block Size

❌ **Wrong:**
```cpp
template<uint32_t TBlockSize, typename T>
struct Kernel {
    auto operator()(...) {
        auto globalIdx = blockIdx * TBlockSize + threadIdx;  // WRONG!
        auto gridStride = gridDim * TBlockSize;              // WRONG!
    }
};
```

✅ **Correct:**
```cpp
template<uint32_t TBlockSize, typename T>
struct Kernel {
    auto operator()(TAcc const& acc, ...) {
        auto blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];
        auto globalIdx = blockIdx * blockDim + threadIdx;   // Correct!
        auto gridStride = gridDim * blockDim;               // Correct!
    }
};
```

**Why:** 
- `TBlockSize` is compile-time (for shared memory allocation)
- Runtime `blockDim` may be different (e.g., CPU serial has blockDim=1)

## Example: Vector Addition

### For GPU (CUDA)
```cpp
Idx const threadsPerBlock = 256;
Idx const elementsPerThread = 8;
Idx const blocksPerGrid = (numElements + (threadsPerBlock * elementsPerThread) - 1)
                         / (threadsPerBlock * elementsPerThread);

WorkDiv1D workDiv{
    alpaka::Vec<Dim1D, Idx>(blocksPerGrid),      // 512 blocks
    alpaka::Vec<Dim1D, Idx>(threadsPerBlock),    // 256 threads
    alpaka::Vec<Dim1D, Idx>(elementsPerThread)}; // 8 elements
// Total: 512 × 256 × 8 = 1,048,576 elements
```

### For CPU Serial
```cpp
Idx const threadsPerBlock = 1;    // Must be 1
Idx const elementsPerThread = (numElements + numBlocks - 1) / numBlocks;
Idx const blocksPerGrid = std::min(1024, numElements);

WorkDiv1D workDiv{
    alpaka::Vec<Dim1D, Idx>(blocksPerGrid),      // 1024 blocks
    alpaka::Vec<Dim1D, Idx>(threadsPerBlock),    // 1 thread
    alpaka::Vec<Dim1D, Idx>(elementsPerThread)}; // ~1024 elements
// Total: 1024 × 1 × 1024 = ~1,048,576 elements
```

## Example: Reduction

Reduction is more complex because it uses shared memory for intra-block cooperation.

### Kernel Template Parameter
```cpp
template<uint32_t TBlockSize, typename T>  // TBlockSize = compile-time max (e.g., 256)
struct ReduceKernel {
    // SharedArray sized at compile time
    auto& sdata = alpaka::declareSharedVar<SharedArray<T, TBlockSize>, __COUNTER__>(acc);
    
    // But use RUNTIME block dimension for calculations
    auto blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];
    auto globalIdx = blockIdx * blockDim + threadIdx;  // Not TBlockSize!
};
```

### Work Division
```cpp
constexpr uint32_t maxBlockSize = 256;  // For shared memory

// Runtime block size (1 for CPU serial, 256 for GPU)
Idx const runtimeBlockSize = std::min(Idx{maxBlockSize}, maxThreadsPerBlock);

// CPU: use elementsPerThread; GPU: use grid-stride
Idx const elementsPerThread = (runtimeBlockSize == 1)
    ? (numElements + numBlocks - 1) / numBlocks  // CPU: many elements
    : 1;                                          // GPU: grid-stride

WorkDiv1D workDiv{
    alpaka::Vec<Dim1D, Idx>(numBlocks),
    alpaka::Vec<Dim1D, Idx>(runtimeBlockSize),
    alpaka::Vec<Dim1D, Idx>(elementsPerThread)};
```

## Summary

| Aspect | GPU | CPU Serial |
|--------|-----|------------|
| Parallelism | Massive (1000s of threads) | None (1 thread at a time) |
| `threadsPerBlock` | 64-1024 | **1** (enforced) |
| `elementsPerThread` | Small (1-8) | **Large** (100s-1000s) |
| Work distribution | Spread across threads | Sequential per block |
| Kernel indexing | Use **runtime** `blockDim` | Use **runtime** `blockDim` |
| Shared memory | Full cooperation | No cooperation (1 thread) |

**Golden Rule:** Always use runtime dimensions (`blockDim`) in your kernel, not compile-time template parameters!
