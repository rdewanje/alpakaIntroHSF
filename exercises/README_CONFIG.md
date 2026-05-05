# Using config.h for Backend Selection

## Overview

The exercises use a **config.h** file for backend selection at compile time. This approach:
- ✅ Makes backend selection explicit and clear
- ✅ Simplifies code (no template meta-programming needed)
- ✅ Mirrors real-world usage in HEP software (e.g., CMSSW)
- ✅ Makes students understand the connection between compile flags and backend

## How It Works

### 1. config.h Defines Types Based on Compile Flags

The `config.h` file uses preprocessor `#if` directives to select the right backend:

```cpp
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
  using Device = alpaka::DevCpu;
  using Platform = alpaka::PlatformCpu;
  using Queue = alpaka::Queue<Device, alpaka::Blocking>;
  using Acc1D = alpaka::AccCpuSerial<Dim1D, Idx>;

#elif defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
  using Device = alpaka::DevCudaRt;
  using Platform = alpaka::PlatformCudaRt;
  using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;
  using Acc1D = alpaka::AccGpuCudaRt<Dim1D, Idx>;

// ... more backends ...
#endif
```

### 2. Your Code Uses These Type Aliases

In your code, you simply use the types defined in `config.h`:

```cpp
#include "../../config.h"

auto main() -> int
{
    // These types are defined by config.h based on compile flags
    auto const platform = Platform{};         // CPU or GPU platform
    auto const device = alpaka::getDevByIdx(platform, 0);
    Queue queue(device);                       // Blocking or NonBlocking queue
    
    // Acc1D is the right accelerator for the selected backend
    auto task = alpaka::createTaskKernel<Acc1D>(workDiv, kernel, ...);
    alpaka::enqueue(queue, task);
}
```

### 3. Compile Flag Selects the Backend

When you compile, the `-D` flag enables one backend:

```bash
# CPU Serial backend
g++ -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED ...

# OpenMP backend
g++ -DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED -fopenmp ...

# CUDA backend (with nvcc)
nvcc -DALPAKA_ACC_GPU_CUDA_ENABLED ...
```

## Type Aliases in config.h

| Type | Description | Example |
|------|-------------|---------|
| `Idx` | Index type | `uint32_t` |
| `Dim1D` | 1D dimension | `alpaka::DimInt<1u>` |
| `Host` | Host device type | `alpaka::DevCpu` |
| `Device` | Accelerator device type | Backend-dependent |
| `Platform` | Platform type | Backend-dependent |
| `Queue` | Queue type | Backend-dependent (Blocking/NonBlocking) |
| `Acc1D` | 1D accelerator type | Backend-dependent |
| `WorkDiv1D` | 1D work division | `alpaka::WorkDivMembers<Dim1D, Idx>` |
| `BufHost<T>` | Host buffer | `alpaka::Buf<Host, T, Dim1D, Idx>` |
| `BufDevice<T>` | Device buffer | `alpaka::Buf<Device, T, Dim1D, Idx>` |

## Comparison with executeForEachAccTag

### Old Approach (executeForEachAccTag)
```cpp
template<alpaka::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int
{
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;
    // ... complex template code ...
}

auto main() -> int
{
    // Runs on ALL enabled backends
    return alpaka::executeForEachAccTag([=](auto const& tag) {
        return example(tag);
    });
}
```

**Problems:**
- Requires understanding template meta-programming
- Hides the actual backend being used
- Runs on multiple backends (confusing for learning)
- Harder to debug

### New Approach (config.h)
```cpp
#include "../../config.h"

auto main() -> int
{
    // Types already defined by config.h
    auto const platform = Platform{};
    auto const device = alpaka::getDevByIdx(platform, 0);
    Queue queue(device);
    
    auto task = alpaka::createTaskKernel<Acc1D>(workDiv, kernel, ...);
    alpaka::enqueue(queue, task);
}
```

**Benefits:**
- Simple, explicit code
- Clear which backend is being used
- No template magic
- Easier to understand and debug
- Matches real-world HEP software patterns

## Switching Backends

Edit `compile.sh` to change the `BACKEND` variable:

```bash
# CPU Serial (default)
BACKEND="-DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED"

# CPU OpenMP (requires -fopenmp)
BACKEND="-DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED"
CXXFLAGS="-std=c++20 -O3 -march=native -fopenmp"

# CPU TBB (requires -ltbb)
BACKEND="-DALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED"
LDFLAGS="-ltbb"

# NVIDIA CUDA (requires nvcc)
CXX=nvcc
BACKEND="-DALPAKA_ACC_GPU_CUDA_ENABLED"
CXXFLAGS="-std=c++20 -O3"

# AMD HIP (requires hipcc)
CXX=hipcc
BACKEND="-DALPAKA_ACC_GPU_HIP_ENABLED"
CXXFLAGS="-std=c++20 -O3"
```

## Available Backends

| Backend | Flag | Compiler | Additional Flags |
|---------|------|----------|------------------|
| CPU Serial | `ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED` | g++/clang++ | None |
| CPU OpenMP | `ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED` | g++/clang++ | `-fopenmp` |
| CPU TBB | `ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED` | g++/clang++ | `-ltbb` |
| CPU Threads | `ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED` | g++/clang++ | `-lpthread` |
| NVIDIA CUDA | `ALPAKA_ACC_GPU_CUDA_ENABLED` | nvcc | CUDA toolkit |
| AMD HIP | `ALPAKA_ACC_GPU_HIP_ENABLED` | hipcc | ROCm |
| Intel GPU | `ALPAKA_SYCL_ONEAPI_GPU` | icpx | oneAPI |

## Example: Running on Different Backends

```bash
# Compile for CPU Serial
./compile.sh
./vector_add_solution
# Output: Using alpaka accelerator: AccCpuSerial<1,unsigned int>

# Edit compile.sh: BACKEND="-DALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED"
#                   Add: -fopenmp to CXXFLAGS
./compile.sh
./vector_add_solution
# Output: Using alpaka accelerator: AccCpuOmp2Blocks<1,unsigned int>
```

## Why This Approach?

This is how **real HEP software** works:
- **CMSSW** (CMS Software) uses a similar config.h approach
- **Production code** compiles once for a specific backend, not all backends
- **Deployment** chooses backend at compile time, not runtime
- **Performance** is better when the backend is known at compile time

Students learn:
- How compile-time configuration works
- The connection between preprocessor flags and code
- Real-world software engineering practices
- How to write portable but efficient code

## Summary

| Aspect | executeForEachAccTag | config.h |
|--------|----------------------|----------|
| Complexity | High (templates) | Low (simple types) |
| Clarity | Hides backend | Explicit backend |
| Runs on | All enabled backends | One selected backend |
| Real-world usage | Demos/testing | Production code |
| Educational value | Template metaprogramming | Portable software design |
| Debugging | Harder | Easier |

The config.h approach is **simpler, clearer, and more realistic** for teaching!
