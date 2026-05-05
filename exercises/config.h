/* Alpaka Backend Configuration
 *
 * This file defines type aliases for the selected backend.
 * The backend is selected at compile time via -DALPAKA_ACC_* flags.
 *
 * Adapted from CMSSW (https://github.com/cms-sw/cmssw/)
 */

#ifndef CONFIG_H
#define CONFIG_H

#include <alpaka/alpaka.hpp>

// Index type
using Idx = uint32_t;

// Dimensions
using Dim1D = alpaka::DimInt<1u>;

// Work division
using WorkDiv1D = alpaka::WorkDivMembers<Dim1D, Idx>;

// Host types (always CPU)
using Host = alpaka::DevCpu;
using HostPlatform = alpaka::PlatformCpu;

// Backend selection based on compile-time flags
// Only ONE of these should be defined

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
  // NVIDIA CUDA backend
  using Device = alpaka::DevCudaRt;
  using Platform = alpaka::PlatformCudaRt;
  using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;
  using Acc1D = alpaka::AccGpuCudaRt<Dim1D, Idx>;

#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
  // AMD HIP/ROCm backend
  using Device = alpaka::DevHipRt;
  using Platform = alpaka::PlatformHipRt;
  using Queue = alpaka::Queue<Device, alpaka::NonBlocking>;
  using Acc1D = alpaka::AccGpuHipRt<Dim1D, Idx>;

#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
  // CPU serial backend (single-threaded)
  using Device = alpaka::DevCpu;
  using Platform = alpaka::PlatformCpu;
  using Queue = alpaka::Queue<Device, alpaka::Blocking>;
  using Acc1D = alpaka::AccCpuSerial<Dim1D, Idx>;

#elif defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
  // CPU OpenMP backend (parallel blocks)
  using Device = alpaka::DevCpu;
  using Platform = alpaka::PlatformCpu;
  using Queue = alpaka::Queue<Device, alpaka::Blocking>;
  using Acc1D = alpaka::AccCpuOmp2Blocks<Dim1D, Idx>;

#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  // CPU TBB backend (parallel blocks)
  using Device = alpaka::DevCpu;
  using Platform = alpaka::PlatformCpu;
  using Queue = alpaka::Queue<Device, alpaka::Blocking>;
  using Acc1D = alpaka::AccCpuTbbBlocks<Dim1D, Idx>;

#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
  // CPU threads backend (parallel threads)
  using Device = alpaka::DevCpu;
  using Platform = alpaka::PlatformCpu;
  using Queue = alpaka::Queue<Device, alpaka::Blocking>;
  using Acc1D = alpaka::AccCpuThreads<Dim1D, Idx>;

#else
  #error "No Alpaka backend enabled! Please define one of: ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED, ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED, ALPAKA_ACC_GPU_CUDA_ENABLED, ALPAKA_ACC_GPU_HIP_ENABLED"
#endif

// Common buffer types
template<typename T>
using BufHost = alpaka::Buf<Host, T, Dim1D, Idx>;

template<typename T>
using BufDevice = alpaka::Buf<Device, T, Dim1D, Idx>;

#endif  // CONFIG_H
