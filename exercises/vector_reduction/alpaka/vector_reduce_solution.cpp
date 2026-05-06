/* Vector Reduction - Alpaka Solution
 *
 * This is the complete solution for the block-wise reduction exercise.
 * Compare with the template to see what was needed to be implemented.
 */

#include "../../config.h"

#include <iostream>

// Helper: A simple array wrapper for shared memory
template<typename T, uint32_t size>
struct SharedArray
{
    T data[size];

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator[](uint32_t index) -> T&
    {
        return data[index];
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto operator[](uint32_t index) const -> T const&
    {
        return data[index];
    }
};

// SOLUTION: Complete reduction kernel implementation
template<uint32_t TBlockSize, typename T>
struct ReduceKernel
{
    template<typename TAcc, typename TElem>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TElem const* const input,
        TElem* output,
        Idx const n) const -> void
    {
        // Declare shared memory for block-level reduction
        auto& sdata = alpaka::declareSharedVar<SharedArray<T, TBlockSize>, __COUNTER__>(acc);

        // Get thread and block indices
        auto const blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
        auto const threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
        auto const blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];
        auto const gridDim = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];

        // Each thread processes multiple elements using grid-stride loop
        T sum = 0;
        for(Idx i = blockIdx * blockDim + threadIdx; i < n; i += gridDim * blockDim)
        {
            sum += input[i];
        }

        // Store to shared memory
        sdata[threadIdx] = sum;
        alpaka::syncBlockThreads(acc);

        // Tree reduction in shared memory
        for(Idx stride = blockDim / 2; stride > 0; stride >>= 1)
        {
            if(threadIdx < stride)
            {
                sdata[threadIdx] += sdata[threadIdx + stride];
            }
            alpaka::syncBlockThreads(acc);
        }

        // Thread 0 writes result
        if(threadIdx == 0)
        {
            output[blockIdx] = sdata[0];
        }
    }
};

auto main() -> int
{
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc1D>() << std::endl;

    // Get devices and create queue
    auto const platformHost = HostPlatform{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    auto const platform = Platform{};
    auto const device = alpaka::getDevByIdx(platform, 0);

    Queue queue(device);

    // Problem size
    Idx const numElements = 1 << 20;  // 1M elements

    using Data = float;

    // Block size must be compile-time constant for shared memory allocation
    // We'll use 256 for the kernel template, but adjust runtime work division for CPU serial
    constexpr uint32_t maxBlockSize = 256;

    // Query device properties
    auto const devProps = alpaka::getAccDevProps<Acc1D>(device);
    Idx const maxThreadsPerBlock = static_cast<Idx>(devProps.m_blockThreadExtentMax[0]);

    // Runtime block size: 256 for GPU, 1 for CPU serial
    Idx const runtimeBlockSize = std::min(Idx{maxBlockSize}, maxThreadsPerBlock);

    // Calculate number of blocks
    // For parallel backends: enough blocks to cover all elements
    // For serial backend (1 thread/block): limit blocks so grid-stride loop works
    //   - Use fewer blocks and let each thread process multiple elements
    //   - Good rule: use ~1024 blocks max for serial, let grid-stride do the rest
    Idx const maxBlocks = (runtimeBlockSize == 1) ? 1024 : (numElements + runtimeBlockSize - 1) / runtimeBlockSize;
    Idx const numBlocks = std::min(maxBlocks, (numElements + runtimeBlockSize - 1) / runtimeBlockSize);

    std::cout << "Reducing " << numElements << " elements" << std::endl;
    std::cout << "Using " << numBlocks << " blocks with " << runtimeBlockSize << " threads each" << std::endl;

    if(runtimeBlockSize == 1)
    {
        std::cout << "Note: CPU serial backend (1 thread/block) - using grid-stride pattern\n";
        std::cout << "      Each thread processes ~" << (numElements / numBlocks) << " elements\n";
    }

    // Allocate host buffer
    alpaka::Vec<Dim1D, Idx> const extentInput(numElements);
    BufHost<Data> bufHostInput(alpaka::allocBuf<Data, Idx>(devHost, extentInput));

    // Initialize input (all ones for simple verification)
    for(Idx i = 0; i < numElements; ++i)
    {
        bufHostInput[i] = 1.0f;
    }

    // Allocate device buffers
    alpaka::Vec<Dim1D, Idx> const extentPartial(numBlocks);
    alpaka::Vec<Dim1D, Idx> const extentOutput(1);

    BufDevice<Data> bufDevInput(alpaka::allocBuf<Data, Idx>(device, extentInput));
    BufDevice<Data> bufDevPartial(alpaka::allocBuf<Data, Idx>(device, extentPartial));
    BufDevice<Data> bufDevOutput(alpaka::allocBuf<Data, Idx>(device, extentOutput));

    // Copy input to device
    alpaka::memcpy(queue, bufDevInput, bufHostInput);

    // Create kernel instances
    // Template parameter is compile-time max block size (for shared memory)
    // Runtime work division will use actual block size
    ReduceKernel<maxBlockSize, Data> kernel1, kernel2;

    // Define work divisions manually
    // Key difference between CPU and GPU:
    //   GPU: many threads per block, few elements per thread
    //   CPU: 1 thread per block, many elements per thread
    Idx const elementsPerThread1 = (runtimeBlockSize == 1)
        ? (numElements + numBlocks - 1) / numBlocks  // CPU: many elements per thread
        : 1;                                          // GPU: 1 element per thread (grid-stride handles rest)

    WorkDiv1D workDiv1{
        alpaka::Vec<Dim1D, Idx>(numBlocks),         // Grid: number of blocks
        alpaka::Vec<Dim1D, Idx>(runtimeBlockSize),  // Block: threads per block
        alpaka::Vec<Dim1D, Idx>(elementsPerThread1)};  // Thread: elements per thread

    // Second reduction
    Idx const elementsPerThread2 = (runtimeBlockSize == 1)
        ? numBlocks  // CPU: process all partial sums in one thread
        : 1;         // GPU: 1 element per thread

    WorkDiv1D workDiv2{
        alpaka::Vec<Dim1D, Idx>(1),                    // Grid: 1 block
        alpaka::Vec<Dim1D, Idx>(runtimeBlockSize),     // Block: threads per block
        alpaka::Vec<Dim1D, Idx>(elementsPerThread2)};  // Thread: elements per thread

    std::cout << "First reduction work division: " << numBlocks << " blocks × "
              << runtimeBlockSize << " threads × " << elementsPerThread1 << " elements/thread" << std::endl;
    std::cout << "Second reduction work division: 1 block × "
              << runtimeBlockSize << " threads × " << elementsPerThread2 << " elements/thread" << std::endl;

    // First reduction: input -> partial sums
    auto const taskKernel1 = alpaka::createTaskKernel<Acc1D>(
        workDiv1,
        kernel1,
        std::data(bufDevInput),
        std::data(bufDevPartial),
        numElements);

    alpaka::enqueue(queue, taskKernel1);
    alpaka::wait(queue);

    // Second reduction: partial sums -> final result
    auto const taskKernel2 = alpaka::createTaskKernel<Acc1D>(
        workDiv2,
        kernel2,
        std::data(bufDevPartial),
        std::data(bufDevOutput),
        numBlocks);

    alpaka::enqueue(queue, taskKernel2);
    alpaka::wait(queue);

    // Copy result back
    Data h_result = 0.0f;
    auto resultView = alpaka::createView(devHost, &h_result, extentOutput);
    alpaka::memcpy(queue, resultView, bufDevOutput);
    alpaka::wait(queue);

    // Verify result
    Data expected = static_cast<Data>(numElements);

    std::cout << "Result: " << h_result << std::endl;
    std::cout << "Expected: " << expected << std::endl;

    if(std::abs(h_result - expected) < 1e-2f)
    {
        std::cout << "Reduction result correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Reduction result incorrect!" << std::endl;
        std::cout << "Error: " << std::abs(h_result - expected) << std::endl;
        return EXIT_FAILURE;
    }
}
