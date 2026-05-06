/* Vector Addition - Alpaka Solution
 *
 * This is the complete solution for the vector addition exercise.
 * Compare with the template to see what was needed to be implemented.
 */

#include "../../config.h"
#include <iostream>
#include <random>


// Vector addition
class VectorAddKernel
{
  public:
    template <typename TAcc, typename TElem>
    ALPAKA_FN_ACC auto operator()(TAcc const&        acc,
                                  TElem const* const A,
                                  TElem const* const B,
                                  TElem* const       C,
                                  Idx const          numElements) const -> void
    {
        // uniformElements() provides a portable iteration pattern
        // Under the hood, it does:
        //   1. Calculates: globalThreadIdx = blockIdx * blockDim + threadIdx
        //   2. Calculates: gridStride = gridDim * blockDim * elementsPerThread
        //   3. Loops: for(i = globalThreadIdx * elementsPerThread; i < numElements; i += gridStride)
        //
        // How to get indices in Alpaka (if you need them manually):
        //   auto blockIdx  = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0];
        //   auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];
        //   auto blockDim  = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0];
        //   auto gridDim   = alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(acc)[0];
        //
        // This is equivalent to CUDA's grid-stride loop:
        //   int tid = blockIdx.x * blockDim.x + threadIdx.x;
        //   int stride = gridDim.x * blockDim.x;
        //   for(int i = tid; i < n; i += stride) { ... }
        //
        // Benefits:
        //   - Handles case where numElements > total_threads
        //   - Each thread processes multiple elements if needed
        //   - Portable across different accelerators
        for (auto i : alpaka::uniformElements(acc, numElements))
        {
            C[i] = A[i] + B[i];
        }
    }
};

auto main() -> int
{
    //get name of accelerator being used 
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc1D>() << std::endl;
    
    //get platform and select host and device 
    auto const platformHost = HostPlatform{};
    auto const devHost      = alpaka::getDevByIdx(platformHost, 0);

    auto const platform = Platform{};
    auto const device   = alpaka::getDevByIdx(platform, 0);
    
    //define our queue
    Queue queue(device);

    // Problem size
    Idx const                     numElements       = 1 << 24;  // 1M elements
    Idx const                     elementsPerThread = 8u;
    alpaka::Vec<Dim1D, Idx> const extent(numElements);

    using Data = float;

    // SOLUTION 3: Allocate host buffers
    BufHost<Data> bufHostA(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost<Data> bufHostB(alpaka::allocBuf<Data, Idx>(devHost, extent));
    BufHost<Data> bufHostC(alpaka::allocBuf<Data, Idx>(devHost, extent));

    // Initialize input data
    std::random_device                   rd{};
    std::default_random_engine           eng{rd()};
    std::uniform_real_distribution<Data> dist(1.0f, 42.0f);

    for (Idx i = 0; i < numElements; ++i)
    {
        bufHostA[i] = dist(eng);
        bufHostB[i] = dist(eng);
        bufHostC[i] = 0.0f;
    }

    // SOLUTION 4: Allocate device buffers
    BufDevice<Data> bufDevA(alpaka::allocBuf<Data, Idx>(device, extent));
    BufDevice<Data> bufDevB(alpaka::allocBuf<Data, Idx>(device, extent));
    BufDevice<Data> bufDevC(alpaka::allocBuf<Data, Idx>(device, extent));

    // SOLUTION 5: Copy host to device
    alpaka::memcpy(queue, bufDevA, bufHostA);
    alpaka::memcpy(queue, bufDevB, bufHostB);
    alpaka::memcpy(queue, bufDevC, bufHostC);

    // SOLUTION 6: Configure kernel launch - manual work division
    VectorAddKernel kernel;

    // Calculate work division manually
    // Work division has 3 levels: Grid (blocks), Block (threads), Thread (elements)
    //
    // Note: Different backends have different constraints:
    //   - CPU Serial: threadsPerBlock must be 1
    //   - GPU backends: threadsPerBlock typically 64-1024
    //   - CPU parallel: threadsPerBlock can vary
    //
    // Query device properties to get max threads per block
    auto const devProps = alpaka::getAccDevProps<Acc1D>(device);
    Idx const maxThreadsPerBlock = static_cast<Idx>(devProps.m_blockThreadExtentMax[0]);

    // Use 256 for GPU, 1 for CPU serial (determined by device properties)
    Idx const threadsPerBlock = std::min(Idx{256}, maxThreadsPerBlock);
    Idx const blocksPerGrid = (numElements + (threadsPerBlock * elementsPerThread) - 1)
                            / (threadsPerBlock * elementsPerThread);

    // Create work division: WorkDivMembers<Dim>(gridSize, blockSize, elementsPerThread)
    WorkDiv1D const workDiv{
        alpaka::Vec<Dim1D, Idx>(blocksPerGrid),        // Grid: number of blocks
        alpaka::Vec<Dim1D, Idx>(threadsPerBlock),      // Block: threads per block
        alpaka::Vec<Dim1D, Idx>(elementsPerThread)};   // Thread: elements per thread

    std::cout << "Work division: " << blocksPerGrid << " blocks × "
              << threadsPerBlock << " threads × " << elementsPerThread << " elements" << std::endl;

    // SOLUTION 7: Launch kernel
    auto const taskKernel = alpaka::createTaskKernel<Acc1D>(
        workDiv, kernel, std::data(bufDevA), std::data(bufDevB), std::data(bufDevC), numElements);

    alpaka::enqueue(queue, taskKernel);
    alpaka::wait(queue);

    // SOLUTION 8: Copy results back
    alpaka::memcpy(queue, bufHostC, bufDevC);
    alpaka::wait(queue);

    // Verify results
    int       errors         = 0;
    const int maxPrintErrors = 20;
    for (Idx i = 0; i < numElements; ++i)
    {
        Data const expected = bufHostA[i] + bufHostB[i];
        if (std::abs(bufHostC[i] - expected) > 1e-5f)
        {
            if (errors < maxPrintErrors)
            {
                std::cerr << "Error at C[" << i << "]: " << bufHostC[i] << " != " << expected << std::endl;
            }
            ++errors;
        }
    }

    if (errors == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << errors << " errors (printed max " << maxPrintErrors << ")" << std::endl;
        return EXIT_FAILURE;
    }
}
