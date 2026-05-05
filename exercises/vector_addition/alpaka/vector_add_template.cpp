/* Vector Addition - Alpaka Template for Students
 *
 * EXERCISE: Implement vector addition using Alpaka
 *
 * Tasks:
 * 1. Complete the VectorAddKernel operator() implementation
 * 2. Set up devices and queue
 * 3. Allocate buffers for host and device
 * 4. Copy data between host and device
 * 5. Configure and launch the kernel
 * 6. Verify the results
 *
 * Compare this with the CUDA version in ../cuda/vector_add.cu
 * The backend is selected at compile time via config.h
 */

#include "../../config.h"

#include <iostream>
#include <random>

// TODO 1: Complete the vector addition kernel
class VectorAddKernel
{
public:
    template<typename TAcc, typename TElem>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        TElem const* const A,
        TElem const* const B,
        TElem* const C,
        Idx const numElements) const -> void
    {
        // TODO: Implement vector addition using alpaka::uniformElements
        // Hint: for(auto i : alpaka::uniformElements(acc, numElements)) { ... }
        //
        // What uniformElements does under the hood:
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
        // CUDA equivalent (grid-stride loop):
        //   int tid = blockIdx.x * blockDim.x + threadIdx.x;
        //   int stride = gridDim.x * blockDim.x;
        //   for(int i = tid; i < n; i += stride) {
        //       C[i] = A[i] + B[i];
        //   }

        // YOUR CODE HERE
    }
};

auto main() -> int
{
    // Print which backend we're using (defined in config.h)
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc1D>() << std::endl;

    // TODO 2: Get devices and create queue
    // Hint: Types are defined in config.h: HostPlatform, Platform, Queue
    // Use alpaka::getDevByIdx(platform, 0) to get device 0
    // YOUR CODE HERE
    auto const platformHost = /* TODO: Get host platform */;
    auto const devHost = /* TODO: Get host device */;

    auto const platform = /* TODO: Get accelerator platform */;
    auto const device = /* TODO: Get accelerator device */;

    Queue queue(/* TODO: Pass device */);

    // Problem size
    Idx const numElements = 1 << 20;  // 1M elements
    Idx const elementsPerThread = 8u;
    alpaka::Vec<Dim1D, Idx> const extent(numElements);

    using Data = float;

    // TODO 3: Allocate host buffers
    // Hint: Use BufHost<Data> (defined in config.h)
    // Use alpaka::allocBuf<Data, Idx>(devHost, extent)
    // YOUR CODE HERE
    BufHost<Data> bufHostA = /* TODO */;
    BufHost<Data> bufHostB = /* TODO */;
    BufHost<Data> bufHostC = /* TODO */;

    // Initialize input data
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_real_distribution<Data> dist(1.0f, 42.0f);

    for(Idx i = 0; i < numElements; ++i)
    {
        bufHostA[i] = dist(eng);
        bufHostB[i] = dist(eng);
        bufHostC[i] = 0.0f;
    }

    // TODO 4: Allocate device buffers
    // Hint: Use BufDevice<Data> (defined in config.h)
    // Use alpaka::allocBuf<Data, Idx>(device, extent)
    // YOUR CODE HERE
    BufDevice<Data> bufDevA = /* TODO */;
    BufDevice<Data> bufDevB = /* TODO */;
    BufDevice<Data> bufDevC = /* TODO */;

    // TODO 5: Copy data from host to device
    // Hint: alpaka::memcpy(queue, destination, source)
    // YOUR CODE HERE
    /* TODO: Copy bufHostA to bufDevA */
    /* TODO: Copy bufHostB to bufDevB */
    /* TODO: Copy bufHostC to bufDevC */

    // TODO 6: Configure kernel launch - manual work division
    // Work division has 3 levels:
    //   - Grid level: how many blocks
    //   - Block level: how many threads per block
    //   - Thread level: how many elements per thread
    //
    // Different backends have different constraints:
    //   - CPU Serial: must have threadsPerBlock = 1
    //   - GPU: typically threadsPerBlock = 64-1024
    // YOUR CODE HERE
    VectorAddKernel kernel;

    // Query device properties to get constraints
    auto const devProps = /* TODO: alpaka::getAccDevProps<Acc1D>(device) */;
    Idx const maxThreadsPerBlock = /* TODO: static_cast<Idx>(devProps.m_blockThreadExtentMax[0]) */;

    // Use 256 for GPU, but respect device limits (CPU serial = 1)
    Idx const threadsPerBlock = /* TODO: std::min(Idx{256}, maxThreadsPerBlock) */;
    Idx const blocksPerGrid = /* TODO: Calculate (numElements + (threadsPerBlock * elementsPerThread) - 1)
                                                / (threadsPerBlock * elementsPerThread) */;

    // Create work division using brace initialization to avoid "most vexing parse"
    WorkDiv1D const workDiv = /* TODO: WorkDiv1D{
        alpaka::Vec<Dim1D, Idx>(blocksPerGrid),
        alpaka::Vec<Dim1D, Idx>(threadsPerBlock),
        alpaka::Vec<Dim1D, Idx>(elementsPerThread)} */;

    // TODO 7: Create and launch kernel
    // Hint: alpaka::createTaskKernel<Acc1D>(...) and alpaka::enqueue(...)
    // YOUR CODE HERE
    auto const taskKernel = /* TODO: Create task */;

    /* TODO: Enqueue task */
    alpaka::wait(queue);

    // TODO 8: Copy results back from device to host
    // Hint: alpaka::memcpy(queue, destination, source) - reverse direction!
    // YOUR CODE HERE
    /* TODO: Copy bufDevC to bufHostC */
    alpaka::wait(queue);

    // Verify results
    int errors = 0;
    const int maxPrintErrors = 20;
    for(Idx i = 0; i < numElements; ++i)
    {
        Data const expected = bufHostA[i] + bufHostB[i];
        if(std::abs(bufHostC[i] - expected) > 1e-5f)
        {
            if(errors < maxPrintErrors)
            {
                std::cerr << "Error at C[" << i << "]: "
                         << bufHostC[i] << " != " << expected << std::endl;
            }
            ++errors;
        }
    }

    if(errors == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << errors << " errors (printed max "
                  << maxPrintErrors << ")" << std::endl;
        return EXIT_FAILURE;
    }
}
