/* Vector Reduction - Alpaka Template for Students
 *
 * EXERCISE: Implement block-wise sum reduction using Alpaka
 *
 * Tasks:
 * 1. Complete the ReduceKernel operator() implementation
 *    - Use shared memory for block-level reduction
 *    - Implement the reduction tree within a block
 *    - Write the block's partial sum to output
 * 2. Set up buffer allocation and memory transfers
 * 3. Launch the kernel twice:
 *    - First pass: reduce input to partial sums (one per block)
 *    - Second pass: reduce partial sums to final result
 * 4. Verify the result
 *
 * Compare with the CUDA version in ../cuda/vector_reduce.cu
 * The backend is selected at compile time via config.h
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

// TODO 1: Complete the reduction kernel
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
        // TODO: Declare shared memory for block-level reduction
        // Hint: alpaka::declareSharedVar<SharedArray<T, TBlockSize>, __COUNTER__>(acc)
        auto& sdata = /* TODO */;

        // TODO: Get thread and block indices
        // Hint: alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc)[0]
        //       alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0]
        //       alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0]
        auto const blockIdx = /* TODO */;
        auto const threadIdx = /* TODO */;
        auto const blockDim = /* TODO */;
        auto const gridDim = /* TODO */;

        // TODO: Grid-stride loop - each thread accumulates multiple elements
        // Hint: for(Idx i = blockIdx * blockDim + threadIdx; i < n; i += gridDim * blockDim)
        T sum = 0;
        // YOUR LOOP HERE

        // Store to shared memory
        sdata[threadIdx] = sum;

        // TODO: Synchronize all threads in the block
        // Hint: alpaka::syncBlockThreads(acc)
        // YOUR CODE HERE

        // TODO: Simple tree-based reduction in shared memory
        // Loop with stride starting from blockDim/2, halving each iteration
        // Hint: for(uint32_t stride = blockDim / 2; stride > 0; stride >>= 1)
        // YOUR CODE HERE
        /*
        for(uint32_t stride = ...; stride > 0; stride >>= 1)
        {
            // TODO: If threadIdx < stride, add neighbor element
            if(threadIdx < stride)
            {
                sdata[threadIdx] += sdata[threadIdx + stride];
            }

            // TODO: Synchronize after each reduction step
            // YOUR CODE HERE
        }
        */

        // TODO: Thread 0 writes the block's result to global memory
        // YOUR CODE HERE
    }
};

auto main() -> int
{
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc1D>() << std::endl;

    // TODO 2: Get devices and create queue
    // YOUR CODE HERE
    auto const platformHost = /* TODO */;
    auto const devHost = /* TODO */;

    auto const platform = /* TODO */;
    auto const device = /* TODO */;

    Queue queue(/* TODO */);

    // Problem size
    Idx const numElements = 1 << 20;  // 1M elements
    constexpr uint32_t blockSize = 256;

    using Data = float;

    // Calculate number of blocks
    Idx const numBlocks = (numElements + blockSize - 1) / blockSize;

    std::cout << "Reducing " << numElements << " elements" << std::endl;
    std::cout << "Using " << numBlocks << " blocks with " << blockSize << " threads each" << std::endl;

    // TODO 3: Allocate host buffer for input
    // YOUR CODE HERE
    alpaka::Vec<Dim1D, Idx> const extentInput(numElements);
    BufHost<Data> bufHostInput = /* TODO */;

    // Initialize input (all ones for simple verification)
    for(Idx i = 0; i < numElements; ++i)
    {
        bufHostInput[i] = 1.0f;
    }

    // TODO 4: Allocate device buffers
    // - Input buffer: size numElements
    // - Partial sums buffer: size numBlocks
    // - Output buffer: size 1
    // YOUR CODE HERE
    alpaka::Vec<Dim1D, Idx> const extentPartial(numBlocks);
    alpaka::Vec<Dim1D, Idx> const extentOutput(1);

    BufDevice<Data> bufDevInput = /* TODO */;
    BufDevice<Data> bufDevPartial = /* TODO */;
    BufDevice<Data> bufDevOutput = /* TODO */;

    // TODO 5: Copy input data to device
    // YOUR CODE HERE

    // TODO 6: Create kernel instances
    // YOUR CODE HERE
    ReduceKernel<blockSize, Data> kernel1, kernel2;

    // TODO 7: Define work divisions manually
    // Work division format: WorkDiv1D(gridSize, blockSize, elementsPerThread)
    //
    // First reduction: multiple blocks process the input
    //   Grid: numBlocks (each block reduces its portion)
    //   Block: blockSize threads per block
    //   Thread: 1 element per iteration (grid-stride loop handles rest)
    //
    // Second reduction: single block reduces partial sums
    //   Grid: 1 block (reduce the numBlocks partial sums)
    //   Block: blockSize threads
    //   Thread: 1 element per iteration
    // YOUR CODE HERE
    WorkDiv1D workDiv1 = /* TODO: WorkDiv1D(
        alpaka::Vec<Dim1D, Idx>(numBlocks),
        alpaka::Vec<Dim1D, Idx>(blockSize),
        alpaka::Vec<Dim1D, Idx>(1)) */;

    WorkDiv1D workDiv2 = /* TODO: WorkDiv1D(
        alpaka::Vec<Dim1D, Idx>(1),
        alpaka::Vec<Dim1D, Idx>(blockSize),
        alpaka::Vec<Dim1D, Idx>(1)) */;

    // TODO 8: First reduction - input to partial sums
    // YOUR CODE HERE
    auto const taskKernel1 = /* TODO: createTaskKernel */;

    /* TODO: enqueue taskKernel1 */
    alpaka::wait(queue);

    // TODO 9: Second reduction - partial sums to final result
    // YOUR CODE HERE
    auto const taskKernel2 = /* TODO: createTaskKernel */;

    /* TODO: enqueue taskKernel2 */
    alpaka::wait(queue);

    // TODO 10: Copy result back to host
    // YOUR CODE HERE
    Data h_result = 0.0f;
    /* TODO: Create view and copy from device */

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
