/* Vector Reduction - CUDA Reference Implementation
 * This performs a block-wise sum reduction on a vector
 * Each block reduces its portion and writes a partial sum
 * A second kernel reduces the partial sums to get the final result
 */

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <iostream>

// CUDA kernel for block-wise reduction
template <unsigned int blockSize>
__global__ void reduceKernel(const float *input, float *output,
                             unsigned int n) {
  // Shared memory for partial sums within a block
  __shared__ float sdata[blockSize];

  // Global thread ID
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize + threadIdx.x;

  // Load data into shared memory
  // Each thread loads one element and performs initial reduction
  float sum = 0.0f;

  // Grid-stride loop to handle cases where n > grid size
  for (unsigned int idx = i; idx < n; idx += gridDim.x * blockSize) {
    sum += input[idx];
  }

  // Store to shared memory
  sdata[tid] = sum;
  __syncthreads();

  // Simple tree reduction in shared memory
  for (unsigned int stride = blockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  // Thread 0 writes result
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

int main() {
  const unsigned int numElements = 1 << 20; // 1 million elements
  const size_t size = numElements * sizeof(float);
  const unsigned int blockSize = 256;

  // Allocate host memory
  float *h_input; // new float[numElements];

  cudaMallocHost(&h_input, numElements * sizeof(float));

  // Initialize input vector
  for (unsigned int i = 0; i < numElements; ++i) {
    h_input[i] = 1.0f; // Simple: all ones, so sum should equal numElements
  }

  // Allocate device memory
  float *d_input, *d_partial;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Calculate number of blocks
  unsigned int numBlocks = (numElements + blockSize - 1) / blockSize;
  cudaMallocAsync(&d_input, size, stream);
  cudaMallocAsync(&d_partial, numBlocks * sizeof(float), stream);

  // Copy data to device
  cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream);

  std::cout << "Reducing " << numElements << " elements" << std::endl;
  std::cout << "Using " << numBlocks << " blocks with " << blockSize
            << " threads each" << std::endl;

  // First reduction: reduce input array to partial sums
  reduceKernel<blockSize>
      <<<numBlocks, blockSize, 0u, stream>>>(d_input, d_partial, numElements);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
    return EXIT_FAILURE;
  }

  // Second reduction: reduce partial sums to final result
  // Use only 1 block for the final reduction
  float *d_output;
  cudaMallocAsync(&d_output, sizeof(float), stream);

  reduceKernel<blockSize>
      <<<1, blockSize, 0u, stream>>>(d_partial, d_output, numBlocks);

  // Wait for kernel to complete

  // Copy result back to host
  float h_result;
  cudaMemcpyAsync(&h_result, d_output, sizeof(float), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  // Verify result
  float expected = static_cast<float>(numElements);

  std::cout << "Result: " << h_result << std::endl;
  std::cout << "Expected: " << expected << std::endl;

  if (std::abs(h_result - expected) < 1e-2) {
    std::cout << "Reduction result correct!" << std::endl;
  } else {
    std::cout << "Reduction result incorrect!" << std::endl;
    std::cout << "Error: " << std::abs(h_result - expected) << std::endl;
  }

  // Free device memory
  cudaFreeAsync(d_input, stream);
  cudaFreeAsync(d_partial, stream);
  cudaFreeAsync(d_output, stream);
  cudaFreeHost(h_input);
  cudaStreamDestroy(stream);

  return (std::abs(h_result - expected) < 1e-2) ? EXIT_SUCCESS : EXIT_FAILURE;
}
