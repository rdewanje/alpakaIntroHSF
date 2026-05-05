/* Vector Addition - CUDA Reference Implementation
 * This is a simple vector addition kernel that computes C = A + B
 */

#include <cstdlib>
#include <iostream>
#include <random>

// kernel
__global__ void vectorAddKernel(const float *A, const float *B, float *C,
                                int numElements) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  const int numElements = 1 << 23;
  const size_t size = numElements * sizeof(float);

  float *h_A = new float[numElements];
  float *h_B = new float[numElements];
  float *h_C = new float[numElements];

  std::random_device rd;
  std::default_random_engine eng(rd());
  std::uniform_real_distribution<float> dist(1.0f, 42.0f);

  for (int i = 0; i < numElements; ++i) {
    h_A[i] = dist(eng);
    h_B[i] = dist(eng);
    h_C[i] = 0.0f;
  }

  // allocate device memory
  // let's use streams!
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  float *d_A, *d_B, *d_C;
  cudaMallocAsync(&d_A, size, stream);
  cudaMallocAsync(&d_B, size, stream);
  cudaMallocAsync(&d_C, size, stream);

  // copy H2D
  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_C, h_C, size, cudaMemcpyHostToDevice, stream);

  // work division
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

  std::cout << "Launching kernel with " << blocksPerGrid << " blocks and "
            << threadsPerBlock << " threads per block" << std::endl;

  // kernel launch
  vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0u, stream>>>(d_A, d_B, d_C,
                                                                  numElements);

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
    return EXIT_FAILURE;
  }

  // copy D2H
  cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Verify results
  int errors = 0;
  const int maxPrintErrors = 20;
  for (int i = 0; i < numElements; ++i) {
    float expected = h_A[i] + h_B[i];
    if (std::abs(h_C[i] - expected) > 1e-5) {
      if (errors < maxPrintErrors) {
        std::cerr << "Error at index " << i << ": " << h_C[i]
                  << " != " << expected << std::endl;
      }
      ++errors;
    }
  }

  if (errors == 0) {
    std::cout << "Execution results correct!" << std::endl;
  } else {
    std::cout << "Found " << errors << " errors (printed max " << maxPrintErrors
              << ")" << std::endl;
  }

  // Free device memory
  cudaFreeAsync(d_A, stream);
  cudaFreeAsync(d_B, stream);
  cudaFreeAsync(d_C, stream);
  cudaStreamDestroy(stream);

  // Free host memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
