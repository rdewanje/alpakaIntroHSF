.PHONY: all clean cpu cuda

# Source files
SOURCES := $(wildcard *.cpp)

# Targets for each backend
TARGETS_CPU  := $(SOURCES:%.cpp=%_cpu)
TARGETS_CUDA := $(SOURCES:%.cpp=%_cuda)

# Default target
all: cpu

cpu: $(TARGETS_CPU)

cuda: $(TARGETS_CUDA)

clean:
	rm -f $(TARGETS_CPU) $(TARGETS_CUDA)

# Paths
ALPAKA_BASE ?= ../../../alpaka
BOOST_BASE  ?= /cvmfs/cms.cern.ch/el8_amd64_gcc13/external/boost/1.80.0-f5051ff020fe9f6f1a74162002ac1ea0
CUDA_BASE   ?= /usr/local/cuda

# Compilers
CXX   := g++
NVCC  := $(CUDA_BASE)/bin/nvcc

# CUDA architecture (adjust for your GPU)
# sm_70 (V100), sm_75 (T4), sm_80 (A100), sm_86 (RTX 30xx), sm_89 (RTX 40xx)
CUDA_ARCH ?= sm_75

# Common flags
CXXFLAGS := -std=c++20 -O2 -g -isystem $(ALPAKA_BASE)/include

# CPU-specific flags
CXX_CPU_FLAGS := -march=native -pthread

# CUDA-specific flags
CXX_CUDA_FLAGS := -arch=$(CUDA_ARCH) \
                  --extended-lambda \
                  --expt-relaxed-constexpr \
                  --expt-extended-lambda \
                  -isystem $(BOOST_BASE)/include

# Pattern rule for CPU serial backend
%_cpu: %.cpp
	$(CXX) $(CXXFLAGS) $(CXX_CPU_FLAGS) \
	    -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED \
	    $< -o $@

# Pattern rule for CUDA backend
%_cuda: %.cpp
	$(NVCC) -x cu -ccbin $(CXX) $(CXXFLAGS) $(CXX_CUDA_FLAGS) \
	    -DALPAKA_ACC_GPU_CUDA_ENABLED \
	    $< -o $@
