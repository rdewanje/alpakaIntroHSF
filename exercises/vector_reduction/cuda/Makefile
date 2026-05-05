# Makefile for CUDA Vector Reduction

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++20

TARGET = vector_reduce
SOURCE = vector_reduce.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
