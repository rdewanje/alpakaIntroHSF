# Makefile for CUDA Vector Addition

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++20

TARGET = vector_add
SOURCE = vector_add.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
