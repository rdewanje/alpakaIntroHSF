# Makefile for CUDA Vector Addition

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70 -std=c++14

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
