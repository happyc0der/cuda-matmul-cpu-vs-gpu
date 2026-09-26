# Linux / WSL build (needs the CUDA Toolkit's nvcc and a host C++ compiler).
# Usage: make            # build everything into build/
#        make ARCH=sm_75 # target a different GPU architecture
NVCC     ?= nvcc
CXX      ?= g++
ARCH     ?= sm_86
BUILD    := build

CUDA_BINS := $(BUILD)/hello $(BUILD)/matrix_mul $(BUILD)/matrix_mul_compare
CPU_BINS  := $(BUILD)/matrix_mul_cpu

all: $(CUDA_BINS) $(CPU_BINS)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%: %.cu | $(BUILD)
	$(NVCC) -O2 -arch=$(ARCH) -o $@ $<

$(BUILD)/matrix_mul_cpu: matrix_mul.cpp | $(BUILD)
	$(CXX) -O2 -std=c++17 -o $@ $<

clean:
	rm -rf $(BUILD)

.PHONY: all clean
