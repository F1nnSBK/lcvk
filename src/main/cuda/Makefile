NVCC ?= nvcc

CUDA_FLAGS = -O3 -lineinfo
ARCH_FLAGS = -arch=sm_80 -arch=sm_86 -arch=sm_89 -arch=sm_90
INCLUDES = -I. -I../.. -I/usr/local/cuda/include

OUT_DIR = ../../../target/cuda

CUDA_SOURCES = pithos_kernels.cu pithos_cuda_memory.cu
CUDA_OBJECTS = $(patsubst %.cu,$(OUT_DIR)/%.o,$(CUDA_SOURCES))

all: $(OUT_DIR)/libpithos_cuda.a

$(OUT_DIR):
	mkdir -p $@

$(OUT_DIR)/%.o: %.cu | $(OUT_DIR)
	$(NVCC) $(CUDA_FLAGS) $(ARCH_FLAGS) $(INCLUDES) -c $< -o $@

$(OUT_DIR)/libpithos_cuda.a: $(CUDA_OBJECTS)
	ar rcs $@ $(CUDA_OBJECTS)

clean:
	rm -rf $(OUT_DIR)/*.o $(OUT_DIR)/libpithos_cuda.*

.PHONY: all clean
