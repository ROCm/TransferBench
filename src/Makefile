# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda

HIPCC=$(ROCM_PATH)/bin/hipcc
NVCC=$(CUDA_PATH)/bin/nvcc

# Compile TransferBenchCuda if nvcc detected
ifeq ("$(shell test -e $(NVCC) && echo found)", "found")
	EXE=../TransferBenchCuda
else
	EXE=../TransferBench
endif

CXXFLAGS = -O3 -Iinclude -I$(ROCM_PATH)/include -lnuma -L$(ROCM_PATH)/lib -lhsa-runtime64

# Compile RDMA executor if IBVerbs is found in the Dynamic Linker cache
ifneq ("$(shell ldconfig -p | grep -c ibverbs)", "0")
	CXXFLAGS += -libverbs
else
	CXXFLAGS += -DDISABLE_RDMA_EXECUTOR	 
endif

NVFLAGS = -O3 -Iinclude -x cu -lnuma -arch=native
LDFLAGS    += -lpthread
all: $(EXE)

../TransferBench: TransferBench.cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

../TransferBenchCuda: TransferBench.cpp $(shell find -regex ".*\.\hpp")
	$(NVCC) $(NVFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f *.o ../TransferBench ../TransferBenchCuda
