# Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda

HIPCC=$(ROCM_PATH)/bin/hipcc
NVCC=$(CUDA_PATH)/bin/nvcc

EXE=TransferBench
CXXFLAGS = -O3 -Iinclude -I$(ROCM_PATH)/include -lnuma -L$(ROCM_PATH)/lib -lhsa-runtime64
NVFLAGS = -O3 -g -Iinclude -x cu -lnuma -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_75,code=sm_75
LDFLAGS    += -lpthread
all: $(EXE)

$(EXE): $(EXE).cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o ../$@ $(LDFLAGS)

TransferBenchCuda: $(EXE).cpp $(shell find -regex ".*\.\hpp")
	$(NVCC) $(NVFLAGS) $< -o ../$@ $(LDFLAGS)

clean:
	rm -f *.o ../$(EXE) ../TransferBenchCuda
