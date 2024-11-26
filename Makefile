#
# Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
#

# Configuration options
ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda

HIPCC=$(ROCM_PATH)/bin/hipcc
NVCC=$(CUDA_PATH)/bin/nvcc

# Compile TransferBenchCuda if nvcc detected
ifeq ("$(shell test -e $(NVCC) && echo found)", "found")
	EXE=TransferBenchCuda
else
	EXE=TransferBench
endif

CXXFLAGS = -I$(ROCM_PATH)/include -lnuma -L$(ROCM_PATH)/lib -lhsa-runtime64
NVFLAGS  = -x cu -lnuma -arch=native
COMMON_FLAGS = -O3 --std=c++20 -I./src/header -I./src/client -I./src/client/Presets
LDFLAGS += -lpthread

all: $(EXE)

TransferBench: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $(COMMON_FLAGS) $< -o $@ $(LDFLAGS)

TransferBenchCuda: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp")
	$(NVCC) $(NVFLAGS) $(COMMON_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f *.o ./TransferBench ./TransferBenchCuda
