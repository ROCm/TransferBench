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

CXXFLAGS = -g -O3 --std=c++20 -I./src/header -I./src/client -I./src/client/Presets -I$(ROCM_PATH)/include -lnuma -L$(ROCM_PATH)/lib -lhsa-runtime64
NVFLAGS  = -O3 -Iinclude -x cu -lnuma -arch=native
LDFLAGS += -lpthread

TransferBench: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp")
	$(HIPCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

/TransferBenchCuda: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp")
	$(NVCC) $(NVFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f *.o ./TransferBench ./TransferBenchCuda
