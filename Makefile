#
# Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
#

# Configuration options
ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda

HIPCC ?= $(ROCM_PATH)/bin/amdclang++
NVCC ?= $(CUDA_PATH)/bin/nvcc

# This can be a space separated string of multiple GPU targets
# Default is the native GPU target
GPU_TARGETS ?= native

DEBUG ?= 0

# Compile TransferBenchCuda if nvcc detected
ifeq ("$(shell test -e $(NVCC) && echo found)", "found")
  EXE=TransferBenchCuda
  CXX=$(NVCC)
else
  EXE=TransferBench
  ifeq ("$(shell test -e $(HIPCC) && echo found)", "found")
    CXX=$(HIPCC)
  else ifeq ("$(shell test -e $(ROCM_PATH)/bin/hipcc && echo found)", "found")
    CXX=$(ROCM_PATH)/bin/hipcc
    $(warning "Could not find $(HIPCC). Using fallback to $(CXX)")
  else
    $(error "Could not find $(HIPCC) or $(ROCM_PATH)/bin/hipcc. Check if the path is correct if you want to build $(EXE)")
  endif
  GPU_TARGETS_FLAGS = $(foreach target,$(GPU_TARGETS),"--offload-arch=$(target)")
endif

CXXFLAGS = -I$(ROCM_PATH)/include -I$(ROCM_PATH)/include/hip -I$(ROCM_PATH)/include/hsa
HIPLDFLAGS= -lnuma -L$(ROCM_PATH)/lib -lhsa-runtime64 -lamdhip64
HIPFLAGS = -x hip -D__HIP_PLATFORM_AMD__ -D__HIPCC__ $(GPU_TARGETS_FLAGS)
NVFLAGS  = -x cu -lnuma -arch=native

ifeq ($(DEBUG), 0)
  COMMON_FLAGS += -O3
else
  COMMON_FLAGS += -O0 -g -ggdb3
endif
COMMON_FLAGS += -I./src/header -I./src/client -I./src/client/Presets

LDFLAGS += -lpthread

# Compile RDMA executor if
# 1) DISABLE_NIC_EXEC is not set to 1
# 2) IBVerbs is found in the Dynamic Linker cache
# 3) infiniband/verbs.h is found in the default include path
NIC_ENABLED = 0
ifneq ($(DISABLE_NIC_EXEC),1)
  ifeq ("$(shell ldconfig -p | grep -c ibverbs)", "0")
    $(info lib IBVerbs not found)
  else ifeq ("$(shell echo '#include <infiniband/verbs.h>' | $(CXX) -E - 2>/dev/null | grep -c 'infiniband/verbs.h')", "0")
    $(info infiniband/verbs.h not found)
  else
    CXXFLAGS += -DNIC_EXEC_ENABLED
    LDFLAGS += -libverbs
    NIC_ENABLED = 1
  endif
  ifeq ($(NIC_ENABLED), 0)
    $(info To use the TransferBench RDMA executor, check if your system has NICs, the NIC drivers are installed, and libibverbs-dev is installed)
  endif
endif


.PHONY : all clean

all: $(EXE)

TransferBench.o: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp") NicStatus
	$(HIPCC) $(CXXFLAGS) $(HIPFLAGS) $(COMMON_FLAGS) -c $< -o $@

TransferBench: TransferBench.o
	$(HIPCC) -o $@ $^ $(HIPLDFLAGS) $(LDFLAGS)

TransferBenchCuda: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp") NicStatus
	$(NVCC) $(NVFLAGS) $(COMMON_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f *.o ./TransferBench.o ./TransferBench ./TransferBenchCuda

NicStatus:
ifeq ($(NIC_ENABLED), 1)
	$(info Building with NIC executor support. Can set DISABLE_NIC_EXEC=1 to disable)
else
	$(info Building without NIC executor support)
endif
