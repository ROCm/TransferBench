#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#

# Configuration options
ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda
MPI_PATH  ?= /usr/local/openmpi

# Optional features (set to 0 to disable, 1 to enable)
# DISABLE_NIC_EXEC: Disable RDMA/NIC executor support
# DISABLE_MPI_COMM: Disable MPI communicator support
# USE_DMABUF: Enable DMA-BUF support for GPU Direct RDMA (default: 0)

HIPCC ?= $(ROCM_PATH)/bin/amdclang++
NVCC ?= $(CUDA_PATH)/bin/nvcc

# ROCm device libraries can live in different locations depending on packaging.
# hipcc/clang needs to find the amdgcn bitcode directory at link time.
ROCM_DEVICE_LIB_PATH ?=
ifneq ($(wildcard $(ROCM_PATH)/amdgcn/bitcode),)
  ROCM_DEVICE_LIB_PATH := $(ROCM_PATH)/amdgcn/bitcode
else ifneq ($(wildcard $(ROCM_PATH)/lib/llvm/amdgcn/bitcode),)
  ROCM_DEVICE_LIB_PATH := $(ROCM_PATH)/lib/llvm/amdgcn/bitcode
endif

# Option to compile with single GFX kernel to drop compilation time
SINGLE_KERNEL ?= 0

# This can be a space separated string of multiple GPU targets
# Default is the native GPU target
GPU_TARGETS ?= native

EXE=TransferBench
DEBUG ?= 0

# Only perform this check if 'make clean' is not the target
ifeq ($(filter clean,$(MAKECMDGOALS)),)
  ifeq ($(MAKECMDGOALS),TransferBenchCuda)
    # Check for nvcc
    ifneq ($(shell test -e $(NVCC) && echo found), found)
      $(error "Could not find $(NVCC).  Please set CUDA_PATH appropriately")
    else
      $(info Compiling TransferBenchCuda using $(NVCC))
    endif
    NVFLAGS = -x cu -lnuma -arch=native
  else
    # Check for HIP compiler
    ifeq ("$(shell test -e $(HIPCC) && echo found)", "found")
      CXX=$(HIPCC)
    else ifeq ("$(shell test -e $(ROCM_PATH)/bin/hipcc && echo found)", "found")
      CXX=$(ROCM_PATH)/bin/hipcc
      $(warning "Could not find $(HIPCC). Using fallback to $(CXX)")
    else
      $(error "Could not find $(HIPCC) or $(ROCM_PATH)/bin/hipcc. Check if the path is correct if you want to build $(EXE)")
    endif
    GPU_TARGETS_FLAGS = $(foreach target,$(GPU_TARGETS),"--offload-arch=$(target)")

    CXXFLAGS = -I$(ROCM_PATH)/include -I$(ROCM_PATH)/include/hip -I$(ROCM_PATH)/include/hsa
    HIPLDFLAGS= -lnuma -L$(ROCM_PATH)/lib -lhsa-runtime64 -lamdhip64
    HIPFLAGS = -Wall -x hip -D__HIP_PLATFORM_AMD__ -D__HIPCC__ $(GPU_TARGETS_FLAGS)
    ifneq ($(strip $(ROCM_DEVICE_LIB_PATH)),)
      HIPFLAGS += --rocm-device-lib-path=$(ROCM_DEVICE_LIB_PATH)
    endif
  endif

  ifeq ($(SINGLE_KERNEL), 1)
    COMMON_FLAGS += -DSINGLE_KERNEL
  endif

  ifeq ($(DEBUG), 0)
    COMMON_FLAGS += -O3
  else
    COMMON_FLAGS += -O0 -g -ggdb3
  endif
  COMMON_FLAGS += -I./src/header -I./src/client -I./src/client/Presets

  LDFLAGS += -lpthread

  NIC_ENABLED = 0
  # Compile RDMA executor if
  # 1) DISABLE_NIC_EXEC is not set to 1
  # 2) IBVerbs is found in the Dynamic Linker cache
  # 3) infiniband/verbs.h is found in the default include path
  DISABLE_NIC_EXEC ?= 0
  ifneq ($(DISABLE_NIC_EXEC),1)
    ifeq ("$(shell ldconfig -p | grep -c ibverbs)", "0")
      $(info lib IBVerbs not found)
    else ifeq ("$(shell echo '#include <infiniband/verbs.h>' | $(CXX) -E - 2>/dev/null | grep -c 'infiniband/verbs.h')", "0")
      $(info infiniband/verbs.h not found)
    else
      COMMON_FLAGS += -DNIC_EXEC_ENABLED
      LDFLAGS += -libverbs
      NIC_ENABLED = 1

      # Disable DMA-BUF support by default (set USE_DMABUF=1 to enable)
      USE_DMABUF ?= 0
      ifneq ($(USE_DMABUF), 0)
        # Check for both ibv_reg_dmabuf_mr and ROCm DMA-BUF export support
        HAVE_IBV_DMABUF := $(shell echo '#include <infiniband/verbs.h>' | $(CXX) -E - 2>/dev/null | grep -c 'ibv_reg_dmabuf_mr')
        HAVE_ROCM_DMABUF := $(shell echo '#include <hsa/hsa_ext_amd.h>' | $(CXX) -I$(ROCM_PATH)/include -E - 2>/dev/null | grep -c 'hsa_amd_portable_export_dmabuf')

        ifeq ($(HAVE_IBV_DMABUF):$(HAVE_ROCM_DMABUF), 0:0)
          $(info Building without DMA-BUF support: missing both ibv_reg_dmabuf_mr and ROCm DMA-BUF export)
        else ifeq ($(HAVE_IBV_DMABUF), 0)
          $(info Building without DMA-BUF support: missing ibv_reg_dmabuf_mr)
        else ifeq ($(HAVE_ROCM_DMABUF), 0)
          $(info Building without DMA-BUF support: missing ROCm DMA-BUF export)
        else
          COMMON_FLAGS += -DHAVE_DMABUF_SUPPORT
          $(info Building with DMA-BUF support)
        endif
      else
        $(info Building with DMA-BUF support disabled (USE_DMABUF=0))
      endif
    endif
    ifeq ($(NIC_ENABLED), 0)
      $(info Building without NIC executor support)
      $(info To use the TransferBench RDMA executor, check if your system has NICs, the NIC drivers are installed, and libibverbs-dev is installed)
    else
      $(info Building with NIC executor support. Can set DISABLE_NIC_EXEC=1 to disable)
    endif
  endif

  MPI_ENABLED = 0
  # Compile with MPI communicator support if
  # 1) DISABLE_MPI_COMM is not set to 1
  # 2) mpi.h is found in the MPI_PATH
  DISABLE_MPI_COMM ?= 0
  ifneq ($(DISABLE_MPI_COMM), 1)
    ifeq ($(wildcard $(MPI_PATH)/include/mpi.h),)
      $(info Unable to find mpi.h at $(MPI_PATH)/include.  Please specify appropriate MPI_PATH)
    else
      MPI_ENABLED = 1
      COMMON_FLAGS += -DMPI_COMM_ENABLED -I$(MPI_PATH)/include
      LDFLAGS += -L/$(MPI_PATH)/lib -lmpi
      ifeq ($(DEBUG), 1)
        LDFLAGS += -lmpi_cxx
      endif
    endif

    ifeq ($(MPI_ENABLED), 0)
      $(info Building without MPI communicator support)
      $(info To use TransferBench with MPI support, install MPI libraries and specify appropriate MPI_PATH)
    else
      $(info Building with MPI communicator support.  Can set DISABLE_MPI_COMM=1 to disable)
   endif
  endif
endif


.PHONY : all clean

all: $(EXE)

TransferBench: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp")
	$(CXX) $(CXXFLAGS) $(HIPFLAGS) $(COMMON_FLAGS) $< -o $@ $(HIPLDFLAGS) $(LDFLAGS)

TransferBenchCuda: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp")
	$(NVCC) $(NVFLAGS) $(COMMON_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f ./TransferBench ./TransferBenchCuda
