#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#

# Configuration options
ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda
MPI_PATH  ?= /usr/local/openmpi

# Optional features (set to 0 to disable, 1 to enable)
# DISABLE_NIC_EXEC: Disable RDMA/NIC executor support (default: 0)
# DISABLE_MPI_COMM: Disable MPI communicator support (default: 0)
# DISABLE_DMABUF: Disable DMA-BUF support for GPU Direct RDMA (default: 1)

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
    $(info Building TransferBenchCuda)
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
      $(info "Could not find $(HIPCC). Using fallback to $(CXX)")
    else
      $(error "Could not find $(HIPCC) or $(ROCM_PATH)/bin/hipcc. Check if the path is correct if you want to build $(EXE)")
    endif
    GPU_TARGETS_FLAGS = $(foreach target,$(GPU_TARGETS),"--offload-arch=$(target)")

    CXXFLAGS = -I. -I$(ROCM_PATH)/include -I$(ROCM_PATH)/include/hip -I$(ROCM_PATH)/include/hsa
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
    $(info Attempting to build with NIC executor support)
    ifeq ("$(shell ldconfig -p | grep -c ibverbs)", "0")
      $(info - ibverbs library not found)
    else ifeq ("$(shell echo '#include <infiniband/verbs.h>' | $(CXX) -E - 2>/dev/null | grep -c 'infiniband/verbs.h')", "0")
      $(info - infiniband/verbs.h not found)
    else
      COMMON_FLAGS += -DNIC_EXEC_ENABLED
      LDFLAGS += -libverbs
      NIC_ENABLED = 1

      # Disable DMA-BUF support by default (set DISABLE_DMABUF=0 to enable)
      DISABLE_DMABUF ?= 1
      ifeq ($(DISABLE_DMABUF), 0)
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
        $(info Building with DMA-BUF support disabled (DISABLE_DMABUF=1))
      endif
    endif
    ifeq ($(NIC_ENABLED), 0)
      $(info - Building without NIC executor support)
      $(info - To use the TransferBench RDMA executor, check if your system has NICs, the NIC drivers are installed, and libibverbs-dev is installed)
    else
      $(info - Building with NIC executor support. Can set DISABLE_NIC_EXEC=1 to disable)
    endif
  endif

  MPI_ENABLED = 0
  # Compile with MPI communicator support if
  # 1) DISABLE_MPI_COMM is not set to 1
  # 2) mpi.h is found in the MPI_PATH
  DISABLE_MPI_COMM ?= 0
  ifneq ($(DISABLE_MPI_COMM), 1)
    $(info Attempting to build with MPI communicator support)
    ifeq ($(wildcard $(MPI_PATH)/include/mpi.h),)
      $(info - Unable to find mpi.h at $(MPI_PATH)/include.  Please specify appropriate MPI_PATH)
    else
      MPI_ENABLED = 1
      COMMON_FLAGS += -DMPI_COMM_ENABLED -I$(MPI_PATH)/include
      LDFLAGS += -L/$(MPI_PATH)/lib -lmpi
      ifeq ($(DEBUG), 1)
        LDFLAGS += -lmpi_cxx
      endif
    endif

    ifeq ($(MPI_ENABLED), 0)
      $(info - Building without MPI communicator support)
      $(info - To use TransferBench with MPI support, install MPI libraries and specify appropriate MPI_PATH)
    else
      $(info - Building with MPI communicator support.  Can set DISABLE_MPI_COMM=1 to disable)
   endif
  endif

  AMD_SMI_ENABLED = 0
  # Enable AMD-SMI support for pod membership detection
  # Compile with AMD-SMI support if
  # 1) DISABLE_AMD_SMI is not set to 1
  # 2) AMD-SMI version >= 26.4.1
  DISABLE_AMD_SMI ?= 0
  ifneq ($(DISABLE_AMD_SMI), 1)
    ifneq ($(MAKECMDGOALS),TransferBenchCuda)
      $(info Attempting to build with amd-smi support)
      # Check for appropriate AMD SMI version (for querying pod membership)
      AMD_SMI_MIN_MAJOR := 26
      AMD_SMI_MIN_MINOR := 4

      AMD_SMI ?= amd-smi
      AMD_SMI_EXISTS := $(shell command -v $(AMD_SMI) >/dev/null 2>&1 && echo yes || echo no)
      ifeq ($(AMD_SMI_EXISTS),no)
        $(info - $(AMD_SMI) not found.  Disabling pod communication support)
      else
        AMD_SMI_VERSION_STR := $(shell $(AMD_SMI) version | sed -n 's/.*Library version: \([0-9]\+\)\.\([0-9]\+\).*/\1 \2/p')
        AMD_SMI_MAJOR := $(word 1,$(AMD_SMI_VERSION_STR))
        AMD_SMI_MINOR := $(word 2,$(AMD_SMI_VERSION_STR))

        AMD_SMI_VERSION_OK := $(shell \
          if [ $(AMD_SMI_MAJOR) -gt $(AMD_SMI_MIN_MAJOR) ] || \
             [ $(AMD_SMI_MAJOR) -eq $(AMD_SMI_MIN_MAJOR) -a $(AMD_SMI_MINOR) -ge $(AMD_SMI_MIN_MINOR) ]; then \
            echo yes; \
          else \
            echo no; \
          fi)

        ifeq ($(AMD_SMI_VERSION_OK),yes)
          $(info - Detected amd-smi version $(AMD_SMI_MAJOR).$(AMD_SMI_MINOR) which has pod support)
          COMMON_FLAGS += -DAMD_SMI_ENABLED
          AMD_SMI_ENABLED = 1
        else
          $(info - Detected amd-smi version $(AMD_SMI_MAJOR).$(AMD_SMI_MINOR) which does not have pod support)
          $(info - Pod membership querying requires amd-smi version of at least $(AMD_SMI_MIN_MAJOR).$(AMD_SMI_MIN_MINOR))
          $(info - Pod membership may be forced in TransferBench by setting FORCE_SINGLE_POD=1)
        endif
      endif
    endif
  endif

  POD_ENABLED = 0
  # Compile with pod support if
  # 1) DISABLE_POD_COMM is not set to 1
  # 2) For HIP: HIP Runtime version >= 8
  #    For CUDA: CUDA Version >= 12.8.1
  DISABLE_POD_COMM ?= 0
  ifneq ($(DISABLE_POD_COMM), 1)
    $(info Attempting to build with pod communication support)
    ifeq ($(MAKECMDGOALS),TransferBenchCuda)
      # Check for appropriate CUDA support for MNNVL
      CUDA_MIN_MAJOR := 12
      CUDA_MIN_MINOR := 2

      CUDA_VERSION_STR := $(shell $(NVCC) --version | grep release | sed -E 's/.*release ([0-9]+)\.([0-9]+).*/\1 \2/')
      CUDA_MAJOR := $(word 1,$(CUDA_VERSION_STR))
      CUDA_MINOR := $(word 2,$(CUDA_VERSION_STR))

      CUDA_VERSION_OK := $(shell \
        if [ $(CUDA_MAJOR) -gt $(CUDA_MIN_MAJOR) ] || \
           [ $(CUDA_MAJOR) -eq $(CUDA_MIN_MAJOR) -a $(CUDA_MINOR) -ge $(CUDA_MIN_MINOR) ]; then \
          echo yes; \
        else \
          echo no; \
        fi)

      ifeq ($(CUDA_VERSION_OK),yes)
        $(info - Detected CUDA version $(CUDA_MAJOR).$(CUDA_MINOR) which has MNNVL support)
        COMMON_FLAGS += -DPOD_COMM_ENABLED
        POD_ENABLED = 1
      else
        $(info - Detected CUDA version $(CUDA_MAJOR).$(CUDA_MINOR) which does not have MNNVL support)
        $(info - Pod support will require CUDA version of at least $(CUDA_MIN_MAJOR).$(CUDA_MIN_MINOR))
      endif
    else
      # Check for appropriate HIP version (for exchanging pod memory handles)
      HIP_MIN_MAJOR := 8
      HIP_MIN_MINOR := 0

      # Check for hipconfig
      HIPCONFIG ?= hipconfig
      HIP_EXISTS := $(shell command -v $(HIPCONFIG) >/dev/null 2>&1 && echo yes || echo no)
      ifeq ($(HIP_EXISTS),yes)
        HIP_VERSION_STR := $(shell $(HIPCONFIG) --version | sed -E 's/([0-9]+)\.([0-9]+).*/\1 \2/')
        HIP_MAJOR := $(word 1,$(HIP_VERSION_STR))
        HIP_MINOR := $(word 2,$(HIP_VERSION_STR))

        HIP_VERSION_OK := $(shell \
          if [ $(HIP_MAJOR) -gt $(HIP_MIN_MAJOR) ] || \
             [ $(HIP_MAJOR) -eq $(HIP_MIN_MAJOR) -a $(HIP_MINOR) -ge $(HIP_MIN_MINOR) ]; then \
             echo yes; \
          else \
            echo no; \
          fi)

        ifeq ($(HIP_VERSION_OK),yes)
          $(info - Detected HIP version $(HIP_MAJOR).$(HIP_MINOR) which has pod support)
          COMMON_FLAGS += -DPOD_COMM_ENABLED
        else
          $(info - Detected HIP version $(HIP_MAJOR).$(HIP_MINOR) which does not have pod support)
          $(info - Pod support requires HIP version of at least $(HIP_MIN_MAJOR).$(HIP_MIN_MINOR))
        endif
      else
        $(info - Unable to determine HIP version via $(HIPCONFIG).  Try specifying path to hipconfig in HIPCONFIG)
        $(info - Disabling pod communication support)
      endif
    endif
  endif
endif

.PHONY : all clean

all: TransferBench

TransferBench: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp")
	$(CXX) $(CXXFLAGS) $(HIPFLAGS) $(COMMON_FLAGS) $< -o $@ $(HIPLDFLAGS) $(LDFLAGS)

TransferBenchCuda: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp")
	$(NVCC) $(NVFLAGS) $(COMMON_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f ./TransferBench ./TransferBenchCuda
