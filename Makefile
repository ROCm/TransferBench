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
# DISABLE_DMA_BUF: Disable DMA-BUF support for GPU Direct RDMA (default: 1)
# DISABLE_AMD_SMI: Disable AMD-SMI pod membership checking support (default: 0)
# DISABLE_NVML: Disable NVML pod membership detection for CUDA builds (default: 0)

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
    else
      ifeq ("$(shell test -e $(ROCM_PATH)/llvm/bin/amdclang++ && echo found)", "found")
        CXX=$(ROCM_PATH)/llvm/bin/amdclang++
      else ifeq ("$(shell test -e $(ROCM_PATH)/llvm/bin/clang++ && echo found)", "found")
        CXX=$(ROCM_PATH)/llvm/bin/clang++
      else ifeq ("$(shell test -e $(ROCM_PATH)/bin/hipcc && echo found)", "found")
        CXX=$(ROCM_PATH)/bin/hipcc
      else
        $(error "Could not find a HIP compiler. Tried: $(HIPCC), $(ROCM_PATH)/llvm/bin/amdclang++, $(ROCM_PATH)/llvm/bin/clang++, $(ROCM_PATH)/bin/hipcc. Check if ROCM_PATH is correct")
      endif
      $(info "Could not find $(HIPCC). Using fallback to $(CXX)")
    endif
    GPU_TARGETS_FLAGS = $(foreach target,$(GPU_TARGETS),"--offload-arch=$(target)")
    $(info Compiling for $(GPU_TARGETS) architecture(s). Can modify this by setting GPU_TARGETS)
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

      # Disable DMA-BUF support by default (set DISABLE_DMA_BUF=0 to enable)
      DISABLE_DMA_BUF ?= 1
      ifeq ($(DISABLE_DMA_BUF), 0)
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
        $(info Building with DMA-BUF support disabled (DISABLE_DMA_BUF=1))
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
      LDFLAGS += -L$(MPI_PATH)/lib -L$(MPI_PATH)/lib64 -lmpi
    endif

    ifeq ($(MPI_ENABLED), 0)
      $(info - Building without MPI communicator support)
      $(info - To use TransferBench with MPI support, install MPI libraries and specify appropriate MPI_PATH)
    else
      $(info - Building with MPI communicator support.  Can set DISABLE_MPI_COMM=1 to disable)
   endif
  endif

  NVML_ENABLED = 0
  # Enable NVML support for pod membership detection on NVIDIA platforms
  # Compile with NVML support if
  # 1) DISABLE_NVML is not set to 1
  # 2) Building TransferBenchCuda
  # 3) nvml.h is found under CUDA_PATH
  DISABLE_NVML ?= 0
  ifneq ($(DISABLE_NVML), 1)
    ifeq ($(MAKECMDGOALS),TransferBenchCuda)
      $(info Attempting to build with NVML support)
      ifneq ($(wildcard $(CUDA_PATH)/include/nvml.h),)
        COMMON_FLAGS += -DNVML_ENABLED
        LDFLAGS += -lnvidia-ml
        NVML_ENABLED = 1
        $(info - Building with NVML support for pod membership detection)
      else
        $(info - nvml.h not found at $(CUDA_PATH)/include. Building without NVML support)
        $(info - Pod membership may be forced by setting TB_FORCE_SINGLE_POD=1)
      endif
    endif
  endif

  POD_ENABLED = 0
  AMD_SMI_ENABLED = 0
  # Compile with pod support if
  # 1) DISABLE_POD_COMM is not set to 1
  # 2) For HIP: hipMemFabricHandle_t is present in the HIP headers
  #    For CUDA: CUDA Version >= 12.2
  DISABLE_POD_COMM ?= 0
  DISABLE_AMD_SMI ?= 0
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
        LDFLAGS += -lcuda
        POD_ENABLED = 1
      else
        $(info - Detected CUDA version $(CUDA_MAJOR).$(CUDA_MINOR) which does not have MNNVL support)
        $(info - Pod support will require CUDA version of at least $(CUDA_MIN_MAJOR).$(CUDA_MIN_MINOR))
      endif
    else
      # Check for the HIP fabric API functions used by TransferBench at runtime.
      HIP_HAS_FABRIC := $(shell \
        printf '%s\n' \
          '#include <hip/hip_runtime_api.h>' \
          'int main() {' \
          '  hipMemFabricHandle_t fabricHandle = {};' \
          '  hipMemGenericAllocationHandle_t allocationHandle = {};' \
          '  hipMemExportToShareableHandle(&fabricHandle, allocationHandle, hipMemHandleTypeFabric, 0);' \
          '  hipMemImportFromShareableHandle(&allocationHandle, &fabricHandle, hipMemHandleTypeFabric);' \
          '  return 0;' \
          '}' | \
        $(CXX) -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__ -x c++ - -c -o /dev/null 2>/dev/null && echo yes || echo no)

      ifeq ($(HIP_HAS_FABRIC),yes)
        $(info - HIP fabric API found; enabling pod communication support)
        COMMON_FLAGS += -DPOD_COMM_ENABLED
        POD_ENABLED = 1
        ifeq ($(DISABLE_AMD_SMI), 1)
          $(info - AMD-SMI disabled via DISABLE_AMD_SMI=1; set TB_FORCE_SINGLE_POD=1 at runtime to override pod membership)
        else
          # Prefer AMD-SMI for pod membership queries; fall back to TB_FORCE_SINGLE_POD=1 at runtime.
          AMD_SMI_HEADER := $(ROCM_PATH)/include/amd_smi/amdsmi.h
          AMD_SMI_LIB    := $(firstword $(wildcard $(ROCM_PATH)/lib/libamd_smi.so $(ROCM_PATH)/lib64/libamd_smi.so))
          ifneq ($(wildcard $(AMD_SMI_HEADER)),)
            ifneq ($(AMD_SMI_LIB),)
              # Check for the AMD-SMI functions used by TransferBench at runtime.
              AMDSMI_HAS_FABRIC := $(shell \
                printf '%s\n' \
                  '#include <amd_smi/amdsmi.h>' \
                  'int main() {' \
                  '  amdsmi_bdf_t bdf = {};' \
                  '  amdsmi_processor_handle h;' \
                  '  amdsmi_get_processor_handle_from_bdf(bdf, &h);' \
                  '  amdsmi_fabric_info_t fi;' \
                  '  amdsmi_get_gpu_fabric_info(h, &fi);' \
                  '  (void)fi.fabric_info.fabric_version.v1.ppod_id;' \
                  '  (void)fi.fabric_info.fabric_version.v1.vpod_id;' \
                  '  return 0;' \
                  '}' | \
                $(CXX) -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__ -x c++ - -c -o /dev/null 2>/dev/null && echo yes || echo no)

              ifeq ($(AMDSMI_HAS_FABRIC),yes)
                $(info - AMD-SMI fabric API found; using AMD-SMI for pod membership queries)
                COMMON_FLAGS += -DAMD_SMI_ENABLED
                LDFLAGS += -L$(dir $(AMD_SMI_LIB)) -lamd_smi
                AMD_SMI_ENABLED = 1
              else
                $(info - AMD-SMI fabric API not found; set TB_FORCE_SINGLE_POD=1 at runtime to override pod membership)
              endif
            else
              $(info - libamd_smi not found under $(ROCM_PATH)/lib; set TB_FORCE_SINGLE_POD=1 at runtime to override pod membership)
            endif
          else
            $(info - amd_smi/amdsmi.h not found under $(ROCM_PATH)/include; set TB_FORCE_SINGLE_POD=1 at runtime to override pod membership)
          endif
        endif
      else
        $(info - HIP fabric API not found; disabling pod communication support)
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
