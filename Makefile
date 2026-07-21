#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#

# Configuration options
ROCM_PATH ?= /opt/rocm
CUDA_PATH ?= /usr/local/cuda
MPI_PATH  ?= /usr/local/openmpi
# pip ROCm wheels ship bin/amdclang++ but often omit bin/amdllvm (which that stub execs).
# Default to llvm/bin when bin/amdllvm is absent so HIP builds work without extra flags.
ifeq ("$(shell test -e $(ROCM_PATH)/bin/amdllvm && echo found)", "found")
HIPCC     ?= $(ROCM_PATH)/bin/amdclang++
else ifeq ("$(shell test -e $(ROCM_PATH)/llvm/bin/amdclang++ && echo found)", "found")
HIPCC     ?= $(ROCM_PATH)/llvm/bin/amdclang++
else
HIPCC     ?= $(ROCM_PATH)/bin/amdclang++
endif
NVCC      ?= $(CUDA_PATH)/bin/nvcc
DEBUG     ?= 0

# Optional features (set to 0 to disable, 1 to enable)
# DISABLE_MPI_COMM:  Disable MPI communicator support                      (default: 0)
# DISABLE_AMD_SMI:   Disable AMD-SMI pod membership checking support       (default: 0)
# DISABLE_NVML:      Disable NVML pod membership detection for CUDA builds (default: 0)
# DISABLE_POD_COMM:  Disable pod communication support                     (default: 0)
# DISABLE_CUMEM:     Disable CUDA driver API (also disables pod on CUDA)   (default: 0)
# ENABLE_ANVIL_EXEC: Enable GPU-initiated SDMA executor (AMD KFD, ROCm only) (default: 0)

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
    COMMON_FLAGS += -O3 -g
  else
    COMMON_FLAGS += -O0 -g -ggdb3
  endif
  COMMON_FLAGS += -I./src/header -I./src/client -I./src/client/Presets -I./third-party/ibverbs

  # libibverbs is loaded dynamically at runtime via dlopen/dlsym (see
  # third-party/ibverbs/IbvDynLoad.hpp), so the build never links against -libverbs
  # and does not require libibverbs-dev to be installed. We only need -ldl so
  # the dynamic loader API is resolvable.
  LDFLAGS += -lpthread -ldl

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

  # TransferBenchCuda: CUDA driver API (libcuda). Independent of POD, but POD on CUDA requires CUMEM.
  DISABLE_CUMEM ?= 0
  ifeq ($(MAKECMDGOALS),TransferBenchCuda)
    ifneq ($(DISABLE_CUMEM),1)
      $(info - Building with CUMEM_ENABLED (CUDA driver API, -lcuda))
      COMMON_FLAGS += -DCUMEM_ENABLED
      LDFLAGS += -lcuda
    else
      $(info - CUDA driver API disabled (DISABLE_CUMEM=1); POD comm unavailable on CUDA)
    endif
  endif

  POD_ENABLED = 0
  AMD_SMI_ENABLED = 0
  # Compile with pod support if
  # 1) DISABLE_POD_COMM is not set to 1
  # 2) For HIP: a small probe program that uses hipMemFabricHandle_t,
  #    hipMemExportToShareableHandle, and hipMemImportFromShareableHandle
  #    compiles and links successfully against amdhip64
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
        ifeq ($(DISABLE_CUMEM),1)
          $(info - Pod communication skipped on CUDA: requires CUMEM_ENABLED (DISABLE_CUMEM=1))
        else
          COMMON_FLAGS += -DPOD_COMM_ENABLED
          POD_ENABLED = 1
        endif
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
        $(CXX) -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__ -x c++ - \
          -L$(ROCM_PATH)/lib -L$(ROCM_PATH)/lib64 -lamdhip64 -o /dev/null 2>/dev/null && echo yes || echo no)

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
                $(CXX) -I$(ROCM_PATH)/include -x c++ - \
                  -L$(dir $(AMD_SMI_LIB)) -lamd_smi -o /dev/null 2>/dev/null && echo yes || echo no)

              ifeq ($(AMDSMI_HAS_FABRIC),yes)
                $(info - AMD-SMI fabric API found; using AMD-SMI for pod membership queries)
                COMMON_FLAGS += -DAMD_SMI_ENABLED
                LDFLAGS += -L$(dir $(AMD_SMI_LIB)) -lamd_smi
                AMD_SMI_ENABLED = 1
              else
                $(info - AMD-SMI fabric API not found; set TB_FORCE_SINGLE_POD=1 at runtime to override pod membership)
              endif
            else
              $(info - libamd_smi not found under $(ROCM_PATH)/lib or $(ROCM_PATH)/lib64; set TB_FORCE_SINGLE_POD=1 at runtime to override pod membership)
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

# Git metadata (branch + short commit hash)
# Priority: git rev-parse > GIT_VERSION file (populated by packaging scripts) > "unknown"
_TB_DIR       := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
TB_GIT_BRANCH := $(shell git -C "$(_TB_DIR)" rev-parse --abbrev-ref HEAD 2>/dev/null || sed -n '1p' "$(_TB_DIR)GIT_VERSION" 2>/dev/null || echo unknown)
TB_GIT_COMMIT := $(shell git -C "$(_TB_DIR)" rev-parse --short HEAD 2>/dev/null || sed -n '2p' "$(_TB_DIR)GIT_VERSION" 2>/dev/null || echo unknown)
COMMON_FLAGS  += -DTB_GIT_BRANCH='"$(TB_GIT_BRANCH)"' -DTB_GIT_COMMIT='"$(TB_GIT_COMMIT)"'

  ANVIL_ENABLED = 0
  # Compile with GPU-initiated SDMA executor (anvil/KFD) if
  # 1) ENABLE_ANVIL_EXEC is set to 1
  # 2) hsakmt/hsakmt.h is found (KFD user-space library header)
  # 3) libhsakmt is found (static or shared)
  # Note: disabled by default; requires AMD ROCm KFD and is AMD-only
  ENABLE_ANVIL_EXEC ?= 0
  ifeq ($(ENABLE_ANVIL_EXEC), 1)
    ifeq ($(MAKECMDGOALS),TransferBenchCuda)
      $(info - Anvil executor not supported for CUDA builds; ignoring ENABLE_ANVIL_EXEC=1)
    else
      $(info Attempting to build with Anvil GPU-initiated SDMA executor support)
      HSAKMT_INC := $(firstword $(wildcard \
        $(ROCM_PATH)/include/hsakmt/hsakmt.h \
        /usr/include/hsakmt/hsakmt.h))
      HSAKMT_LIB := $(firstword $(wildcard \
        $(ROCM_PATH)/lib/libhsakmt.a \
        $(ROCM_PATH)/lib64/libhsakmt.a \
        $(ROCM_PATH)/lib/libhsakmt.so \
        /usr/lib/x86_64-linux-gnu/libhsakmt.so))
      ifeq ($(HSAKMT_INC),)
        $(info - hsakmt/hsakmt.h not found; cannot build Anvil executor)
        $(info - Install libhsakmt-dev or ensure ROCM_PATH is set correctly)
      else ifeq ($(HSAKMT_LIB),)
        $(info - libhsakmt not found; cannot build Anvil executor)
        $(info - Install libhsakmt or ensure ROCM_PATH is set correctly)
      else
        ANVIL_ENABLED = 1
        COMMON_FLAGS += -DANVIL_EXEC_ENABLED -I./src/anvil
        # XIO_SDMA_OSS7: fused COPY_LINEAR_WAIT_SIGNAL_MI4 packet ABI. Portable
        # across archs - the MI4 structs and host code are ABI-only, and the fused
        # *device* code is arch-gated via XIO_SDMA_OSS7_ENABLED (anvil_device.hpp)
        # so it only codegens on gfx1250/gfx950 regardless of GPU_TARGETS. The
        # fused path is still selected at runtime only on gfx1250. On by default;
        # set ENABLE_XIO_SDMA_OSS7=0 to force the separate COPY_LINEAR + ATOMIC path.
        ENABLE_XIO_SDMA_OSS7 ?= 1
        ifeq ($(ENABLE_XIO_SDMA_OSS7), 1)
          COMMON_FLAGS += -DXIO_SDMA_OSS7=1
          $(info - Building with XIO_SDMA_OSS7 (fused MI4 SDMA packets; device code gated to gfx1250/gfx950))
        else
          $(info - Building without XIO_SDMA_OSS7 (separate COPY_LINEAR + ATOMIC path))
        endif
        # Link hsakmt; use -Wl, to pass the static archive directly to the
        # linker — without it, -x hip causes the compiler to parse the .a
        # as source and emit "expected unqualified-id" / UTF-8 errors.
        ifeq ($(suffix $(HSAKMT_LIB)),.a)
          LDFLAGS += -Wl,$(HSAKMT_LIB) -ldrm_amdgpu -ldrm
        else
          LDFLAGS += -lhsakmt
        endif
        $(info - Building with Anvil GPU-initiated SDMA executor. Can set ENABLE_ANVIL_EXEC=0 to disable)
      endif
    endif
    ifeq ($(ANVIL_ENABLED), 0)
      $(info - Building without Anvil GPU-initiated SDMA executor support)
    endif
  endif
endif

.PHONY : all clean

all: TransferBench

ANVIL_SRCS =
ifeq ($(ANVIL_ENABLED), 1)
  ANVIL_SRCS = ./src/anvil/anvil.cpp
endif

TransferBench: ./src/client/Client.cpp $(ANVIL_SRCS) $(shell find -regex ".*\.\hpp")
	$(CXX) $(CXXFLAGS) $(HIPFLAGS) $(COMMON_FLAGS) ./src/client/Client.cpp $(ANVIL_SRCS) -o $@ $(HIPLDFLAGS) $(LDFLAGS)

TransferBenchCuda: ./src/client/Client.cpp $(shell find -regex ".*\.\hpp")
	$(NVCC) $(NVFLAGS) $(COMMON_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f ./TransferBench ./TransferBenchCuda
