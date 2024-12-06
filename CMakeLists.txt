# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
if (DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE STRING "ROCm install directory")
else()
    set(ROCM_PATH "/opt/rocm" CACHE STRING "ROCm install directory")
endif()
cmake_minimum_required(VERSION 3.5)

project(TransferBench VERSION 1.52.0 LANGUAGES CXX)

# Default GPU architectures to build
#==================================================================================================
set(DEFAULT_GPUS
      gfx906
      gfx908
      gfx90a
      gfx942
      gfx1030
      gfx1100
      gfx1101
      gfx1102
      gfx1200
      gfx1201)

# Build only for local GPU architecture
if (BUILD_LOCAL_GPU_TARGET_ONLY)
  message(STATUS "Building only for local GPU target")
  if (COMMAND rocm_local_targets)
    rocm_local_targets(DEFAULT_GPUS)
  else()
    message(WARNING "Unable to determine local GPU targets. Falling back to default GPUs.")
  endif()
endif()

# Determine which GPU architectures to build for
set(AMDGPU_TARGETS "${DEFAULT_GPUS}" CACHE STRING "Target default GPUs if AMDGPU_TARGETS is not defined.")

# Check if clang compiler can offload to AMDGPU_TARGETS
if (COMMAND rocm_check_target_ids)
  message(STATUS "Checking for ROCm support for GPU targets: " "${AMDGPU_TARGETS}")
  rocm_check_target_ids(SUPPORTED_GPUS TARGETS ${AMDGPU_TARGETS})
else()
  message(WARNING "Unable to check for supported GPU targets. Falling back to default GPUs.")
  set(SUPPORTED_GPUS ${DEFAULT_GPUS})
endif()

set(GPU_TARGETS "${SUPPORTED_GPUS}" CACHE STRING "GPU targets to compile for.")
message(STATUS "Compiling for ${GPU_TARGETS}")

foreach(target ${GPU_TARGETS})
 list(APPEND static_link_flags --offload-arch=${target})
endforeach()
list(JOIN static_link_flags " " flags_str)
set( CMAKE_CXX_FLAGS "${flags_str} ${CMAKE_CXX_FLAGS}")

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -L${ROCM_PATH}/lib")
include_directories(${ROCM_PATH}/include)
link_libraries(numa hsa-runtime64 pthread)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ..)
add_executable(TransferBench src/TransferBench.cpp)
target_include_directories(TransferBench PRIVATE src/include)

find_package(ROCM 0.8 REQUIRED PATHS ${ROCM_PATH})
include(ROCMInstallTargets)
include(ROCMCreatePackage)
set(ROCMCHECKS_WARN_TOOLCHAIN_VAR OFF)

set(PACKAGE_NAME TB)
set(LIBRARY_NAME TransferBench)

rocm_install(TARGETS TransferBench COMPONENT devel)

rocm_package_add_dependencies(DEPENDS numactl hsa-rocr)

rocm_create_package(
    NAME ${LIBRARY_NAME}
    DESCRIPTION "TransferBench package"
    MAINTAINER "RCCL Team <gilbert.lee@amd.com>"
)
