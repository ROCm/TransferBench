# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# CMake Toolchain file to define compilers and path to ROCm
#==================================================================================================
if (NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/toolchain-linux.cmake")
  message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
endif()

set(VERSION_STRING "1.63.00")
project(TransferBench VERSION ${VERSION_STRING} LANGUAGES CXX)

## Load CMake modules
#==================================================================================================
include(CheckIncludeFiles)
include(CheckSymbolExists)
include(cmake/Dependencies.cmake) # rocm-cmake, rocm_local_targets

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Build options
#==================================================================================================
option(BUILD_LOCAL_GPU_TARGET_ONLY "Build only for GPUs detected on this machine" OFF)
option(ENABLE_NIC_EXEC             "Enable RDMA NIC Executor in TransferBench"    OFF)

# Default GPU architectures to build
#==================================================================================================
set(DEFAULT_GPUS
      gfx906
      gfx908
      gfx90a
      gfx942
      gfx950
      gfx1030
      gfx1100
      gfx1101
      gfx1102
      gfx1150
      gfx1151
      gfx1200
      gfx1201)

## Build only for local GPU architecture
if(BUILD_LOCAL_GPU_TARGET_ONLY)
  message(STATUS "Building only for local GPU target")
  if (COMMAND rocm_local_targets)
    rocm_local_targets(DEFAULT_GPUS)
  else()
    message(WARNING "Unable to determine local GPU targets. Falling back to default GPUs.")
  endif()
endif()

## Determine which GPU architectures to build for
set(GPU_TARGETS "${DEFAULT_GPUS}" CACHE STRING "Target default GPUs if GPU_TARGETS is not defined.")

## Check if clang compiler can offload to GPU_TARGETS
if (COMMAND rocm_check_target_ids)
  message(STATUS "Checking for ROCm support for GPU targets: " "${GPU_TARGETS}")
  rocm_check_target_ids(SUPPORTED_GPUS TARGETS ${GPU_TARGETS})
else()
  message(WARNING "Unable to check for supported GPU targets. Falling back to default GPUs.")
  set(SUPPORTED_GPUS ${DEFAULT_GPUS})
endif()

set(GPU_TARGETS "${SUPPORTED_GPUS}")
message(STATUS "Compiling for ${GPU_TARGETS}")

## NOTE: Reload rocm-cmake in order to update GPU_TARGETS
include(cmake/Dependencies.cmake) # Reloading to use desired GPU_TARGETS instead of defaults

# Check for required dependencies
#==================================================================================================
## Try to establish ROCM_PATH (for find_package)
if(NOT DEFINED ROCM_PATH)
  # Guess default location
  set(ROCM_PATH "/opt/rocm")
  message(WARNING "Unable to find ROCM_PATH: Falling back to ${ROCM_PATH}")
else()
  message(STATUS "ROCM_PATH found: ${ROCM_PATH}")
endif()
set(ENV{ROCM_PATH} ${ROCM_PATH})

## Set CMAKE flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_PREFIX_PATH  # Add ROCM_PATH to CMake search paths for finding HIP / HSA
            ${ROCM_PATH}
            ${ROCM_PATH}/llvm
            ${ROCM_PATH}/hip
            /opt/rocm
            /opt/rocm/llvm
            /opt/rocm/hip)

## Check for HIP
find_package(hip REQUIRED CONFIG PATHS ${CMAKE_PREFIX_PATH})
message(STATUS "HIP compiler: ${HIP_COMPILER}")

## Ensuring that CXX compiler meets expectations
if(NOT (("${CMAKE_CXX_COMPILER}" MATCHES ".*hipcc") OR ("${CMAKE_CXX_COMPILER}" MATCHES ".*clang\\+\\+")))
  message(FATAL_ERROR "On ROCm platform 'hipcc' or HIP-aware Clang must be used as C++ compiler.")
endif()

## Check for Threads
find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

## Check for numa support
find_library(NUMA_LIBRARY numa)
find_path(NUMA_INCLUDE_DIR numa.h)
if(NUMA_LIBRARY AND NUMA_INCLUDE_DIR)
  add_library(numa SHARED IMPORTED)
  set_target_properties(numa PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}" IMPORTED_LOCATION "${NUMA_LIBRARY}" INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
endif()

## Check for hsa support
find_library(HSA_LIBRARY hsa-runtime64 PATHS ${ROCM_PATH} ${ROCM_PATH}/lib)
find_path(HSA_INCLUDE_DIR hsa.h PATHS ${ROCM_PATH}/include ${ROCM_PATH}/include/hsa)
if(HSA_LIBRARY AND HSA_INCLUDE_DIR)
  add_library(hsa-runtime64 SHARED IMPORTED)
  set_target_properties(hsa-runtime64 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HSA_INCLUDE_DIR}" IMPORTED_LOCATION "${HSA_LIBRARY}" INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HSA_INCLUDE_DIR}")
endif()

## Check for infiniband verbs support
if(DEFINED ENV{ENABLE_NIC_EXEC} OR DEFINED ENABLE_NIC_EXEC)
  message(STATUS "For CMake builds, NIC executor also requires explicit opt-in by setting CMake flag -DENABLE_NIC_EXEC=1 or environment flag ENABLE_NIC_EXEC=1")
  find_library(IBVERBS_LIBRARY ibverbs)
  find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h)
  if(IBVERBS_LIBRARY AND IBVERBS_INCLUDE_DIR)
    add_library(ibverbs SHARED IMPORTED)
    set_target_properties(ibverbs PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${IBVERBS_INCLUDE_DIR}" IMPORTED_LOCATION "${IBVERBS_LIBRARY}" INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${IBVERBS_INCLUDE_DIR}")
    set(IBVERBS_FOUND 1)
    message(STATUS "Building with NIC executor support. Can set DISABLE_NIC_EXEC=1 to disable")
  else()
    if(NOT IBVERBS_LIBRARY)
      message(WARNING "IBVerbs library not found")
    elseif(NOT IBVERBS_INCLUDE_DIR)
      message(WARNING "infiniband/verbs.h not found")
    endif()
    message(WARNING "Building without NIC executor support. To use the TransferBench RDMA executor, check if your system has NICs, the NIC drivers are installed, and libibverbs-dev is installed")
  endif()
else()
  message(STATUS "Disabling NIC Executor support")
  message(WARNING "For CMake builds, NIC executor requires explicit opt-in by setting ENABLE_NIC_EXEC=1")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY .)

add_executable(TransferBench src/client/Client.cpp)

target_include_directories(TransferBench PRIVATE src/header)
target_include_directories(TransferBench PRIVATE src/client)
target_include_directories(TransferBench PRIVATE src/client/Presets)
target_include_directories(TransferBench PRIVATE ${NUMA_INCLUDE_DIR})
target_include_directories(TransferBench PRIVATE ${HSA_INCLUDE_DIR})
if(IBVERBS_FOUND)
  target_include_directories(TransferBench PRIVATE ${IBVERBS_INCLUDE_DIR})
  target_link_libraries(TransferBench PRIVATE ${IBVERBS_LIBRARY})
  target_compile_definitions(TransferBench PRIVATE NIC_EXEC_ENABLED)
endif()

target_link_libraries(TransferBench PRIVATE -fgpu-rdc)             # Required when linking relocatable device code
target_link_libraries(TransferBench PRIVATE Threads::Threads)
target_link_libraries(TransferBench INTERFACE hip::host)
target_link_libraries(TransferBench PRIVATE hip::device)
target_link_libraries(TransferBench PRIVATE dl)
target_link_libraries(TransferBench PRIVATE ${NUMA_LIBRARY})
target_link_libraries(TransferBench PRIVATE ${HSA_LIBRARY})

rocm_install(TARGETS TransferBench COMPONENT devel)
rocm_setup_version(VERSION ${VERSION_STRING})

# Package specific CPACK vars
rocm_package_add_dependencies(DEPENDS "numactl" "hsa-rocr")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

set(PACKAGE_NAME TB)
set(LIBRARY_NAME TransferBench)

rocm_create_package(
  NAME ${LIBRARY_NAME}
  DESCRIPTION "TransferBench package"
  MAINTAINER "RCCL Team <gilbert.lee@amd.com>"
)
