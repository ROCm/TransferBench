# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# CMake Toolchain file to define compilers and path to ROCm
#==================================================================================================
if (NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/toolchain-linux.cmake")
  message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
endif()

set(VERSION_STRING "1.66.02")
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
option(ENABLE_MPI_COMM             "Enable MPI Communicator support"              OFF)
option(DISABLE_DMABUF              "Disable DMA-BUF support for GPU Direct RDMA"  ON)
option(BUILD_RELOCATABLE_PACKAGE   "Build with RVS-style relocatable RPATH and amdrocm<MAJOR>-transferbench package naming" OFF)

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
if(DEFINED ENV{DISABLE_NIC_EXEC} AND "$ENV{DISABLE_NIC_EXEC}" STREQUAL "1")
  message(STATUS "Disabling NIC Executor support as env. flag DISABLE_NIC_EXEC was enabled")
elseif(NOT ENABLE_NIC_EXEC)
  message(STATUS "For CMake builds, NIC executor so requires explicit opt-in by setting CMake flag -DENABLE_NIC_EXEC=ON")
  message(STATUS "Disabling NIC Executor support")
else()
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
endif()

## Check for DMA-BUF support (requires IBVERBS_FOUND)
if(IBVERBS_FOUND AND NOT DISABLE_DMABUF)
  message(STATUS "Checking for DMA-BUF support...")

  # Check for ibv_reg_dmabuf_mr
  include(CheckSymbolExists)
  set(CMAKE_REQUIRED_INCLUDES ${IBVERBS_INCLUDE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES ${IBVERBS_LIBRARY})
  check_symbol_exists(ibv_reg_dmabuf_mr "infiniband/verbs.h" HAVE_IBV_DMABUF)

  # Check for hsa_amd_portable_export_dmabuf
  set(CMAKE_REQUIRED_INCLUDES ${HSA_INCLUDE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES ${HSA_LIBRARY})
  check_symbol_exists(hsa_amd_portable_export_dmabuf "hsa_ext_amd.h" HAVE_ROCM_DMABUF)

  # Enable DMA-BUF only if both APIs are available
  if(HAVE_IBV_DMABUF AND HAVE_ROCM_DMABUF)
    set(DMABUF_SUPPORT_FOUND 1)
    message(STATUS "Building with DMA-BUF support")
  else()
    if(NOT HAVE_IBV_DMABUF AND NOT HAVE_ROCM_DMABUF)
      message(WARNING "Building without DMA-BUF support: missing both ibv_reg_dmabuf_mr and ROCm DMA-BUF export")
    elseif(NOT HAVE_IBV_DMABUF)
      message(WARNING "Building without DMA-BUF support: missing ibv_reg_dmabuf_mr")
    else()
      message(WARNING "Building without DMA-BUF support: missing ROCm DMA-BUF export")
    endif()
  endif()
elseif(NOT DISABLE_DMABUF)
  message(WARNING "DMA-BUF support requires ENABLE_NIC_EXEC=ON")
endif()

## Check for MPI support
set(MPI_PATH "" CACHE PATH "Path to MPI installation (takes priority over system MPI)")
if(NOT ENABLE_MPI_COMM)
  message(STATUS "For CMake builds, MPI Communicator requires explicit opt-in by setting CMake flag -DENABLE_MPI_COMM=ON")
  message(STATUS "Disabling MPI Communicator support")
else()
  # First check user-specified MPI_PATH (similar to Makefile)
  if(MPI_PATH AND EXISTS "${MPI_PATH}/include/mpi.h")
    find_library(MPI_LIBRARY NAMES mpi PATHS ${MPI_PATH}/lib NO_DEFAULT_PATH)
    if(MPI_LIBRARY)
      set(MPI_COMM_FOUND 1)
      set(MPI_INCLUDE_DIR "${MPI_PATH}/include")
      set(MPI_LINK_DIR "${MPI_PATH}/lib")
      message(STATUS "Building with MPI Communicator support (found at MPI_PATH: ${MPI_PATH})")
    else()
      message(WARNING "Found mpi.h at ${MPI_PATH}/include but could not find MPI library at ${MPI_PATH}/lib")
    endif()
  else()
    # Fall back to find_package
    if(MPI_PATH)
      message(STATUS "Unable to find mpi.h at ${MPI_PATH}/include, trying find_package")
    endif()
    find_package(MPI QUIET)
    if(MPI_CXX_FOUND)
      set(MPI_COMM_FOUND 1)
      message(STATUS "Building with MPI Communicator support (found via find_package)")
      message(STATUS "- Using MPI include path: ${MPI_CXX_INCLUDE_PATH}")
      message(STATUS "- Using MPI library:: ${MPI_CXX_LIBRARIES}")
    else()
      message(WARNING "MPI not found. Please specify appropriate MPI_PATH or install MPI libraries (e.g., OpenMPI or MPICH)")
    endif()
  endif()
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
if(MPI_COMM_FOUND)
  if(TARGET MPI::MPI_CXX)
    # Found via find_package
    target_include_directories(TransferBench PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(TransferBench PRIVATE MPI::MPI_CXX)
  else()
    # Found via MPI_PATH fallback
    target_include_directories(TransferBench PRIVATE ${MPI_INCLUDE_DIR})
    target_link_libraries(TransferBench PRIVATE ${MPI_LIBRARY})
  endif()
  target_compile_definitions(TransferBench PRIVATE MPI_COMM_ENABLED)
endif()
if(DMABUF_SUPPORT_FOUND)
  target_compile_definitions(TransferBench PRIVATE HAVE_DMABUF_SUPPORT)
endif()
if (HAVE_PARALLEL_JOBS)
  target_compile_options(TransferBench PRIVATE -parallel-jobs=12)
endif()


target_link_libraries(TransferBench PRIVATE -fgpu-rdc)             # Required when linking relocatable device code
target_link_libraries(TransferBench PRIVATE Threads::Threads)
target_link_libraries(TransferBench INTERFACE hip::host)
target_link_libraries(TransferBench PRIVATE hip::device)
target_link_libraries(TransferBench PRIVATE dl)
target_link_libraries(TransferBench PRIVATE ${NUMA_LIBRARY})
target_link_libraries(TransferBench PRIVATE ${HSA_LIBRARY})

if(BUILD_RELOCATABLE_PACKAGE)
  # RVS-style relocatable packaging: bypass rocm_install/rocm_create_package and
  # drive CPack directly so CMAKE_INSTALL_PREFIX / CPACK_PACKAGING_INSTALL_PREFIX
  # set by the caller (build_packages_local.sh) are honored.
  if(NOT DEFINED ROCM_MAJOR_VERSION)
    set(ROCM_MAJOR_VERSION "7")
  endif()

  install(TARGETS TransferBench RUNTIME DESTINATION bin COMPONENT devel)

  set(CPACK_PACKAGE_NAME "amdrocm${ROCM_MAJOR_VERSION}-transferbench")
  set(CPACK_PACKAGE_VERSION "${VERSION_STRING}")
  set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
  set(CPACK_PACKAGE_CONTACT "RCCL Team <gilbert.lee@amd.com>")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "TransferBench: benchmark simultaneous transfers between CPU/GPU/NIC")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

  # DEB
  set(CPACK_DEBIAN_PACKAGE_NAME       "${CPACK_PACKAGE_NAME}")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS    "numactl, libnuma1, hsa-rocr")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_CONTACT}")
  if(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE "$ENV{CPACK_DEBIAN_PACKAGE_RELEASE}")
  endif()

  # RPM
  set(CPACK_RPM_PACKAGE_NAME    "${CPACK_PACKAGE_NAME}")
  set(CPACK_RPM_PACKAGE_LICENSE "MIT")
  set(CPACK_RPM_PACKAGE_REQUIRES "numactl, hsa-rocr")
  set(CPACK_RPM_PACKAGE_VENDOR  "${CPACK_PACKAGE_VENDOR}")
  if(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE "$ENV{CPACK_RPM_PACKAGE_RELEASE}")
  endif()
  # Use the actual install prefix (caller-controlled in relocatable mode)
  # rather than hard-coded /opt/... paths.
  if(DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
    set(_rpm_exclude_prefix "${CPACK_PACKAGING_INSTALL_PREFIX}")
  else()
    set(_rpm_exclude_prefix "${CMAKE_INSTALL_PREFIX}")
  endif()
  set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION
        "/opt" "/opt/rocm"
        "${_rpm_exclude_prefix}"
        "${_rpm_exclude_prefix}/bin")

  # TGZ
  set(CPACK_ARCHIVE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-Linux")

  set(CPACK_GENERATOR "DEB;RPM;TGZ")
  include(CPack)
else()
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
endif()
