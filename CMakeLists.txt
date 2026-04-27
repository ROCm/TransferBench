# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

# MPI::MPI_CXX and hip:: config targets require >= 3.9; 3.16 for modern policy defaults.
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# Pre-project: ROCM_PATH detection and compiler/flag selection.
# Must be before project() so CMake uses the right compiler on first configure.
# Priority: -DROCM_PATH / $ROCM_PATH env > PATH (amdclang++) > /opt/rocm
#==================================================================================================

# ROCM_PATH: Check CMake cache or environment.
if(NOT ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to ROCm installation.")
  endif()
endif()

# ROCM_PATH: Derive from PATH; walk up from amdclang++/clang++ to the ROCm root.
# Handles both ${ROCM_PATH}/bin/ and ${ROCM_PATH}/llvm/bin/ layouts.
if(NOT ROCM_PATH)
  find_program(_rocm_bin_hint NAMES amdclang++ clang++)
  if(_rocm_bin_hint)
    get_filename_component(_bin_dir    "${_rocm_bin_hint}" DIRECTORY)
    get_filename_component(_parent     "${_bin_dir}"       DIRECTORY)
    if(EXISTS "${_parent}/lib/libamdhip64.so" OR EXISTS "${_parent}/lib64/libamdhip64.so")
      set(ROCM_PATH "${_parent}" CACHE PATH "Path to ROCm installation (auto-detected from PATH).")
      message(STATUS "ROCM_PATH auto-detected from PATH: ${ROCM_PATH}")
    else()
      get_filename_component(_grandparent "${_parent}" DIRECTORY)
      if(EXISTS "${_grandparent}/lib/libamdhip64.so" OR EXISTS "${_grandparent}/lib64/libamdhip64.so")
        set(ROCM_PATH "${_grandparent}" CACHE PATH "Path to ROCm installation (auto-detected from PATH).")
        message(STATUS "ROCM_PATH auto-detected from PATH: ${ROCM_PATH}")
      endif()
    endif()
  endif()
  unset(_rocm_bin_hint CACHE)
  unset(_rocm_bin_hint)
  unset(_bin_dir)
  unset(_parent)
  unset(_grandparent)
endif()

# ROCM_PATH: Fallback.
if(NOT ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation.")
  message(WARNING "ROCM_PATH not found; falling back to ${ROCM_PATH}")
endif()

if(NOT EXISTS "${ROCM_PATH}")
  message(FATAL_ERROR "ROCM_PATH=${ROCM_PATH} does not exist")
endif()

message(STATUS "ROCM_PATH: ${ROCM_PATH}")

# Compiler detection: amdclang++ > llvm/amdclang++ > llvm/clang++
# Respects -DCMAKE_CXX_COMPILER and $CXX / $CC env vars.
if(NOT CMAKE_CXX_COMPILER)
  if(DEFINED ENV{CXX} AND NOT "$ENV{CXX}" STREQUAL "")
    set(CMAKE_CXX_COMPILER "$ENV{CXX}" CACHE PATH "Path to C++ compiler")
  elseif(EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/amdclang++" CACHE PATH "Path to C++ compiler")
  elseif(EXISTS "${ROCM_PATH}/llvm/bin/amdclang++")
    set(CMAKE_CXX_COMPILER "${ROCM_PATH}/llvm/bin/amdclang++" CACHE PATH "Path to C++ compiler")
  elseif(EXISTS "${ROCM_PATH}/llvm/bin/clang++")
    set(CMAKE_CXX_COMPILER "${ROCM_PATH}/llvm/bin/clang++" CACHE PATH "Path to C++ compiler")
  else()
    message(FATAL_ERROR
      "Cannot find amdclang++/clang++ under ${ROCM_PATH}/bin or ${ROCM_PATH}/llvm/bin")
  endif()
endif()

if(NOT CMAKE_C_COMPILER)
  if(DEFINED ENV{CC} AND NOT "$ENV{CC}" STREQUAL "")
    set(CMAKE_C_COMPILER "$ENV{CC}" CACHE PATH "Path to C compiler")
  else()
    get_filename_component(_cxx_dir  "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_cxx_name "${CMAKE_CXX_COMPILER}" NAME)
    string(REPLACE "clang++" "clang" _cc_name "${_cxx_name}")
    if(EXISTS "${_cxx_dir}/${_cc_name}")
      set(CMAKE_C_COMPILER "${_cxx_dir}/${_cc_name}" CACHE PATH "Path to C compiler")
    endif()
    unset(_cxx_dir)
    unset(_cxx_name)
    unset(_cc_name)
  endif()
endif()

# Seed default per-config flags. _INIT vars are written to cache on first configure;
# user overrides via -DCMAKE_CXX_FLAGS_DEBUG=... or $CXXFLAGS/$CFLAGS take precedence.
if(NOT (DEFINED ENV{CXXFLAGS} AND NOT "$ENV{CXXFLAGS}" STREQUAL ""))
  set(CMAKE_CXX_FLAGS_DEBUG_INIT          "-O0 -g -ggdb3")
  set(CMAKE_CXX_FLAGS_RELEASE_INIT        "-O3")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O3 -g")
endif()
if(NOT (DEFINED ENV{CFLAGS} AND NOT "$ENV{CFLAGS}" STREQUAL ""))
  set(CMAKE_C_FLAGS_DEBUG_INIT          "-O0 -g -ggdb3")
  set(CMAKE_C_FLAGS_RELEASE_INIT        "-O3")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO_INIT "-O3 -g")
endif()

set(ENV{ROCM_PATH} "${ROCM_PATH}")

# TransferBench project definitions
#==================================================================================================
set(VERSION_STRING "1.67.00")
project(TransferBench VERSION ${VERSION_STRING} LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Load CMake modules
# Extend MODULE_PATH before any include() that searches it.
#==================================================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CheckIncludeFiles)
include(CheckSymbolExists)
include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)

# Build options
#==================================================================================================
option(BUILD_LOCAL_GPU_TARGET_ONLY "Build only for GPUs detected on this machine" OFF)
option(ENABLE_NIC_EXEC             "Enable RDMA NIC Executor in TransferBench"    OFF)
option(ENABLE_MPI_COMM             "Enable MPI Communicator support"              OFF)
option(ENABLE_DMA_BUF              "Enable DMA-BUF support for GPU Direct RDMA"   OFF)
option(ENABLE_AMD_SMI              "Enable AMD-SMI pod membership queries"        OFF)
option(ENABLE_POD_COMM             "Enable pod communication"                     OFF)

include(cmake/Dependencies.cmake) # rocm-cmake, rocm_local_targets, rocm_check_target_ids

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
      gfx1201
      gfx1250)

if(BUILD_LOCAL_GPU_TARGET_ONLY)
  message(STATUS "Building only for local GPU target")
  if(COMMAND rocm_local_targets)
    rocm_local_targets(DEFAULT_GPUS)
  else()
    message(WARNING "Unable to determine local GPU targets. Falling back to default GPUs.")
  endif()
  # FORCE so re-runs pick up the freshly detected local set.
  set(GPU_TARGETS "${DEFAULT_GPUS}" CACHE STRING "GPU architectures to build for." FORCE)
else()
  # Seeded once on first configure; hip-config-amd.cmake applies the same priority but warns on AMDGPU_TARGETS.
  if(NOT DEFINED CACHE{GPU_TARGETS})
    if(DEFINED AMDGPU_TARGETS AND NOT AMDGPU_TARGETS STREQUAL "")
      set(GPU_TARGETS "${AMDGPU_TARGETS}" CACHE STRING "GPU architectures to build for.")
      message(STATUS "GPU_TARGETS seeded from CMake AMDGPU_TARGETS: ${GPU_TARGETS}")
    elseif(DEFINED ENV{AMDGPU_TARGETS} AND NOT "$ENV{AMDGPU_TARGETS}" STREQUAL "")
      set(GPU_TARGETS "$ENV{AMDGPU_TARGETS}" CACHE STRING "GPU architectures to build for.")
      message(STATUS "GPU_TARGETS seeded from environment AMDGPU_TARGETS: ${GPU_TARGETS}")
    else()
      set(GPU_TARGETS "${DEFAULT_GPUS}" CACHE STRING "GPU architectures to build for.")
    endif()
  endif()
endif()

# Check if clang can offload to each GPU_TARGETS entry.
if(COMMAND rocm_check_target_ids)
  message(STATUS "Checking for ROCm support for GPU targets: ${GPU_TARGETS}")
  rocm_check_target_ids(SUPPORTED_GPUS TARGETS ${GPU_TARGETS})
else()
  message(WARNING "Unable to check for supported GPU targets. Falling back to default GPUs.")
  set(SUPPORTED_GPUS ${DEFAULT_GPUS})
endif()

if(SUPPORTED_GPUS)
  set(GPU_TARGETS "${SUPPORTED_GPUS}" CACHE STRING "GPU architectures to build for." FORCE)
else()
  message(WARNING "rocm_check_target_ids returned no supported GPUs; keeping existing GPU_TARGETS=${GPU_TARGETS}")
endif()
message(STATUS "- Compiling for ${GPU_TARGETS}")

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Search only the active ROCm installation. ROCM_PATH is already resolved by the
# pre-project() block, so this is always the right install.
list(APPEND CMAKE_PREFIX_PATH
            ${ROCM_PATH}
            ${ROCM_PATH}/llvm
            ${ROCM_PATH}/hip)

find_package(hip REQUIRED CONFIG)
message(STATUS "HIP compiler: ${HIP_COMPILER}")

if(NOT (("${CMAKE_CXX_COMPILER}" MATCHES ".*hipcc") OR ("${CMAKE_CXX_COMPILER}" MATCHES ".*clang\\+\\+")))
  message(FATAL_ERROR "On ROCm platform 'hipcc' or HIP-aware Clang must be used as C++ compiler.")
endif()

## Check for Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

## Check for numa support
find_library(NUMA_LIBRARY numa)
find_path(NUMA_INCLUDE_DIR numa.h)
if(NUMA_LIBRARY AND NUMA_INCLUDE_DIR)
  add_library(numa SHARED IMPORTED)
  set_target_properties(numa PROPERTIES IMPORTED_LOCATION "${NUMA_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}" INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "NUMA library or headers not found; TransferBench requires libnuma")
endif()

## Check for hsa support
find_library(HSA_LIBRARY hsa-runtime64 PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64 NO_DEFAULT_PATH)
find_path(HSA_INCLUDE_DIR hsa/hsa.h PATHS ${ROCM_PATH}/include)
if(HSA_LIBRARY AND HSA_INCLUDE_DIR)
  add_library(hsa-runtime64 SHARED IMPORTED)
  set_target_properties(hsa-runtime64 PROPERTIES IMPORTED_LOCATION "${HSA_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${HSA_INCLUDE_DIR}" INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HSA_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "HSA library or headers not found under ${ROCM_PATH}; TransferBench requires libhsa-runtime64")
endif()

## Check for infiniband verbs support
if(DEFINED ENV{DISABLE_NIC_EXEC} AND "$ENV{DISABLE_NIC_EXEC}" STREQUAL "1")
  message(STATUS "Disabling NIC Executor support as env. flag DISABLE_NIC_EXEC was enabled")
elseif(NOT ENABLE_NIC_EXEC)
  message(STATUS "For CMake builds, NIC Executor support requires explicit opt-in by setting CMake flag -DENABLE_NIC_EXEC=ON")
  message(STATUS "- Disabling NIC Executor support")
else()
  message(STATUS "Attempting to build with NIC executor support")

  find_library(IBVERBS_LIBRARY ibverbs)
  find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h)
  if(IBVERBS_LIBRARY AND IBVERBS_INCLUDE_DIR)
    add_library(ibverbs SHARED IMPORTED)
    set_target_properties(ibverbs PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${IBVERBS_INCLUDE_DIR}" IMPORTED_LOCATION "${IBVERBS_LIBRARY}" INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${IBVERBS_INCLUDE_DIR}")
    set(IBVERBS_FOUND 1)
    message(STATUS "- Building with NIC executor support. Can set DISABLE_NIC_EXEC=1 to disable")
  else()
    if(NOT IBVERBS_LIBRARY)
      message(WARNING "- IBVerbs library not found")
    elseif(NOT IBVERBS_INCLUDE_DIR)
      message(WARNING "- infiniband/verbs.h not found")
    endif()
    message(WARNING "- Building without NIC executor support. To use the TransferBench RDMA executor, check if your system has NICs, the NIC drivers are installed, and libibverbs-dev is installed")
  endif()
endif()

## Check for DMA-BUF support (requires IBVERBS)
if(IBVERBS_FOUND)
  if(DEFINED ENV{DISABLE_DMA_BUF} AND "$ENV{DISABLE_DMA_BUF}" STREQUAL "1")
    message(STATUS "Disabling DMA-BUF support as env. flag DISABLE_DMA_BUF was enabled")
  elseif(NOT ENABLE_DMA_BUF)
    message(STATUS "For CMake builds, DMA-BUF support requires explicit opt-in by setting CMake flags -DENABLE_DMA_BUF=ON")
    message(STATUS "- Disabling DMA-BUF support")
  else()
    message(STATUS "Attempting to build with DMA-BUF support")

    # Check for ibv_reg_dmabuf_mr
    cmake_push_check_state()
    set(CMAKE_REQUIRED_INCLUDES ${IBVERBS_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES ${IBVERBS_LIBRARY})
    check_symbol_exists(ibv_reg_dmabuf_mr "infiniband/verbs.h" HAVE_IBV_DMABUF)
    cmake_pop_check_state()

    # Check for hsa_amd_portable_export_dmabuf
    cmake_push_check_state()
    set(CMAKE_REQUIRED_INCLUDES ${HSA_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES ${HSA_LIBRARY})
    check_symbol_exists(hsa_amd_portable_export_dmabuf "hsa/hsa_ext_amd.h" HAVE_ROCM_DMABUF)
    cmake_pop_check_state()

    # Enable DMA-BUF only if both APIs are available
    if(HAVE_IBV_DMABUF AND HAVE_ROCM_DMABUF)
      set(DMABUF_SUPPORT_FOUND 1)
      message(STATUS "- Building with DMA-BUF support")
    else()
      if(NOT HAVE_IBV_DMABUF AND NOT HAVE_ROCM_DMABUF)
        message(WARNING "- Building without DMA-BUF support: missing both ibv_reg_dmabuf_mr and ROCm DMA-BUF export")
      elseif(NOT HAVE_IBV_DMABUF)
        message(WARNING "- Building without DMA-BUF support: missing ibv_reg_dmabuf_mr")
      else()
        message(WARNING "- Building without DMA-BUF support: missing ROCm DMA-BUF export")
      endif()
    endif()
  endif()
endif()

## Check for MPI support
set(MPI_PATH "" CACHE PATH "Path to MPI installation (takes priority over system MPI)")
if(DEFINED ENV{DISABLE_MPI_COMM} AND "$ENV{DISABLE_MPI_COMM}" STREQUAL "1")
  message(STATUS "Disabling MPI Communicator support as env. flag DISABLE_MPI_COMM was enabled")
elseif(NOT ENABLE_MPI_COMM)
  message(STATUS "For CMake builds, MPI Communicator requires explicit opt-in by setting CMake flag -DENABLE_MPI_COMM=ON")
  message(STATUS "Disabling MPI Communicator support")
else()
  message(STATUS "Attempting to build with MPI communicator support")
  if(MPI_PATH AND EXISTS "${MPI_PATH}/include/mpi.h")
    find_library(MPI_LIBRARY NAMES mpi PATHS ${MPI_PATH}/lib ${MPI_PATH}/lib64 NO_DEFAULT_PATH)
    if(MPI_LIBRARY)
      set(MPI_COMM_FOUND 1)
      set(MPI_INCLUDE_DIR "${MPI_PATH}/include")
      message(STATUS "- Building with MPI Communicator support (found at MPI_PATH: ${MPI_PATH})")
    else()
      message(WARNING "- Found mpi.h at ${MPI_PATH}/include but could not find MPI library at ${MPI_PATH}/lib")
    endif()
  else()
    if(MPI_PATH)
      message(STATUS "- Unable to find mpi.h at ${MPI_PATH}/include, trying find_package")
    endif()
    find_package(MPI QUIET)
    if(MPI_CXX_FOUND)
      set(MPI_COMM_FOUND 1)
      message(STATUS "- Building with MPI Communicator support (found via find_package)")
      message(STATUS "  - Using MPI include path: ${MPI_CXX_INCLUDE_DIRS}")
      message(STATUS "  - Using MPI library: ${MPI_CXX_LIBRARIES}")
    else()
      message(WARNING "- MPI not found. Please specify appropriate MPI_PATH or install MPI libraries (e.g., OpenMPI or MPICH)")
    endif()
  endif()
endif()

## Check for pod communication support
if(ENABLE_AMD_SMI AND NOT ENABLE_POD_COMM)
  message(WARNING "ENABLE_AMD_SMI=ON has no effect without ENABLE_POD_COMM=ON; AMD-SMI detection will be skipped")
endif()
if(DEFINED ENV{DISABLE_POD_COMM} AND "$ENV{DISABLE_POD_COMM}" STREQUAL "1")
  message(STATUS "Disabling pod communication support as env. flag DISABLE_POD_COMM was enabled")
elseif(NOT ENABLE_POD_COMM)
  message(STATUS "For CMake builds, pod communication support requires explicit opt-in by setting CMake flag -DENABLE_POD_COMM=ON")
  message(STATUS "- Disabling pod communication support")
else()
  find_library(HIP_RUNTIME_LIBRARY amdhip64 PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64 NO_DEFAULT_PATH)
  if(NOT HIP_RUNTIME_LIBRARY)
    message(FATAL_ERROR "libamdhip64 not found under ${ROCM_PATH}/lib or ${ROCM_PATH}/lib64; cannot probe for HIP fabric API")
  endif()
  # Probe for the HIP fabric API functions used by TransferBench at runtime.
  cmake_push_check_state()
  set(CMAKE_REQUIRED_INCLUDES "${ROCM_PATH}/include")
  set(CMAKE_REQUIRED_LIBRARIES "${HIP_RUNTIME_LIBRARY}")
  set(CMAKE_REQUIRED_DEFINITIONS "-D__HIP_PLATFORM_AMD__")
  check_cxx_source_compiles("
    #include <hip/hip_runtime_api.h>
    int main() {
      hipMemFabricHandle_t fabricHandle = {};
      hipMemGenericAllocationHandle_t allocationHandle = {};
      hipMemExportToShareableHandle(&fabricHandle, allocationHandle, hipMemHandleTypeFabric, 0);
      hipMemImportFromShareableHandle(&allocationHandle, &fabricHandle, hipMemHandleTypeFabric);
      return 0;
    }" HIP_HAS_FABRIC_API)
  cmake_pop_check_state()

  if(HIP_HAS_FABRIC_API)
    message(STATUS "- HIP fabric API found; enabling pod communication support")
    set(POD_COMM_FOUND 1)

    # Check for AMD-SMI support
    # Try amd-smi for pod membership queries; fall back to TB_FORCE_SINGLE_POD=1 at runtime.
    if(DEFINED ENV{DISABLE_AMD_SMI} AND "$ENV{DISABLE_AMD_SMI}" STREQUAL "1")
      message(STATUS "- AMD-SMI disabled via env. flag DISABLE_AMD_SMI was enabled")
      message(WARNING "Set TB_FORCE_SINGLE_POD=1 at runtime to override pod membership")
    elseif(NOT ENABLE_AMD_SMI)
      message(STATUS "- For CMake builds, AMD-SMI requires explicit opt-in by setting CMake flag -DENABLE_AMD_SMI=ON")
      message(WARNING "Set TB_FORCE_SINGLE_POD=1 at runtime to override pod membership")
    else()
      find_path(AMD_SMI_INCLUDE_DIR amd_smi/amdsmi.h PATHS ${ROCM_PATH}/include NO_DEFAULT_PATH)
      find_library(AMD_SMI_LIBRARY amd_smi PATHS ${ROCM_PATH}/lib ${ROCM_PATH}/lib64 NO_DEFAULT_PATH)
      if(AMD_SMI_INCLUDE_DIR AND AMD_SMI_LIBRARY)
        # Probe for the AMD-SMI functions used by TransferBench at runtime.
        cmake_push_check_state()
        set(CMAKE_REQUIRED_INCLUDES "${AMD_SMI_INCLUDE_DIR}")
        set(CMAKE_REQUIRED_LIBRARIES "${AMD_SMI_LIBRARY}")
        check_cxx_source_compiles("
          #include <amd_smi/amdsmi.h>
          int main() {
            amdsmi_bdf_t bdf = {};
            amdsmi_processor_handle h;
            amdsmi_get_processor_handle_from_bdf(bdf, &h);
            amdsmi_fabric_info_t fi;
            amdsmi_get_gpu_fabric_info(h, &fi);
            (void)fi.fabric_info.fabric_version.v1.ppod_id;
            (void)fi.fabric_info.fabric_version.v1.vpod_id;
            return 0;
          }" AMDSMI_HAS_FABRIC)
        cmake_pop_check_state()

        if(AMDSMI_HAS_FABRIC)
          message(STATUS "- AMD-SMI fabric API found; using AMD-SMI for pod membership queries")
          set(AMD_SMI_FOUND 1)
        else()
          message(STATUS "- AMD-SMI fabric API not found")
          message(WARNING "Set TB_FORCE_SINGLE_POD=1 at runtime to override pod membership")
        endif()
      else()
        if(NOT AMD_SMI_INCLUDE_DIR)
          message(STATUS "- amd_smi/amdsmi.h not found under ${ROCM_PATH}/include")
        endif()
        if(NOT AMD_SMI_LIBRARY)
          message(STATUS "- libamd_smi not found under ${ROCM_PATH}/lib or ${ROCM_PATH}/lib64")
        endif()
        message(STATUS "- AMD-SMI not available")
        message(WARNING "Set TB_FORCE_SINGLE_POD=1 at runtime to override pod membership")
      endif()
    endif()
  else()
    message(STATUS "- HIP fabric API not found; disabling pod communication support")
  endif()
endif()

set(PACKAGE_NAME TB)
set(LIBRARY_NAME TransferBench)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

add_executable(TransferBench src/client/Client.cpp)

target_include_directories(TransferBench PRIVATE
  src/header
  src/client
  src/client/Presets)

if(IBVERBS_FOUND)
  target_include_directories(TransferBench PRIVATE ${IBVERBS_INCLUDE_DIR})
  target_link_libraries(TransferBench PRIVATE ${IBVERBS_LIBRARY})
  target_compile_definitions(TransferBench PRIVATE NIC_EXEC_ENABLED)
endif()
if(MPI_COMM_FOUND)
  if(TARGET MPI::MPI_CXX)
    target_link_libraries(TransferBench PRIVATE MPI::MPI_CXX)
  else()
    target_include_directories(TransferBench PRIVATE ${MPI_INCLUDE_DIR})
    target_link_libraries(TransferBench PRIVATE ${MPI_LIBRARY})
  endif()
  target_compile_definitions(TransferBench PRIVATE MPI_COMM_ENABLED)
endif()
if(DMABUF_SUPPORT_FOUND)
  target_compile_definitions(TransferBench PRIVATE HAVE_DMABUF_SUPPORT)
endif()
if(AMD_SMI_FOUND)
  target_include_directories(TransferBench PRIVATE ${AMD_SMI_INCLUDE_DIR})
  target_link_libraries(TransferBench PRIVATE ${AMD_SMI_LIBRARY})
  target_compile_definitions(TransferBench PRIVATE AMD_SMI_ENABLED)
endif()
if(POD_COMM_FOUND)
  target_compile_definitions(TransferBench PRIVATE POD_COMM_ENABLED)
endif()

check_cxx_compiler_flag(-parallel-jobs=12 HAVE_PARALLEL_JOBS)
if(HAVE_PARALLEL_JOBS)
  message(STATUS "Enabling parallel compile jobs: -parallel-jobs=12")
  target_compile_options(TransferBench PRIVATE -parallel-jobs=12)
else()
  message(STATUS "Compiler does not support -parallel-jobs=12 (or the check failed); skipping -parallel-jobs optimisation")
endif()

target_link_options(TransferBench PRIVATE -fgpu-rdc)
target_link_libraries(TransferBench PRIVATE Threads::Threads)
target_link_libraries(TransferBench PRIVATE hip::host hip::device dl)
target_link_libraries(TransferBench PRIVATE hsa-runtime64)
target_link_libraries(TransferBench PRIVATE numa)

rocm_install(TARGETS TransferBench COMPONENT devel)
rocm_setup_version(VERSION ${VERSION_STRING})

rocm_package_add_dependencies(DEPENDS "numactl" "hsa-rocr")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
set(CPACK_RPM_PACKAGE_LICENSE "MIT")

rocm_create_package(
  NAME ${LIBRARY_NAME}
  DESCRIPTION "TransferBench package"
  MAINTAINER "RCCL Team <gilbert.lee@amd.com>"
)
