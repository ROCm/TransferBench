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
set(TRANSFERBENCH_VERSION_MAJOR 1)
set(TRANSFERBENCH_VERSION_MINOR 67)
set(TRANSFERBENCH_VERSION_PATCH_FALLBACK "00")

# Auto-compute patch from git: count commits since the last v<MAJOR>.<MINOR>.* tag.
# Falls back to TRANSFERBENCH_VERSION_PATCH_FALLBACK when git is unavailable,
# this is not a git checkout, or no matching tag exists. Mirrors the RVS flow.
set(TRANSFERBENCH_VERSION_PATCH "${TRANSFERBENCH_VERSION_PATCH_FALLBACK}")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --tags --abbrev=0 --match
            "v${TRANSFERBENCH_VERSION_MAJOR}.${TRANSFERBENCH_VERSION_MINOR}.*"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tb_last_tag
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _tb_describe_rc
    ERROR_QUIET)
  if(_tb_describe_rc EQUAL 0 AND _tb_last_tag)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-list --count "${_tb_last_tag}..HEAD"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      OUTPUT_VARIABLE _tb_commit_count
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _tb_count_rc
      ERROR_QUIET)
    if(_tb_count_rc EQUAL 0 AND _tb_commit_count MATCHES "^[0-9]+$")
      set(TRANSFERBENCH_VERSION_PATCH "${_tb_commit_count}")
    endif()
  endif()
endif()

set(VERSION_STRING
    "${TRANSFERBENCH_VERSION_MAJOR}.${TRANSFERBENCH_VERSION_MINOR}.${TRANSFERBENCH_VERSION_PATCH}")
message(STATUS "TransferBench version: ${VERSION_STRING}")

project(TransferBench VERSION ${VERSION_STRING} LANGUAGES CXX)

# Git metadata: branch + short commit hash
# Priority: git rev-parse > GIT_VERSION file (populated by packaging scripts) > "unknown"
if(GIT_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE TB_GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    RESULT_VARIABLE _tb_git_branch_result
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE TB_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
    RESULT_VARIABLE _tb_git_commit_result
  )
endif()
if(NOT TB_GIT_BRANCH OR TB_GIT_BRANCH STREQUAL "" OR
   NOT TB_GIT_COMMIT OR TB_GIT_COMMIT STREQUAL "")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/GIT_VERSION")
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/GIT_VERSION" _tb_version_lines LIMIT_COUNT 2)
    list(LENGTH _tb_version_lines _tb_version_line_count)
    if(_tb_version_line_count GREATER 0)
      list(GET _tb_version_lines 0 TB_GIT_BRANCH)
    else()
      set(TB_GIT_BRANCH "unknown")
    endif()
    if(_tb_version_line_count GREATER 1)
      list(GET _tb_version_lines 1 TB_GIT_COMMIT)
    else()
      set(TB_GIT_COMMIT "unknown")
    endif()
  else()
    set(TB_GIT_BRANCH "unknown")
    set(TB_GIT_COMMIT "unknown")
  endif()
endif()
message(STATUS "TransferBench ${VERSION_STRING} (${TB_GIT_BRANCH}:${TB_GIT_COMMIT})")


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
option(ENABLE_MPI_COMM             "Enable MPI Communicator support"              OFF)
option(ENABLE_AMD_SMI              "Enable AMD-SMI pod membership queries"        OFF)
option(ENABLE_POD_COMM             "Enable pod communication"                     OFF)
option(BUILD_RELOCATABLE_PACKAGE   "Build with RVS-style relocatable RPATH and amdrocm<MAJOR>-transferbench package naming" OFF)

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
    rocm_local_targets(LOCAL_GPU_TARGETS)
    if(LOCAL_GPU_TARGETS)
      set(DEFAULT_GPUS ${LOCAL_GPU_TARGETS})
    else()
      message(WARNING "No local GPUs detected; falling back to default GPU list.")
    endif()
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

## NIC / RDMA executor
##
## libibverbs is loaded dynamically at runtime via dlopen/dlsym
## (see third-party/ibverbs/IbvDynLoad.hpp). The build neither links against
## -libverbs nor requires libibverbs-dev to be installed on the build host.
## Only libdl needs to be linked so that dlopen/dlsym resolve.
message(STATUS "NIC executor: libibverbs is loaded dynamically at runtime (no -libverbs link, no build-host dependency)")

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
  src/client/Presets
  third-party/ibverbs)

# libdl supplies dlopen/dlsym used by third-party/ibverbs/IbvDynLoad.hpp.
target_link_libraries(TransferBench PRIVATE ${CMAKE_DL_LIBS})
if(MPI_COMM_FOUND)
  if(TARGET MPI::MPI_CXX)
    target_link_libraries(TransferBench PRIVATE MPI::MPI_CXX)
  else()
    target_include_directories(TransferBench PRIVATE ${MPI_INCLUDE_DIR})
    target_link_libraries(TransferBench PRIVATE ${MPI_LIBRARY})
  endif()
  target_compile_definitions(TransferBench PRIVATE MPI_COMM_ENABLED)
endif()
if(AMD_SMI_FOUND)
  target_include_directories(TransferBench PRIVATE ${AMD_SMI_INCLUDE_DIR})
  target_link_libraries(TransferBench PRIVATE ${AMD_SMI_LIBRARY})
  target_compile_definitions(TransferBench PRIVATE AMD_SMI_ENABLED)
endif()
if(POD_COMM_FOUND)
  target_compile_definitions(TransferBench PRIVATE POD_COMM_ENABLED)
endif()

target_compile_definitions(TransferBench PRIVATE
  TB_GIT_BRANCH="${TB_GIT_BRANCH}"
  TB_GIT_COMMIT="${TB_GIT_COMMIT}"
)

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

# gcc <9 ships std::filesystem in a separate library (libstdc++fs).
# Required on AlmaLinux 8 / manylinux_2_28; harmless no-op stub on newer toolchains.
target_link_libraries(TransferBench PRIVATE stdc++fs)

if(BUILD_RELOCATABLE_PACKAGE)
  # RVS-style relocatable packaging: bypass rocm_install/rocm_create_package and
  # drive CPack directly so CMAKE_INSTALL_PREFIX / CPACK_PACKAGING_INSTALL_PREFIX
  # set by the caller (build_packages_local.sh) are honored.
  #
  # The CPack flow below uses CPACK_ARCHIVE_FILE_NAME (3.13+) and the
  # DEB-DEFAULT / RPM-DEFAULT canonical-naming sentinels (3.6+). The
  # project-wide cmake_minimum_required is 3.5 for the non-packaging build, so
  # gate the relocatable path locally to avoid silently producing a literal
  # "DEB-DEFAULT" filename on a too-old CMake.
  if(CMAKE_VERSION VERSION_LESS 3.13)
    message(FATAL_ERROR
      "BUILD_RELOCATABLE_PACKAGE requires CMake >= 3.13 "
      "(found ${CMAKE_VERSION}); needed for CPACK_ARCHIVE_FILE_NAME and "
      "the DEB-DEFAULT / RPM-DEFAULT canonical-naming sentinels.")
  endif()
  if(NOT DEFINED ROCM_MAJOR_VERSION)
    set(ROCM_MAJOR_VERSION "7")
  endif()

  # Relocatable RPATH (matches the RVS reference flow). Mirrors what
  # build_packages_local.sh used to inject via -DCMAKE_INSTALL_RPATH=, so
  # plain `cmake -DBUILD_RELOCATABLE_PACKAGE=ON ..` now produces the same
  # RPATH as a CI/packaged build.
  set(CMAKE_SKIP_RPATH FALSE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
  set(CMAKE_INSTALL_RPATH
      "\$ORIGIN:\$ORIGIN/../lib:/opt/rocm/extras-${ROCM_MAJOR_VERSION}/lib:/opt/rocm/lib:/opt/rocm/lib/llvm/lib:/opt/rocm/core-${ROCM_MAJOR_VERSION}/lib:/opt/rocm/core-${ROCM_MAJOR_VERSION}/lib/llvm/lib")
  set(CMAKE_BUILD_RPATH "${CMAKE_INSTALL_RPATH}")
  # Strip implicit SDK-from-build-host paths on install so the ephemeral
  # $HOME/rocm-sdk/install path the CI script uses does not leak into the
  # packaged binary's RPATH.
  if(NOT CMAKE_VERSION VERSION_LESS "3.16")
    set(CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE)
  endif()

  install(TARGETS TransferBench RUNTIME DESTINATION bin COMPONENT devel)

  set(CPACK_PACKAGE_NAME "amdrocm${ROCM_MAJOR_VERSION}-transferbench")
  set(CPACK_PACKAGE_VERSION "${VERSION_STRING}")
  set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc.")
  set(CPACK_PACKAGE_CONTACT "RCCL Team <gilbert.lee@amd.com>")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "TransferBench: benchmark simultaneous transfers between CPU/GPU/NIC")
  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")

  # Per-build release tag, threaded into DEB/RPM metadata AND the TGZ filename.
  # Prefer the explicit -D from build_packages_local.sh; fall back to the env
  # vars CPack itself reads (so direct cmake invocations still work).
  # Quote variable references throughout so the comparison is unambiguous
  # under any CMP0054 setting.
  set(_tb_pkg_release "")
  if(NOT "${TRANSFERBENCH_PACKAGE_RELEASE}" STREQUAL "")
    set(_tb_pkg_release "${TRANSFERBENCH_PACKAGE_RELEASE}")
  elseif(NOT "$ENV{CPACK_RPM_PACKAGE_RELEASE}" STREQUAL "")
    set(_tb_pkg_release "$ENV{CPACK_RPM_PACKAGE_RELEASE}")
  elseif(NOT "$ENV{CPACK_DEBIAN_PACKAGE_RELEASE}" STREQUAL "")
    set(_tb_pkg_release "$ENV{CPACK_DEBIAN_PACKAGE_RELEASE}")
  endif()

  # DEB
  set(CPACK_DEBIAN_PACKAGE_NAME       "${CPACK_PACKAGE_NAME}")
  set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS    "numactl, libnuma1")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_CONTACT}")
  if(NOT "${_tb_pkg_release}" STREQUAL "")
    set(CPACK_DEBIAN_PACKAGE_RELEASE "${_tb_pkg_release}")
  elseif(DEFINED ENV{CPACK_DEBIAN_PACKAGE_RELEASE})
    set(CPACK_DEBIAN_PACKAGE_RELEASE "$ENV{CPACK_DEBIAN_PACKAGE_RELEASE}")
  endif()
  # Canonical filename: <name>_<version>-<release>_<arch>.deb
  set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

  # RPM
  set(CPACK_RPM_PACKAGE_NAME    "${CPACK_PACKAGE_NAME}")
  set(CPACK_RPM_PACKAGE_LICENSE "MIT")
  set(CPACK_RPM_PACKAGE_REQUIRES "numactl")
  set(CPACK_RPM_PACKAGE_VENDOR  "${CPACK_PACKAGE_VENDOR}")
  if(NOT "${_tb_pkg_release}" STREQUAL "")
    set(CPACK_RPM_PACKAGE_RELEASE "${_tb_pkg_release}")
  elseif(DEFINED ENV{CPACK_RPM_PACKAGE_RELEASE})
    set(CPACK_RPM_PACKAGE_RELEASE "$ENV{CPACK_RPM_PACKAGE_RELEASE}")
  endif()
  # Canonical filename: <name>-<version>-<release>.<arch>.rpm
  set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")
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

  # Advisory install-time check for libhsa-runtime64.so.1. The package declares
  # no hard ROCm dep so it can install on TheRock-tarball systems where no ROCm
  # component is tracked by apt/dpkg; the postinst warns (never fails) when the
  # HSA runtime is not discoverable, so a missing runtime surfaces at install
  # time instead of as a dynamic-linker error on first invocation.
  set(_tb_postinst_src "${CMAKE_CURRENT_SOURCE_DIR}/packaging/postinst-check-hsa.sh")
  set(_tb_postinst_deb "${CMAKE_BINARY_DIR}/packaging/postinst")
  configure_file("${_tb_postinst_src}" "${_tb_postinst_deb}" COPYONLY)
  set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${_tb_postinst_deb}")
  set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${_tb_postinst_src}")

  # TGZ — embed release tag so successive runs do not collide on the same key.
  # CMake 3.13+ honors CPACK_ARCHIVE_FILE_NAME for archive generators, but
  # CMake 3.22 (Ubuntu 22.04) falls back to CPACK_PACKAGE_FILE_NAME for TGZ.
  # Set both to the same suffixed value. DEB/RPM are unaffected because they
  # use the explicit DEB-DEFAULT / RPM-DEFAULT canonical-naming tokens above.
  if("${_tb_pkg_release}" STREQUAL "")
    set(_tb_archive_name "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-Linux")
  else()
    set(_tb_archive_name "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${_tb_pkg_release}-Linux")
  endif()
  set(CPACK_ARCHIVE_FILE_NAME "${_tb_archive_name}")
  set(CPACK_PACKAGE_FILE_NAME "${_tb_archive_name}")

  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.20")
    configure_file(
      cmake/WriteGitVersion.cmake.in
      ${CMAKE_BINARY_DIR}/WriteGitVersion.cmake
      @ONLY
    )
    set(CPACK_PRE_BUILD_SCRIPTS "${CMAKE_BINARY_DIR}/WriteGitVersion.cmake")
  elseif(NOT TB_GIT_BRANCH STREQUAL "unknown")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/GIT_VERSION" "${TB_GIT_BRANCH}\n${TB_GIT_COMMIT}\n")
  endif()

  set(CPACK_GENERATOR "DEB;RPM;TGZ")
  include(CPack)
else()
  rocm_install(TARGETS TransferBench COMPONENT devel)
  rocm_setup_version(VERSION ${VERSION_STRING})

  # Package specific CPACK vars
  rocm_package_add_dependencies(DEPENDS "numactl")

  set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
  set(CPACK_RPM_PACKAGE_LICENSE "MIT")

  set(PACKAGE_NAME TB)
  set(LIBRARY_NAME TransferBench)

  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.20")
    configure_file(
      cmake/WriteGitVersion.cmake.in
      ${CMAKE_BINARY_DIR}/WriteGitVersion.cmake
      @ONLY
    )
    set(CPACK_PRE_BUILD_SCRIPTS "${CMAKE_BINARY_DIR}/WriteGitVersion.cmake")
  elseif(NOT TB_GIT_BRANCH STREQUAL "unknown")
    file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/GIT_VERSION" "${TB_GIT_BRANCH}\n${TB_GIT_COMMIT}\n")
  endif()

  rocm_create_package(
    NAME ${LIBRARY_NAME}
    DESCRIPTION "TransferBench package"
    MAINTAINER "RCCL Team <gilbert.lee@amd.com>"
  )
endif()
