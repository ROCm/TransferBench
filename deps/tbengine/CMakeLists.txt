# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#

#
# --- Header-Only vs Regular Compiled Library ---
#
##  Header-Only Library:
#   A header-only library is a library that consists entirely of header files (.h or .hpp)
#       - It does not require separate compilation into binary files (.lib, .a, .so, etc)
#
#   - Simplicity and Ease of Distribution:
#       - No need for separate compilation or linking steps
#       - Just include the headers
#           - Great for header-only utilities or template-heavy libraries
#
#   - Heavy Use of Templates or Inline Functions:
#       - Templates must be defined in headers, so template libraries are often header-only (like, Eigen, Catch2)
#       - Inline functions benefit from this as well for potential compiler optimizations
#
#   - Small to Medium Size Libraries:
#       - Ideal when the codebase is not too large, avoiding long compile times
#
#   - Performance-Critical Components:
#       - Enables the compiler to inline aggressively across translation units
#
#   - Cross-Platform or Header-Only Dependencies:
#       - Avoids needing to build for multiple platforms or compilers
#
##  Regular Compiled Library:
#   A regular compiled library is a library that is compiled into binary files (.lib, .a, .so, etc)
#       - It requires separate compilation and linking steps
#
#   - Large Codebase / Long Compile Times
#       - Avoid recompiling all code that includes the library headers
#
#   - Improved Encapsulation
#       - Hides implementation details, reduces header bloat, and maintains a clean API
#       - Binary distribution keeps proprietary code hidden
#
#   - ABI Stability & Compatibility
#       - Enables decoupling user code from library internals
#       - Users don’t need to recompile their code when internals of the library change (if ABI remains stable)
#
#   - Reduced Binary Size:
#       - Prevents code bloat due to duplication in each translation unit
#
#   - Dynamic Loading / Plugin Systems
#       - Necessary if you want runtime dynamic linking (like, via dlopen)
#
#   - Separate Build and Test Pipelines
#       - Easier to build and test the library independently from the application
#
##  Hybrid Approach:
#   - Public API in headers, and compiled internals:
#       - Templates or inline functions stay in headers
#       - Logic-heavy or stable parts go into .so or .a files
#

#
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(${AMD_PROJECT_LIBRARY_NAME}
    VERSION ${PROJECT_TARGET_VERSION_TEXT}
    DESCRIPTION "TransferBench Engine Library"
    LANGUAGES CXX HIP
)

#
# --- Load CMake modules ---
set(AMD_PROJECT_CMAKE_DIRECTORY ${AMD_PROJECT_BASE_DIRECTORY}/cmake)
set(AMD_PROJECT_CMAKE_MODULES_DIRECTORY ${AMD_PROJECT_BASE_DIRECTORY}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH "${AMD_PROJECT_CMAKE_MODULES_DIRECTORY}")


#
# --- CMake Toolchain file to define compilers and path to ROCm ---
if (NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${AMD_PROJECT_CMAKE_DIRECTORY}/rocm_clang_toolchain.cmake")
    message(STATUS ">> CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
endif()

#
# --- Check if ROCM_BASE_PATH is defined, if not, exit with error ---
if(NOT ROCM_BASE_PATH)
    message(FATAL_ERROR ">> ROCM_BASE_PATH was not set ...")
endif()

#
include(CheckIncludeFiles)
include(CheckSymbolExists)
include(GNUInstallDirs)
include(${AMD_PROJECT_CMAKE_DIRECTORY}/build_utils.cmake)               # setup_default_compiler_flags
include(${AMD_PROJECT_CMAKE_MODULES_DIRECTORY}/Dependencies.cmake)      # rocm-cmake, rocm_local_targets

#
set (TRANSFERBENCH_TBENGINE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set (TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY ${TRANSFERBENCH_TBENGINE_DIRECTORY}/include)
set (TRANSFERBENCH_TBENGINE_SRC_DIRECTORY ${TRANSFERBENCH_TBENGINE_DIRECTORY}/src)
set (TRANSFERBENCH_INSTALL_INCLUDE_DIRECTORY ${ROCM_BASE_PATH}/include/${AMD_PROJECT_PACKAGE_NAME})
set (TRANSFERBENCH_INSTALL_LIBRARY_DIRECTORY ${ROCM_BASE_PATH}/lib)
set (TRANSFERBENCH_INSTALL_BIN_DIRECTORY ${ROCM_BASE_PATH}/bin)


#
#   Default GPU architectures to build
set(BUILD_TRANSFERBENCH_DEFAULT_GPUS_LIST
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
)


#
#   Build only for local GPU architecture
set(ROCMCHECKS_WARN_TOOLCHAIN_VAR OFF)
if (TB_LOCAL_GPU_TARGET_ONLY)
    message(STATUS "  >> Building only for local GPU target")
    if (COMMAND rocm_local_targets)
        rocm_local_targets(BUILD_TRANSFERBENCH_DEFAULT_GPUS_LIST)
    else()
        message(WARNING "  >> Unable to determine local GPU targets. Falling back to default GPUs.")
  endif()
endif()

#
#   Determine which GPU architectures to build for
set(TRANSFERBENCH_GPU_TARGETS "${BUILD_TRANSFERBENCH_DEFAULT_GPUS_LIST}" CACHE STRING "Target default GPUs if TRANSFERBENCH_GPU_TARGETS is not defined.")

#
#   Check if clang compiler can offload to GPU_TARGETS
if (COMMAND rocm_check_target_ids)
    message(STATUS ">> Checking for ROCm support for GPU targets: " "${TRANSFERBENCH_GPU_TARGETS}")
    rocm_check_target_ids(TRANSFERBENCH_SUPPORTED_GPUS TARGETS ${TRANSFERBENCH_GPU_TARGETS})
else()
    message(WARNING ">> Unable to check for supported GPU targets. Falling back to default GPUs.")
    set(TRANSFERBENCH_SUPPORTED_GPUS ${BUILD_TRANSFERBENCH_DEFAULT_GPUS_LIST})
endif()

set(TRANSFERBENCH_COMPILING_TARGETS "${TRANSFERBENCH_SUPPORTED_GPUS}" CACHE STRING "GPU targets to compile for.")
message(STATUS ">> Building for: ${TRANSFERBENCH_COMPILING_TARGETS}")
foreach(target ${TRANSFERBENCH_COMPILING_TARGETS})
    list(APPEND STATIC_LINK_FLAGS --offload-arch=${target})
endforeach()
list(JOIN STATIC_LINK_FLAGS " " FLAGS_STR)

#
#   NOTE: Reload rocm-cmake in order to update GPU_TARGETS
#   Reloading to use desired GPU_TARGETS instead of defaults
include(${AMD_PROJECT_CMAKE_MODULES_DIRECTORY}/Dependencies.cmake)

#
get_rocm_install_path(ROCM_PATH)


#
#   Set CMAKE flags
if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- HIP Package ---
#   Check for HIP
#
#   Add ROCM_BASE_PATH to CMake search paths for finding HIP / HSA
list(APPEND CMAKE_PREFIX_PATH
            ${ROCM_PATH}
            ${ROCM_PATH}/llvm
            ${ROCM_PATH}/hip
            ${ROCM_PATH}/hsa
            /opt/rocm
            /opt/rocm/llvm
            /opt/rocm/hip
            /opt/rocm/hsa
)

#
#   Check for HIP
find_package(hip REQUIRED CONFIG PATHS ${CMAKE_PREFIX_PATH})
developer_status_message("DEVEL" "  >> HIP Include Dirs: ${hip_INCLUDE_DIRS} ...")
developer_status_message("DEVEL" "  >> HIP Libraries: ${hip_LIBRARIES} ...")

#
#   Ensuring that CXX compiler meets expectations
if(NOT (("${CMAKE_CXX_COMPILER}" MATCHES ".*hipcc") OR ("${CMAKE_CXX_COMPILER}" MATCHES ".*clang\\+\\+")))
    message(FATAL_ERROR ">> On ROCm platform CMAKE_CXX_COMPILER must be 'hipcc' or 'HIP-aware Clang'.")
endif()

#
#   Check for Threads
find_package(Threads REQUIRED)
set(THREADS_PREFER_PTHREAD_FLAG ON)

#
#   Check for numa support
set(WAS_NUMA_FOUND OFF)
set(NUMA_LIBRARY_NAME "numa")
find_library(NUMA_LIBRARY ${NUMA_LIBRARY_NAME})
find_path(NUMA_INCLUDE_DIR numa.h)
if(NUMA_LIBRARY AND NUMA_INCLUDE_DIR)
    set(WAS_NUMA_FOUND ON)
    add_library(${NUMA_LIBRARY_NAME} SHARED IMPORTED)
    set_target_properties(${NUMA_LIBRARY_NAME}
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}"
                IMPORTED_LOCATION "${NUMA_LIBRARY}"
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}"
    )
endif()
developer_status_message("DEVEL" "NUMA_INCLUDE_DIR: ${NUMA_INCLUDE_DIR} ...")
developer_status_message("DEVEL" "NUMA_LIBRARY_NAME: ${NUMA_LIBRARY_NAME} ...")
developer_status_message("DEVEL" "NUMA_LIBRARY: ${NUMA_LIBRARY} ...")


#
#   Check for hsa support: 'libamdhip64.so' (libhsa-runtime-dev package)
#  ${ROCM_PATH}/include/hsa/hsa.h
find_path(HIP_ROOT_DIR
            NAMES
                "include/hip/hip_runtime.h"
            HINTS
                ${ROCM_PATH}
                /opt/rocm/
)
if(NOT HIP_ROOT_DIR)
    message(FATAL_ERROR ">> HIP_ROOT_DIR 'hip_runtime.h' not found. Ensure ROCm is properly set up ...")
endif()

set(HIP_INCLUDE_ROOT_DIR "${HIP_ROOT_DIR}/include")
set(HIP_LIBRARY_ROOT_DIR "${HIP_ROOT_DIR}/lib")
developer_status_message("DEVEL" "HIP_ROOT_DIR: ${HIP_ROOT_DIR} ...")
developer_status_message("DEVEL" "HIP_INCLUDE_ROOT_DIR: ${HIP_INCLUDE_ROOT_DIR} ...")
developer_status_message("DEVEL" "HIP_LIBRARY_ROOT_DIR: ${HIP_LIBRARY_ROOT_DIR} ...")

set(WAS_HSA_FOUND OFF)
set(HSA_LIBRARY_NAME "hsa-runtime64")
find_library(HSA_LIBRARY ${HSA_LIBRARY_NAME} PATHS ${HIP_LIBRARY_ROOT_DIR} ${ROCM_PATH})
find_path(HSA_INCLUDE_DIR "hsa/hsa.h" PATHS ${HIP_INCLUDE_ROOT_DIR} NO_DEFAULT_PATH)
if(HSA_LIBRARY AND HSA_INCLUDE_DIR)
    set(WAS_HSA_FOUND ON)
    add_library(${HSA_LIBRARY_NAME} SHARED IMPORTED)
    set_target_properties(${HSA_LIBRARY_NAME}
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${HSA_INCLUDE_DIR}"
                IMPORTED_LOCATION "${HSA_LIBRARY}"
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HSA_INCLUDE_DIR}"
    )
endif()
developer_status_message("DEVEL" "HSA_INCLUDE_DIR: ${HSA_INCLUDE_DIR} ...")
developer_status_message("DEVEL" "HSA_LIBRARY_NAME: ${HSA_LIBRARY_NAME} ...")
developer_status_message("DEVEL" "HSA_LIBRARY: ${HSA_LIBRARY} ...")


#
#   Check for hip support: 'libamdhip64.so' (libamdhip64-dev package)
#   HIP_LIBRARY will be set to: "hip::host;hip::device" by find_library())
#   ${ROCM_PATH}/include/hip/hip_ext.h
set(WAS_HIP_FOUND OFF)
set(HIP_LIBRARY_NAME "amdhip64")
find_library(HIP_LIBRARY ${HIP_LIBRARY_NAME} PATHS ${HIP_LIBRARY_ROOT_DIR} ${ROCM_PATH})
find_path(HIP_INCLUDE_DIR "hip/hip_ext.h" PATHS ${HIP_INCLUDE_ROOT_DIR} NO_DEFAULT_PATH)
if(NOT HIP_INCLUDE_DIR)
    message(FATAL_ERROR ">> HIP_INCLUDE_DIR 'hip_ext.h' not found. Ensure ROCm is properly set up ...")
endif()

if(HIP_LIBRARY AND HIP_INCLUDE_DIR)
    set(HIP_LIBRARY "${HIP_LIBRARY_ROOT_DIR}/libamdhip64.so")
    set(WAS_HIP_FOUND ON)
    add_library(${HIP_LIBRARY_NAME} SHARED IMPORTED)
    set_target_properties(${HIP_LIBRARY_NAME}
            PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
                IMPORTED_LOCATION "${HIP_LIBRARY}"
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
    )
endif()
developer_status_message("DEVEL" "HIP_INCLUDE_DIR: ${HIP_INCLUDE_DIR} ...")
developer_status_message("DEVEL" "HIP_LIBRARY_NAME: ${HIP_LIBRARY_NAME} ...")
developer_status_message("DEVEL" "HIP_LIBRARY: ${HIP_LIBRARY} ...")

#
#   Library/interface names

set(AMD_PROJECT_HEADER_ONLY_LIBRARY_NAME "${AMD_PROJECT_LIBRARY_NAME}_header")
set(TRANSFERBENCH_HEADER_NAME_PREFIX "tbengine")
set(TRANSFERBENCH_INTERFACE_TARGET_NAME "${AMD_PROJECT_PACKAGE_NAME}_engine")
set(TRANSFERBENCH_EXPORT_TARGET_NAME "${AMD_PROJECT_PACKAGE_NAME}_engineTargets")
set(TRANSFERBENCH_PACKAGE_CONFIG_FILE_NAME "${AMD_PROJECT_PACKAGE_NAME}_engineConfig")
set(TRANSFERBENCH_INTERFACE_NAMESPACE "${AMD_PROJECT_PACKAGE_NAME}")
set(TRANSFERBENCH_INTERFACE_CORE_NAME_ALIAS "${TRANSFERBENCH_INTERFACE_NAMESPACE}::core")
set(TRANSFERBENCH_INTERFACE_TARGET_NAME_ALIAS "${TRANSFERBENCH_INTERFACE_NAMESPACE}::engine")
set(AMD_PROJECT_STATIC_LIBRARY_NAME "${AMD_PROJECT_LIBRARY_NAME}_static")
set(AMD_PROJECT_SHARED_LIBRARY_NAME "${AMD_PROJECT_LIBRARY_NAME}_shared")
set(AMD_PROJECT_OBJECT_LIBRARY_NAME "${AMD_PROJECT_LIBRARY_NAME}_object_library")
set(AMD_PROJECT_HIP_OBJECT_LIBRARY_NAME "${AMD_PROJECT_LIBRARY_NAME}_hip_object_library")


#
#   Check for infiniband verbs support
set(WAS_IBVERBS_FOUND OFF)
if(DEFINED ENV{DISABLE_NIC_EXEC} AND "$ENV{DISABLE_NIC_EXEC}" STREQUAL "1")
    message(STATUS ">> Disabling 'NIC Executor' support. 'DISABLE_NIC_EXEC' was enabled ...")
elseif(NOT TB_ENABLE_NIC_EXEC)
    message(STATUS ">> For CMake builds, NIC executor requires explicit opt-in by setting CMake flag '-DTB_ENABLE_NIC_EXEC=1|ON' ...")
    message(STATUS ">> Disabling 'NIC Executor' support ...")
else()
    set(IBVERBS_LIBRARY_NAME "ibverbs")
    find_library(IBVERBS_LIBRARY ${IBVERBS_LIBRARY_NAME})
    find_path(IBVERBS_INCLUDE_DIR "infiniband/verbs.h")
    if(IBVERBS_LIBRARY AND IBVERBS_INCLUDE_DIR)
        set(WAS_IBVERBS_FOUND ON)
        add_library(${IBVERBS_LIBRARY_NAME} SHARED IMPORTED)
        set_target_properties(${IBVERBS_LIBRARY_NAME}
                PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "${IBVERBS_INCLUDE_DIR}"
                    IMPORTED_LOCATION "${IBVERBS_LIBRARY}"
                    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${IBVERBS_INCLUDE_DIR}"
        )
        message(STATUS ">> Building with 'NIC executor' support. Can set 'DISABLE_NIC_EXEC=1' to disable")
    else()
        if (NOT IBVERBS_LIBRARY)
            message(WARNING ">> 'IBVerbs' library not found ...")
        endif()
        if (NOT IBVERBS_INCLUDE_DIR)
            message(WARNING ">> 'infiniband/verbs.h' not found ...")
        endif()
        message(WARNING "Building without 'NIC executor' support. To use the TransferBench RDMA executor, \n"
                        " check if your system has NICs, the NIC drivers are installed, and 'libibverbs-dev' is installed")
    endif()
endif()


#
#   Check for DMA-BUF support (requires IBVERBS_FOUND)
set(WAS_DMABUF_SUPPORT_FOUND OFF)
if(WAS_IBVERBS_FOUND AND NOT TB_DISABLE_DMABUF)
    message(STATUS ">> Checking for DMA-BUF support ...")

    #   Check for ibv_reg_dmabuf_mr
    include(CheckSymbolExists)
    set(CMAKE_REQUIRED_INCLUDES ${IBVERBS_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES ${IBVERBS_LIBRARY})
    check_symbol_exists(ibv_reg_dmabuf_mr "infiniband/verbs.h" HAVE_IBV_DMABUF)

    #   Check for hsa_amd_portable_export_dmabuf
    set(CMAKE_REQUIRED_INCLUDES ${HSA_INCLUDE_DIR})
    set(CMAKE_REQUIRED_LIBRARIES ${HSA_LIBRARY})
    check_symbol_exists(hsa_amd_portable_export_dmabuf "hsa_ext_amd.h" HAVE_ROCM_DMABUF)

    #   Enable DMA-BUF only if both APIs are available
    if(HAVE_IBV_DMABUF AND HAVE_ROCM_DMABUF)
        set(WAS_DMABUF_SUPPORT_FOUND ON)
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
elseif(NOT TB_DISABLE_DMABUF)
    message(WARNING "DMA-BUF support requires TB_ENABLE_NIC_EXEC=ON")
endif()


#
#   Check for MPI support
set(WAS_MPI_COMM_FOUND OFF)
set(MPI_CXX_TARGET_NAME "")
set(MPI_LIBRARY_NAME "mpi")
set(MPI_PATH "" CACHE PATH "Path to MPI installation (takes priority over system MPI)")
if(NOT TB_ENABLE_MPI_COMM)
    message(STATUS ">> For CMake builds, MPI Communicator requires explicit opt-in by setting CMake flag '-DTB_ENABLE_MPI_COMM=ON' ...")
    message(STATUS ">> Disabling MPI Communicator support ...")
else()
    #   First check user-specified MPI_PATH (similar to Makefile)
    if(MPI_PATH AND EXISTS "${MPI_PATH}/include/mpi.h")
        find_library(MPI_LIBRARY NAMES ${MPI_LIBRARY_NAME} PATHS ${MPI_PATH}/lib NO_DEFAULT_PATH)
        if(MPI_LIBRARY)
            set(WAS_MPI_COMM_FOUND ON)
            set(MPI_INCLUDE_DIRS "${MPI_PATH}/include")
            set(MPI_LIBRARIES "${MPI_PATH}/lib")
            message(STATUS ">> Building with MPI Communicator support: (MPI_PATH: '${MPI_PATH}') ...")
        else()
            message(WARNING ">> Found 'mpi.h' at '${MPI_PATH}/include' but could not find MPI library at '${MPI_PATH}/lib' ...")
        endif()
    else()
        #   Fall back to find_package
        if(MPI_PATH)
            message(STATUS "Unable to find 'mpi.h' at '${MPI_PATH}/include', trying find_package() ...")
        endif()

        set(MPI_CXX_SKIP_MPICXX TRUE)
        set(MPI_GUESS_LIBRARY_NAME TRUE)
        find_package(MPI REQUIRED COMPONENTS CXX)
        if(MPI_FOUND)
            set(WAS_MPI_COMM_FOUND ON)
            if(MPI_CXX_FOUND)
                set(MPI_CXX_TARGET_NAME "MPI::MPI_CXX")
            endif()

            add_library(${MPI_LIBRARY_NAME} SHARED IMPORTED)
            set_target_properties(${MPI_LIBRARY_NAME}
                    PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_DIRS}"
                        IMPORTED_LOCATION "${MPI_CXX_LIBRARIES}"
                        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_DIRS}"
            )

            message(STATUS ">> Building with MPI Communicator support (found via find_package()) ...")
            message(STATUS "  >> MPI version: '${MPI_CXX_VERSION}' ...")
            message(STATUS "  >> Using MPI include path: '${MPI_CXX_INCLUDE_DIRS}' ...")
            message(STATUS "  >> Using MPI library: '${MPI_CXX_LIBRARIES}' ...")
        else()
            message(WARNING "  >> MPI not found. Please specify appropriate MPI_PATH or install MPI libraries (e.g., 'OpenMPI' or 'MPICH') ...")
        endif()
    endif()

    developer_status_message("DEVEL" "  >> MPI_CXX_TARGET_NAME: ${MPI_CXX_TARGET_NAME} ...")
    developer_status_message("DEVEL" "  >> MPI_CXX_COMPILE_OPTIONS: ${MPI_CXX_COMPILE_OPTIONS} ...")
    developer_status_message("DEVEL" "  >> MPI_CXX_INCLUDE_PATH: ${MPI_CXX_INCLUDE_PATH} ...")
    developer_status_message("DEVEL" "  >> MPI_CXX_LINK_FLAGS: ${MPI_CXX_LINK_FLAGS} ...")
    developer_status_message("DEVEL" "  >> MPI_CXX_LIB_NAMES: ${MPI_CXX_LIB_NAMES} ...")
    developer_status_message("DEVEL" "  >> MPI_CXX_COMPILER_INCLUDE_DIRS: ${MPI_CXX_COMPILER_INCLUDE_DIRS} ...")
    developer_status_message("DEVEL" "  >> MPI_CXX_INCLUDE_DIRS: ${MPI_CXX_INCLUDE_DIRS} ...")
    developer_status_message("DEVEL" "  >> MPI_CXX_LIBRARIES: ${MPI_CXX_LIBRARIES} ...")
endif()


# --- Get TB commit and branch ---
# That's useful for tracking which version of the code was used to build the library
if(DEFINED TRANSFERBENCH_COMMIT_HASH_LONG AND DEFINED TRANSFERBENCH_COMMIT_BRANCH)
    set(GIT_COMMIT_HASH_LONG "${TRANSFERBENCH_COMMIT_HASH_LONG}")
    set(GIT_BRANCH "${TRANSFERBENCH_COMMIT_BRANCH}")
else()
    # Get info about the current branch
    execute_process(
        COMMAND git rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RESULT_BRANCH
        ERROR_QUIET
    )

    # Get hash log info for the current branch
    execute_process(
        COMMAND git log -1 --format=%H
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH_LONG
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RESULT_HASH_LONG
        ERROR_QUIET
    )
endif()

if(GIT_COMMIT_HASH_LONG STREQUAL "" OR GIT_BRANCH STREQUAL "")
    message(WARNING "[[ No commit hash/branch were found. ]]")
else()
    set(TRANSFERBENCH_HEADER_VERSION ${PROJECT_TARGET_VERSION_TEXT})
    developer_status_message("DEVEL" ">> Setting TransferBench commit/branch info in '${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}/TransferBench.hpp' ...")
    developer_status_message("DEVEL" "  >> GIT_BRANCH=\"${GIT_BRANCH}\"")
    developer_status_message("DEVEL" "  >> GIT_COMMIT_HASH_LONG=\"${GIT_COMMIT_HASH_LONG}\"")
    developer_status_message("DEVEL" "  >> TRANSFERBENCH_HEADER_VERSION=\"${TRANSFERBENCH_HEADER_VERSION}\"")

    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/)
    configure_file(
        ${AMD_PROJECT_CMAKE_DIRECTORY}/${TRANSFERBENCH_HEADER_NAME_PREFIX}_version.hpp.in
        ${CMAKE_BINARY_DIR}/include/${TRANSFERBENCH_HEADER_NAME_PREFIX}_version.hpp
        @ONLY
    )
endif()
# --- End of Get TB commit and branch ---

#
#   Header/Source files
set(TRANSFERBENCH_ENGINE_HEADER_SOURCES
    ${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}/TransferBench.hpp
)

set(TRANSFERBENCH_ENGINE_HEADER_IMPL_SOURCES
    ${TRANSFERBENCH_TBENGINE_SRC_DIRECTORY}/TransferBench.cpp
)

set(TRANSFERBENCH_ENGINE_ALL_SOURCES
    ${TRANSFERBENCH_ENGINE_HEADER_SOURCES}
    ${TRANSFERBENCH_ENGINE_HEADER_IMPL_SOURCES}
)


#
# --- Object Libraries, and avoid recompilation ---
#   Common public interface target
add_library(${TRANSFERBENCH_INTERFACE_TARGET_NAME} INTERFACE)
add_library(${TRANSFERBENCH_INTERFACE_TARGET_NAME_ALIAS} ALIAS ${TRANSFERBENCH_INTERFACE_TARGET_NAME})
target_include_directories(${TRANSFERBENCH_INTERFACE_TARGET_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}>
        $<INSTALL_INTERFACE:include>
)

#
#   For dynamic linking: HIP object library (for use in other targets)
add_library(${AMD_PROJECT_HIP_OBJECT_LIBRARY_NAME}
    OBJECT
        ${TRANSFERBENCH_ENGINE_HEADER_IMPL_SOURCES}
)

set_target_properties(${AMD_PROJECT_HIP_OBJECT_LIBRARY_NAME}
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${AMD_PROJECT_HIP_OBJECT_LIBRARY_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/>
        $<BUILD_INTERFACE:${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}>
        $<INSTALL_INTERFACE:include>
)
setup_default_compiler_flags(${AMD_PROJECT_HIP_OBJECT_LIBRARY_NAME})


#
#   For static linking: Standard object library (for use in other targets)
add_library(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
    OBJECT
        ${TRANSFERBENCH_ENGINE_HEADER_IMPL_SOURCES}
)

set_target_properties(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${NUMA_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${HIP_INCLUDE_ROOT_DIR}>
        $<BUILD_INTERFACE:${HIP_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${HSA_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/>
        $<BUILD_INTERFACE:${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}>
        $<INSTALL_INTERFACE:include>
)
setup_default_compiler_flags(${AMD_PROJECT_OBJECT_LIBRARY_NAME})

target_link_libraries(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
    PUBLIC
        ${NUMA_LIBRARY}
        ${HIP_LIBRARY}
        ${HSA_LIBRARY}
        hip::host
        Threads::Threads
        dl
)


if(WAS_IBVERBS_FOUND)
    target_include_directories(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
        PRIVATE
            $<BUILD_INTERFACE:${IBVERBS_INCLUDE_DIR}>
    )

    target_link_libraries(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
        PRIVATE
            ${IBVERBS_LIBRARY}
    )
    target_compile_definitions(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
        PRIVATE
            NIC_EXEC_ENABLED
    )
endif()

if(WAS_MPI_COMM_FOUND)
    if(MPI_CXX_FOUND AND TARGET ${MPI_CXX_TARGET_NAME})
        #   Found via find_package
        target_include_directories(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
            PRIVATE
                $<BUILD_INTERFACE:${MPI_CXX_INCLUDE_DIRS}>
        )
        target_link_libraries(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
            PRIVATE
                ${MPI_CXX_TARGET_NAME}
        )
    else()
        #   Found via MPI_PATH fallback
        target_include_directories(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
            PRIVATE
                $<BUILD_INTERFACE:${MPI_INCLUDE_DIRS}>
        )
        target_link_libraries(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
            PRIVATE
                ${MPI_LIBRARIES}
        )
    endif()
    target_compile_definitions(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
        PRIVATE
            MPI_COMM_ENABLED
    )
endif()

if(WAS_DMABUF_SUPPORT_FOUND)
    target_compile_definitions(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
        PRIVATE
            HAVE_DMABUF_SUPPORT
    )
endif()

if (HAVE_PARALLEL_JOBS)
    target_compile_options(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
        PRIVATE
            "-parallel-jobs=12"
    )
endif()

set_target_properties(${AMD_PROJECT_OBJECT_LIBRARY_NAME}
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        EXCLUDE_FROM_ALL ON
        EXCLUDE_FROM_DEFAULT_BUILD ON
)
# ---

#
# --- Different build types ---
if(TB_ENGINE_SHARED)
    message(STATUS ">> Building TransferBench 'shared' library ...")
    add_library(${AMD_PROJECT_SHARED_LIBRARY_NAME} SHARED
        $<TARGET_OBJECTS:${AMD_PROJECT_HIP_OBJECT_LIBRARY_NAME}>
    )
    add_library(${TRANSFERBENCH_INTERFACE_CORE_NAME_ALIAS} ALIAS ${AMD_PROJECT_SHARED_LIBRARY_NAME})

    developer_status_message("DEVEL" "  >> PROJECT_TARGET_BINARY_VERSION: '${PROJECT_TARGET_BINARY_VERSION}' ")
    developer_status_message("DEVEL" "  >> PROJECT_TARGET_VERSION_TEXT: '${PROJECT_TARGET_VERSION_TEXT}' ")
    developer_status_message("DEVEL" "  >> PROJECT_TARGET_VERSION: '${PROJECT_TARGET_VERSION}' ")
    developer_status_message("DEVEL" "  >> PROJECT_MAJOR.MINOR_VERSION: '${AMD_PROJECT_VERSION_MAJOR}'.'${AMD_PROJECT_VERSION_MINOR}' ")
    set_target_properties(${AMD_PROJECT_SHARED_LIBRARY_NAME}
        PROPERTIES
            OUTPUT_NAME ${AMD_PROJECT_LIBRARY_NAME}
            VERSION ${PROJECT_TARGET_VERSION}
            SOVERSION ${AMD_PROJECT_VERSION_MAJOR}
            LINKER_LANGUAGE CXX
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(${AMD_PROJECT_SHARED_LIBRARY_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}>
            $<INSTALL_INTERFACE:include/${AMD_PROJECT_PACKAGE_NAME}>
    )

    target_link_libraries(${AMD_PROJECT_SHARED_LIBRARY_NAME}
        PUBLIC
            ${NUMA_LIBRARY}
            ${HSA_LIBRARY}
            Threads::Threads
            dl

        PRIVATE
            hip::device
    )

    target_compile_definitions(${AMD_PROJECT_SHARED_LIBRARY_NAME}
        PUBLIC
            TRANSFERBENCH_SHARED
    )

    #   Shared library specific compile options
    setup_default_compiler_flags(${AMD_PROJECT_SHARED_LIBRARY_NAME})
    add_common_flag("-fgpu-rdc" ${AMD_PROJECT_SHARED_LIBRARY_NAME})

    #   Install shared library
    install(
        TARGETS ${AMD_PROJECT_SHARED_LIBRARY_NAME}
        EXPORT ${TRANSFERBENCH_EXPORT_TARGET_NAME}
        RUNTIME DESTINATION ${TRANSFERBENCH_INSTALL_BIN_DIRECTORY}
        LIBRARY DESTINATION ${TRANSFERBENCH_INSTALL_LIBRARY_DIRECTORY}
        ARCHIVE DESTINATION ${TRANSFERBENCH_INSTALL_LIBRARY_DIRECTORY}
        INCLUDES DESTINATION ${TRANSFERBENCH_INSTALL_INCLUDE_DIRECTORY}
        COMPONENT devel
    )

    install(
        DIRECTORY ${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}/
        DESTINATION ${TRANSFERBENCH_INSTALL_INCLUDE_DIRECTORY}
        COMPONENT devel
        FILES_MATCHING
            PATTERN "*.hpp"
    )

    #   Install generated version header
    install(
        FILES ${CMAKE_BINARY_DIR}/include/${TRANSFERBENCH_HEADER_NAME_PREFIX}_version.hpp
        DESTINATION ${TRANSFERBENCH_INSTALL_INCLUDE_DIRECTORY}
        COMPONENT devel
    )

endif()

if(TB_ENGINE_STATIC)
    message(STATUS ">> Building TransferBench 'static' library ...")
    add_library(${AMD_PROJECT_STATIC_LIBRARY_NAME} STATIC
        $<TARGET_OBJECTS:${AMD_PROJECT_OBJECT_LIBRARY_NAME}>
    )
    add_library(${TRANSFERBENCH_INTERFACE_CORE_NAME_ALIAS} ALIAS ${AMD_PROJECT_STATIC_LIBRARY_NAME})

    set_target_properties(${AMD_PROJECT_STATIC_LIBRARY_NAME}
        PROPERTIES
            OUTPUT_NAME ${AMD_PROJECT_LIBRARY_NAME}
            POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(${AMD_PROJECT_STATIC_LIBRARY_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}>
            $<INSTALL_INTERFACE:include/${AMD_PROJECT_PACKAGE_NAME}>
    )

    target_link_libraries(${AMD_PROJECT_STATIC_LIBRARY_NAME}
        PUBLIC
            ${NUMA_LIBRARY}
            ${HSA_LIBRARY}
            Threads::Threads
            hip::host
            dl
    )

    target_compile_definitions(${AMD_PROJECT_STATIC_LIBRARY_NAME}
        PUBLIC
            TRANSFERBENCH_STATIC
    )

    #   Static library specific compile options
    setup_default_compiler_flags(${AMD_PROJECT_STATIC_LIBRARY_NAME})
    add_common_flag("-fgpu-rdc" ${AMD_PROJECT_STATIC_LIBRARY_NAME})

    #   Install static library
    install(
        TARGETS ${AMD_PROJECT_STATIC_LIBRARY_NAME}
        EXPORT ${TRANSFERBENCH_EXPORT_TARGET_NAME}
        RUNTIME DESTINATION ${TRANSFERBENCH_INSTALL_BIN_DIRECTORY}
        LIBRARY DESTINATION ${TRANSFERBENCH_INSTALL_LIBRARY_DIRECTORY}
        ARCHIVE DESTINATION ${TRANSFERBENCH_INSTALL_LIBRARY_DIRECTORY}
        INCLUDES DESTINATION ${TRANSFERBENCH_INSTALL_INCLUDE_DIRECTORY}
        COMPONENT devel
    )

    install(
        DIRECTORY ${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}/
        DESTINATION ${TRANSFERBENCH_INSTALL_INCLUDE_DIRECTORY}
        COMPONENT devel
        FILES_MATCHING
            PATTERN "*.hpp"
    )

    #   Install generated version header
    install(
        FILES ${CMAKE_BINARY_DIR}/include/${TRANSFERBENCH_HEADER_NAME_PREFIX}_version.hpp
        DESTINATION ${TRANSFERBENCH_INSTALL_INCLUDE_DIRECTORY}
        COMPONENT devel
    )

endif()

if(TB_ENGINE_HEADER_ONLY)
    message(STATUS ">> Building TransferBench 'header-only' library ...")
    add_library(${AMD_PROJECT_HEADER_ONLY_LIBRARY_NAME} INTERFACE)
    add_library(${TRANSFERBENCH_INTERFACE_CORE_NAME_ALIAS} ALIAS ${AMD_PROJECT_HEADER_ONLY_LIBRARY_NAME})

    set_target_properties(${AMD_PROJECT_HEADER_ONLY_LIBRARY_NAME}
        PROPERTIES
            INTERFACE_CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_include_directories(${AMD_PROJECT_HEADER_ONLY_LIBRARY_NAME}
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}>
            $<INSTALL_INTERFACE:include/${AMD_PROJECT_PACKAGE_NAME}>
    )

    target_link_libraries(${AMD_PROJECT_HEADER_ONLY_LIBRARY_NAME}
        INTERFACE
            ${AMD_PROJECT_HIP_OBJECT_LIBRARY_NAME}
            ${NUMA_LIBRARY}
            ${HIP_LIBRARY}
            ${HSA_LIBRARY}
            hip::device
            Threads::Threads
            dl
    )

    #   Install header-only library headers
    install(
        DIRECTORY ${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}/
        DESTINATION ${TRANSFERBENCH_INSTALL_INCLUDE_DIRECTORY}
        COMPONENT devel
        FILES_MATCHING
            PATTERN "*.hpp"
    )

    #   Install generated version header
    install(
        FILES ${CMAKE_BINARY_DIR}/include/${TRANSFERBENCH_HEADER_NAME_PREFIX}_version.hpp
        DESTINATION ${TRANSFERBENCH_INSTALL_INCLUDE_DIRECTORY}
        COMPONENT devel
    )

    #   Install header-only target
    install(
        TARGETS ${AMD_PROJECT_HEADER_ONLY_LIBRARY_NAME}
        EXPORT ${TRANSFERBENCH_EXPORT_TARGET_NAME}
        COMPONENT devel
    )

endif()

#
#   Common install. Export targets and package config for all build types
if(TB_ENGINE_HEADER_ONLY OR TB_ENGINE_STATIC OR TB_ENGINE_SHARED)
    #   Install export targets
    install(
        EXPORT ${TRANSFERBENCH_EXPORT_TARGET_NAME}
        FILE ${TRANSFERBENCH_EXPORT_TARGET_NAME}.cmake
        NAMESPACE "${TRANSFERBENCH_INTERFACE_NAMESPACE}::"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${AMD_PROJECT_PACKAGE_NAME}
        COMPONENT devel
    )

    #   Generate and install package config files
    include(CMakePackageConfigHelpers)

    #   Generate package version file
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/${TRANSFERBENCH_PACKAGE_CONFIG_FILE_NAME}Version.cmake
        VERSION ${PROJECT_TARGET_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    #   Generate package config file
    configure_package_config_file(
        ${AMD_PROJECT_CMAKE_DIRECTORY}/${TRANSFERBENCH_HEADER_NAME_PREFIX}_config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/${TRANSFERBENCH_PACKAGE_CONFIG_FILE_NAME}.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${AMD_PROJECT_PACKAGE_NAME}
    )

    #   Install package config files
    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/${TRANSFERBENCH_PACKAGE_CONFIG_FILE_NAME}.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/${TRANSFERBENCH_PACKAGE_CONFIG_FILE_NAME}Version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${AMD_PROJECT_PACKAGE_NAME}
        COMPONENT devel
    )

endif()


## End of CMakeLists.txt

