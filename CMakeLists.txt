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
cmake_minimum_required(VERSION 3.25)

#
#   Find git and set the GIT_EXECUTABLE variable (good practice, used by submodules)
find_package(Git QUIET)
if(NOT GIT_FOUND)
    message(FATAL_ERROR ">> Git is required to build this project. 'git' not found! ")
endif()
set(CMAKE_GIT_EXECUTABLE ${GIT_EXECUTABLE} CACHE FILEPATH "Path to the Git executable.")

#
#   Flag to enable / disable verbose output.
set(CMAKE_VERBOSE_MAKEFILE ON)

#
#   Export compile commands for linters and auto-completers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#
#   C++ standard settings
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ Standard in use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INCLUDE_DIRECTORIES_BEFORE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#
#   Project specific directories
set(AMD_PROJECT_BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Base directory for this project" FORCE)
set(AMD_PROJECT_CMAKE_DIRECTORY ${AMD_PROJECT_BASE_DIRECTORY}/cmake)
set(AMD_PROJECT_CMAKE_MODULES_DIRECTORY ${AMD_PROJECT_BASE_DIRECTORY}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH "${AMD_PROJECT_CMAKE_MODULES_DIRECTORY}")

#
#   Build options
option(TRANSFERBENCH_ENGINE_HEADER_ONLY           "Make TB engine header-only available (interface target)"     OFF)
option(TRANSFERBENCH_ENGINE_STATIC                "Build TB static library"                                     ON)
option(TRANSFERBENCH_ENGINE_SHARED                "Build TB shared library"                                     OFF)
option(TRANSFERBENCH_CLIENT                       "Build TransferBench client"                                  ON)
option(TRANSFERBENCH_TREAT_WARNINGS_AS_ERRORS     "Treat default warnings as errors"                            ON)
option(TRANSFERBENCH_COMPRESS_DEBUG_INFO          "Compressed debug information"                                ON)
option(TRANSFERBENCH_LOCAL_GPU_TARGET_ONLY        "Build only for GPUs detected on this machine"                OFF)
option(TRANSFERBENCH_ENABLE_NIC_EXEC              "Enable RDMA NIC Executor in TransferBench"                   OFF)
option(TRANSFERBENCH_HARDENING_ENABLED            "Build the project with hardening flags"                      ON)
option(TRANSFERBENCH_LINKER_TRY_MOLD              "Give preference to 'Mold' linker (faster) if available"      ON)
option(TRANSFERBENCH_ENABLE_CPPCHECK_WARNINGS     "CppCheck static analysis warnings (for Developers)"          ON)


#
#   Setup build utils
include(${AMD_PROJECT_CMAKE_DIRECTORY}/build_utils.cmake)

#
#   CMake project info
set(AMD_PROJECT_NAME "TransferBench")
set(AMD_PROJECT_PACKAGE_NAME "transferbench")
set(AMD_PROJECT_LIBRARY_NAME "amd_tb")
set(AMD_PROJECT_DESCRIPTION "TransferBench utility")
set(AMD_PROJECT_AUTHOR_ORGANIZATION "AMD, Inc.")
set(AMD_PROJECT_GITHUB_REPO   "https://github.com/ROCm/TransferBench")
set(AMD_PROJECT_AUTHOR_DOMAIN "https://www.amd.com")
set(AMD_PROJECT_VERSION_MAJOR "")
set(AMD_PROJECT_VERSION_MINOR "")
set(AMD_PROJECT_VERSION_PATCH "")

#
#   Specify name of project to build, and validate requirements
setup_build_version(PROJECT_TARGET_VERSION PROJECT_TARGET_VERSION_TEXT)
set_variable_in_parent(PROJECT_TARGET_BINARY_VERSION ${PROJECT_TARGET_VERSION})
setup_cmake(AMD_PROJECT_NAME PROJECT_TARGET_BINARY_VERSION)

if(PROJECT_TARGET_VERSION AND ((AMD_PROJECT_VERSION_MAJOR STREQUAL "") OR (AMD_PROJECT_VERSION_MINOR STREQUAL "") OR (AMD_PROJECT_VERSION_PATCH STREQUAL "")))
    string(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)" "\\1" AMD_PROJECT_VERSION_MAJOR ${PROJECT_TARGET_VERSION})
    string(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)" "\\2" AMD_PROJECT_VERSION_MINOR ${PROJECT_TARGET_VERSION})
    string(REGEX REPLACE "([0-9]+)\\.([0-9]+)\\.([0-9]+)" "\\3" AMD_PROJECT_VERSION_PATCH ${PROJECT_TARGET_VERSION})
else()
    message(FATAL_ERROR ">> 'PROJECT_TARGET_VERSION' was not properly set!")
endif()
set(AMD_PROJECT_VERSION_FULL "${AMD_PROJECT_VERSION_MAJOR}.${AMD_PROJECT_VERSION_MINOR}.${AMD_PROJECT_VERSION_PATCH}")

#
#   Project information
project(${AMD_PROJECT_NAME}
    VERSION ${PROJECT_TARGET_VERSION_TEXT}
    LANGUAGES CXX
    DESCRIPTION ${AMD_PROJECT_DESCRIPTION}
    HOMEPAGE_URL ${AMD_PROJECT_GITHUB_REPO}
)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(TRANSFERBENCH_IS_TOP_LEVEL TRUE)
else()
    set(TRANSFERBENCH_IS_TOP_LEVEL FALSE)
endif()


#
#   Set the ROCm base path
setup_rocm_requirements()
add_build_definitions()

#
#   Build info
message(STATUS "[[ Building Project: " ${AMD_PROJECT_NAME} " v." ${PROJECT_TARGET_VERSION_TEXT} " { Build: " ${CMAKE_BUILD_TYPE} "} ]] ...")


#
# --- Auto Dependent Build options ---
include(CMakeDependentOption)
cmake_dependent_option(
    AUTO_BUILD_STATIC_FOR_TB_CLIENT
    "TRANSFERBENCH_ENGINE_STATIC build auto-enabled for TB client"
    ON
    "TRANSFERBENCH_CLIENT AND NOT (TRANSFERBENCH_ENGINE_HEADER_ONLY OR TRANSFERBENCH_ENGINE_SHARED OR TRANSFERBENCH_ENGINE_STATIC)"
    OFF
)

if(AUTO_BUILD_STATIC_FOR_TB_CLIENT)
    set(TRANSFERBENCH_ENGINE_STATIC ON CACHE BOOL "Build TB static library" FORCE)
endif()


cmake_dependent_option(
    AUTO_BUILD_TB_LIBRARY
    "Auto-enabled if any TB library option is set"
    ON
    "TRANSFERBENCH_ENGINE_HEADER_ONLY OR TRANSFERBENCH_ENGINE_SHARED OR TRANSFERBENCH_ENGINE_STATIC"
    OFF
)

if(AUTO_BUILD_TB_LIBRARY)
    set(TRANSFERBENCH_ENGINE_REQUIRED ON CACHE BOOL "TB engine is required" FORCE)
endif()


#   Validate conflicting build options
if(TRANSFERBENCH_CLIENT AND TRANSFERBENCH_ENGINE_HEADER_ONLY AND (TRANSFERBENCH_ENGINE_SHARED OR TRANSFERBENCH_ENGINE_STATIC))
    message(FATAL_ERROR ">> Conflicting build options: CLIENT cannot be built with: HEADER_ONLY and STATIC or SHARED! ")
endif()
# ---

if(TRANSFERBENCH_ENGINE_REQUIRED)
    add_subdirectory(deps/tbengine)
endif()

if(TRANSFERBENCH_CLIENT)
    add_subdirectory(client)
endif()


#
#  Packaging
if(TRANSFERBENCH_ENGINE_REQUIRED OR TRANSFERBENCH_CLIENT)
    ##rocm_install(TARGETS ${AMD_PROJECT_NAME} COMPONENT devel)
    rocm_package_add_dependencies(DEPENDS "numactl" "hsa-rocr")
    rocm_setup_version(VERSION ${VERSION_STRING})
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")

    rocm_create_package(
        NAME ${AMD_PROJECT_PACKAGE_NAME}
        DESCRIPTION "TransferBench package"
        MAINTAINER "RCCL Team <gilbert.lee@amd.com>"
    )
endif()

## End of CMakeLists.txt
