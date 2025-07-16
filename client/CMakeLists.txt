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
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)


# Load CMake modules
#==================================================================================================
set(AMD_PROJECT_CMAKE_DIRECTORY ${AMD_PROJECT_BASE_DIRECTORY}/cmake)
set(AMD_PROJECT_CMAKE_MODULES_DIRECTORY ${AMD_PROJECT_BASE_DIRECTORY}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH "${AMD_PROJECT_CMAKE_MODULES_DIRECTORY}")


# CMake Toolchain file to define compilers and path to ROCm
#==================================================================================================
if (NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${AMD_PROJECT_CMAKE_DIRECTORY}/rocm_clang_toolchain.cmake")
    message(STATUS ">> CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
endif()

#
#   Build options
option(TRANSFERBENCH_CLIENT_USE_ENGINE_STATIC     "TransferBench client links with 'static' library (default)"  ON)
option(TRANSFERBENCH_CLIENT_USE_ENGINE_SHARED     "TransferBench client links with 'shared' library"            OFF)
option(TRANSFERBENCH_CLIENT_USE_HEADER_ONLY       "TransferBench uses 'header-only' interface"                  OFF)


#
include(${AMD_PROJECT_CMAKE_DIRECTORY}/build_utils.cmake)               # setup_default_compiler_flags
include(${AMD_PROJECT_CMAKE_MODULES_DIRECTORY}/Dependencies.cmake)      # rocm-cmake, rocm_local_targets

#
set (TRANSFERBENCH_TBENGINE_DIRECTORY ${AMD_PROJECT_BASE_DIRECTORY}/deps/tbengine)
set (TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY ${TRANSFERBENCH_TBENGINE_DIRECTORY}/include)
set (TRANSFERBENCH_CLIENT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set (TRANSFERBENCH_CLIENT_INCLUDE_DIRECTORY ${TRANSFERBENCH_CLIENT_DIRECTORY}/include)
set (TRANSFERBENCH_CLIENT_PRESETS_INCLUDE_DIRECTORY ${TRANSFERBENCH_CLIENT_INCLUDE_DIRECTORY}/Presets)
set (TRANSFERBENCH_CLIENT_SRC_DIRECTORY ${TRANSFERBENCH_CLIENT_DIRECTORY}/src)


#
#   Library names
set(AMD_PROJECT_HEADER_ONLY_LIBRARY_NAME "${AMD_PROJECT_LIBRARY_NAME}_header")
set(TRANSFERBENCH_INTERFACE_TARGET_NAME "${AMD_PROJECT_PACKAGE_NAME}_engine")
set(TRANSFERBENCH_INTERFACE_TARGET_NAME_ALIAS "${AMD_PROJECT_PACKAGE_NAME}::engine")
set(AMD_PROJECT_STATIC_LIBRARY_NAME "${AMD_PROJECT_LIBRARY_NAME}_static")
set(AMD_PROJECT_SHARED_LIBRARY_NAME "${AMD_PROJECT_LIBRARY_NAME}_shared")
set(AMD_PROJECT_OBJECT_LIBRARY_NAME "${AMD_PROJECT_LIBRARY_NAME}_object_library")
set(AMD_PROJECT_HIP_OBJECT_LIBRARY_NAME "${AMD_PROJECT_LIBRARY_NAME}_hip_object_library")
set(AMD_PROJECT_CLIENT_NAME "${AMD_PROJECT_NAME}")

#
#   Subproject build information
setup_build_version(TRANSFERBENCH_CLIENT_TARGET_VERSION TRANSFERBENCH_CLIENT_TARGET_VERSION_TEXT)
set_variable_in_parent(TRANSFERBENCH_CLIENT_TARGET_BINARY_VERSION ${TRANSFERBENCH_CLIENT_TARGET_VERSION})

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/)
configure_file(
    ${AMD_PROJECT_CMAKE_DIRECTORY}/tbclient_version.hpp.in
    ${CMAKE_BINARY_DIR}/include/tbclient_version.hpp
    @ONLY
)


#
#   Header/Source files
set(TRANSFERBENCH_CLIENT_SOURCES
    ${TRANSFERBENCH_CLIENT_SRC_DIRECTORY}/Client.cpp
)

set(TRANSFERBENCH_COMMON_INCLUDES
    ${TRANSFERBENCH_CLIENT_INCLUDE_DIRECTORY}
    ${TRANSFERBENCH_CLIENT_PRESETS_INCLUDE_DIRECTORY}
)


#
message(STATUS ">> Building TransferBench 'client' ...")
developer_status_message("DEVEL" "  >> TRANSFERBENCH_CLIENT_TARGET_VERSION: '${TRANSFERBENCH_CLIENT_TARGET_VERSION}' ")
add_executable(${AMD_PROJECT_CLIENT_NAME} ${TRANSFERBENCH_CLIENT_SOURCES})
set_target_properties(${AMD_PROJECT_CLIENT_NAME}
    PROPERTIES
        LINKER_LANGUAGE CXX
)

target_link_libraries(${AMD_PROJECT_CLIENT_NAME}
    PRIVATE
        ${TRANSFERBENCH_INTERFACE_TARGET_NAME_ALIAS}
)

target_include_directories(${AMD_PROJECT_CLIENT_NAME}
    PRIVATE
        ${TRANSFERBENCH_COMMON_INCLUDES}
        ${CMAKE_BINARY_DIR}/include/
        ${TRANSFERBENCH_TBENGINE_INCLUDE_DIRECTORY}
)
setup_default_compiler_flags(${AMD_PROJECT_CLIENT_NAME})


#
#   Linking the appropriate TransferBench library
#   Priority: Static > Shared > Header-Only
set(WAS_TB_LINKED OFF)
if(TB_ENGINE_STATIC)
    message(STATUS "  >> Client build with 'static' library: ${AMD_PROJECT_STATIC_LIBRARY_NAME} ...")
    target_link_libraries(${AMD_PROJECT_CLIENT_NAME}
        PRIVATE
            ${AMD_PROJECT_STATIC_LIBRARY_NAME}
    )
    set(WAS_TB_LINKED ON)

elseif(TB_ENGINE_SHARED)
    message(STATUS "  >> Client build with 'shared' library: ${AMD_PROJECT_SHARED_LIBRARY_NAME} ...")
    target_link_libraries(${AMD_PROJECT_CLIENT_NAME}
        PRIVATE
            ${AMD_PROJECT_SHARED_LIBRARY_NAME}
    )
    set(WAS_TB_LINKED ON)

elseif(TB_ENGINE_HEADER_ONLY)
    message(STATUS "  >> Client build with 'header-only': ${AMD_PROJECT_HEADER_ONLY_LIBRARY_NAME} ...")
    target_link_libraries(${AMD_PROJECT_CLIENT_NAME}
        PRIVATE
            ${AMD_PROJECT_HEADER_ONLY_LIBRARY_NAME}
    )

    target_compile_definitions(${AMD_PROJECT_CLIENT_NAME}
        PRIVATE
            TRANSFERBENCH_HEADER_IMPLEMENTATION_DETAILS
    )

endif()


# Install client executable
install(
    TARGETS ${AMD_PROJECT_CLIENT_NAME}
    RUNTIME DESTINATION bin
    COMPONENT devel
)

