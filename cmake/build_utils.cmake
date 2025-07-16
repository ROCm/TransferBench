# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#   Required Includes
include(FetchContent)

#
#   Note: All functions definitions here
function(get_rocm_install_path rocm_install_base_path)
    if(NOT DEFINED ROCM_INSTALL_PATH_FOR_BUILD)
        message(STATUS ">> Checking ROCm install path settings...")
        set(TMP_ROCM_INSTALL_PATH "")
        if(DEFINED ENV{ROCM_PATH} OR DEFINED ROCM_PATH)
            if(DEFINED ENV{ROCM_PATH})
                message(STATUS "  >> Environment variable ROCM_PATH: '$ENV{ROCM_PATH}'")
                set(TMP_ROCM_INSTALL_PATH "$ENV{ROCM_PATH}")
            endif()
            if(DEFINED ROCM_PATH)
                message(STATUS "  >> CMake variable ROCM_PATH: '${ROCM_PATH}'")
                set(TMP_ROCM_INSTALL_PATH "${ROCM_PATH}")
            endif()
        elseif(DEFINED ENV{ROCM_INSTALL_PATH} OR DEFINED ROCM_INSTALL_PATH)
            if(DEFINED ENV{ROCM_INSTALL_PATH})
                message(STATUS "  >> Environment variable ROCM_INSTALL_PATH: '$ENV{ROCM_INSTALL_PATH}'")
                set(TMP_ROCM_INSTALL_PATH "$ENV{ROCM_PATH}")
            endif()
            if(DEFINED ROCM_INSTALL_PATH)
                message(STATUS "  >> CMake variable ROCM_INSTALL_PATH: '${ROCM_INSTALL_PATH}'")
            endif()
        else()
            set(TMP_ROCM_INSTALL_PATH "/opt/rocm")
            message(STATUS "  >> Using default ROCm install path: '${TMP_ROCM_INSTALL_PATH}'")
        endif()
        set(ROCM_INSTALL_PATH_FOR_BUILD "${TMP_ROCM_INSTALL_PATH}" CACHE STRING "ROCm install directory for build" FORCE)
        set(ROCM_INSTALL_PATH_FOR_BUILD "${ROCM_INSTALL_PATH_FOR_BUILD}" PARENT_SCOPE)
    else()
        set(${rocm_install_base_path} "${ROCM_INSTALL_PATH_FOR_BUILD}" PARENT_SCOPE)
    endif()
endfunction()

function(setup_build_version version_num version_text)
    set(TARGET_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
    if(NOT EXISTS "${TARGET_VERSION_FILE}")
        message(FATAL_ERROR "  >> VERSION file not found at: '${TARGET_VERSION_FILE}' ...")
    endif()

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${TARGET_VERSION_FILE})
    file(READ "${TARGET_VERSION_FILE}" file_version)
    string(STRIP ${file_version} file_version)
    string(REPLACE ".wip" "" file_version_text ${file_version})
    string(REPLACE ".WIP" "" file_version_text ${file_version})
    set(${version_num} ${file_version} PARENT_SCOPE)
    set(${version_text} ${file_version_text} PARENT_SCOPE)
endfunction()

function(setup_rocm_requirements)
    message(STATUS ">> Checking ROCm environment...")
    get_rocm_install_path(ROCM_BASE_PATH)

    #
    #find_package(ROCM 0.8 REQUIRED PATHS ${ROCM_BASE_PATH})
    find_package(ROCmCMakeBuildTools REQUIRED PATHS ${ROCM_BASE_PATH})
    find_package(ROCM REQUIRED PATHS ${ROCM_BASE_PATH})
    find_package(HSA-RUNTIME64 REQUIRED PATHS ${ROCM_BASE_PATH})

    set(ROCM_WARN_TOOLCHAIN_VAR OFF CACHE BOOL "ROCM_WARN_TOOLCHAIN warnings disabled: 'OFF'")
    set(ROCMCHECKS_WARN_TOOLCHAIN_VAR OFF CACHE BOOL "ROCMCHECKS_WARN_TOOLCHAIN_VAR warnings disabled: 'OFF'")
endfunction()

function(add_include_from_library target_name library_name)
    get_target_property(LIBRARY_INCLUDE_DIRECTORIES ${library_name} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(${target_name} PRIVATE ${LIBRARY_INCLUDE_DIRECTORIES})
endfunction()

function(add_source_definitions target_name definition_text)
    set_property(SOURCE ${target_name} APPEND PROPERTY COMPILE_DEFINITIONS "${definition_text}")
endfunction()

function(build_transferbench_engine)
    include(ROCMInstallTargets)
    include(ROCMCreatePackage)
endfunction()

function(has_build_debug_mode debug_mode_result)
    if(NOT DEFINED IS_BUILD_DEBUG_MSG_MODE_ENABLED)
        if(AMD_APP_DEBUG_BUILD_INFO OR
            (DEFINED ENV{AMD_APP_DEBUG_BUILD_INFO} AND
            ("$ENV{AMD_APP_DEBUG_BUILD_INFO}" STREQUAL "ON") OR
            ("$ENV{AMD_APP_DEBUG_BUILD_INFO}" STREQUAL "1")) OR
            (DEFINED BUILD_DEBUG_MSG_MODE AND (BUILD_DEBUG_MSG_MODE STREQUAL "ON")))
            set(IS_BUILD_DEBUG_MSG_MODE_ENABLED BOOL TRUE)
            set(IS_BUILD_DEBUG_MSG_MODE_ENABLED BOOL TRUE PARENT_SCOPE)
            set(${debug_mode_result} BOOL TRUE PARENT_SCOPE)
        else()
            set(IS_BUILD_DEBUG_MSG_MODE_ENABLED BOOL FALSE)
            set(IS_BUILD_DEBUG_MSG_MODE_ENABLED BOOL FALSE PARENT_SCOPE)
            set(${debug_mode_result} BOOL FALSE PARENT_SCOPE)
        endif()
    else()
        if(IS_BUILD_DEBUG_MSG_MODE_ENABLED)
            set(${debug_mode_result} BOOL TRUE PARENT_SCOPE)
        else()
            set(${debug_mode_result} BOOL FALSE PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(get_target target_name target_type)
    get_target_property(IMPORTED_TARGET ${target_name} IMPORTED)
    if(IMPORTED_TARGET)
        set(${target_type} INTERFACE PARENT_SCOPE)
    else()
        set(${target_type} PRIVATE PARENT_SCOPE)
    endif()
endfunction()

function(add_c_flag)
    if (ARGC EQUAL 1)
        add_compile_options($<$<COMPILE_LANGUAGE:C>:${ARGV0}>)
    elseif(ARGC EQUAL 2)
        get_target(${ARGV1} TYPE)
        target_compile_options(${ARGV1} ${TYPE} $<$<COMPILE_LANGUAGE:C>:${ARGV0}>)
    endif()
endfunction()

function(add_cxx_flag)
    if (ARGC EQUAL 1)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${ARGV0}>)
    elseif(ARGC EQUAL 2)
        get_target(${ARGV1} TYPE)
        target_compile_options(${ARGV1} ${TYPE} $<$<COMPILE_LANGUAGE:CXX>:${ARGV0}>)
    endif()
endfunction()

function(add_linker_flag)
    if (ARGC EQUAL 1)
        add_link_options(${ARGV0})
    elseif(ARGC EQUAL 2)
        get_target(${ARGV1} TYPE)
        target_link_options(${ARGV1} ${TYPE} ${ARGV0})
    endif()
endfunction()

function(add_c_cxx_flag)
    add_c_flag(${ARGV0} ${ARGV1})
    add_cxx_flag(${ARGV0} ${ARGV1})
endfunction()

function(add_common_flag)
    add_c_flag(${ARGV0} ${ARGV1})
    add_cxx_flag(${ARGV0} ${ARGV1})
endfunction()

function(add_cppcheck target_name)
    if(NOT TRANSFERBENCH_ENABLE_CPPCHECK_WARNINGS)
        return()
    endif()

    find_program(CPPCHECK_EXECUTABLE NAMES cppcheck REQUIRED)
    if(NOT CPPCHECK_EXECUTABLE)
        message(WARNING ">> Skipping 'cppcheck' target for: ${target_name}. Could not find 'Cppcheck' ...")
        return()
    endif()

    set(CPPCHECK_CONFIG_FILE "cppcheck_static_supp.config")
    set(CPPCHECK_REPORT_FILE "cppcheck_report.txt")
    set(TARGET_BUILD_DIRECTORY $<TARGET_FILE_DIR:${target_name}>)
    set(CPPCHECK_OPTION_LIST
        --enable=all
        --quiet
        --std=c++${CMAKE_CXX_STANDARD}
        --inline-suppr
        --check-level=exhaustive
        --error-exitcode=10
        --suppressions-list=${CMAKE_SOURCE_DIR}/dist/${CPPCHECK_CONFIG_FILE}
        --checkers-report=${TARGET_BUILD_DIRECTORY}/${CPPCHECK_REPORT_FILE}
    )
    set_target_properties(${target_name}
        PROPERTIES
            CXX_CPPCHECK "${CPPCHECK_EXECUTABLE};${CPPCHECK_OPTION_LIST}"
    )

    has_build_debug_mode(HAS_DEBUG_MODE_ENABLED)
    if(HAS_DEBUG_MODE_ENABLED)
        developer_status_message("DEVEL" ">> CppCheck settings for: '${target_name}' ...")
        developer_status_message("DEVEL" "  >> Target Build Directory: '${TARGET_BUILD_DIRECTORY}' ")
        developer_status_message("DEVEL" "  >> Cpp std: 'c++${CMAKE_CXX_STANDARD}' ")
        developer_status_message("DEVEL" "  >> suppressions-list: '${CMAKE_SOURCE_DIR}/dist/${CPPCHECK_CONFIG_FILE}' ")
        developer_status_message("DEVEL" "  >> checkers-report: ${TARGET_BUILD_DIRECTORY}/${CPPCHECK_REPORT_FILE}' ")
        developer_status_message("DEVEL" "  >> CppCheck located at: '${CPPCHECK_EXECUTABLE}' ")
        developer_status_message("DEVEL" "  >> CppCheck options: '${CPPCHECK_OPTION_LIST}' ")
    endif()
endfunction()

function(check_compiler_requirements component_name)
    ##  We need to make sure we have C++ enabled, or we get errors like:
    ##  'check_compiler_flag: CXX: needs to be enabled before use'
    get_property(project_enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if(NOT project_enabled_languages OR NOT "CXX" IN_LIST project_enabled_languages)
        enable_language(CXX)
    endif()

    
    ##  Check if we are able to use Lightning (Clang++) as default compiler
    ##  Note:   If this condition is met, we used rocm_clang_toolchain.cmake and the toolchain was already
    ##          checked and set up.
    if(NOT IS_LIGHTNING_CLANG_DEFAULT_COMPILER AND NOT ROCM_CLANG_TOOLCHAIN_USED)
        message(FATAL_ERROR ">> ROCm 'Lightning Clang++' Toolchain: was not set (rocm_clang_toolchain.cmake) ...")
    endif()
   

    ##  Check if the compiler is compatible with the C++ standard.
    ##  Note:   Minimum required is ${CMAKE_CXX_STANDARD} = 20, but we check for 23, 20, and 17.
    if(NOT DEFINED IS_COMPILER_SUPPORTS_CXX23_STANDARD OR NOT DEFINED IS_COMPILER_SUPPORTS_CXX20_STANDARD OR NOT DEFINED IS_COMPILER_SUPPORTS_CXX17_STANDARD)
        include(CheckCXXCompilerFlag)
        message(STATUS ">> Checking Compiler: '${CMAKE_CXX_COMPILER}' for C++ standard ...")

        ## Just to have independent checks/variables
        set(CHECK_CMAKE_CXX_STANDARD 23)
        if(NOT DEFINED IS_COMPILER_SUPPORTS_CXX23_STANDARD)
            set(IS_COMPILER_SUPPORTS_CHECK "IS_COMPILER_SUPPORTS_CXX${CHECK_CMAKE_CXX_STANDARD}_STANDARD")
            check_cxx_compiler_flag("-std=c++${CHECK_CMAKE_CXX_STANDARD}" COMPILER_SUPPORTS_CXX23_STANDARD)
            if(COMPILER_SUPPORTS_CXX23_STANDARD)
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL TRUE)
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL TRUE PARENT_SCOPE)
                developer_status_message("DEVEL" " >> Compiler: ${CMAKE_CXX_COMPILER} supports CXX Standard '${CHECK_CMAKE_CXX_STANDARD}' ...")
            else()
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL FALSE)
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL FALSE PARENT_SCOPE)
            endif()
        endif()

        set(CHECK_CMAKE_CXX_STANDARD 20)
        if(NOT DEFINED IS_COMPILER_SUPPORTS_CXX20_STANDARD)
            set(IS_COMPILER_SUPPORTS_CHECK "IS_COMPILER_SUPPORTS_CXX${CHECK_CMAKE_CXX_STANDARD}_STANDARD")
            check_cxx_compiler_flag("-std=c++${CHECK_CMAKE_CXX_STANDARD}" COMPILER_SUPPORTS_CXX20_STANDARD)
            if(COMPILER_SUPPORTS_CXX20_STANDARD)
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL TRUE)
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL TRUE PARENT_SCOPE)
                developer_status_message("DEVEL" "  >> Compiler: ${CMAKE_CXX_COMPILER} supports CXX Standard '${CHECK_CMAKE_CXX_STANDARD}' ...")
            else()
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL FALSE)
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL FALSE PARENT_SCOPE)
            endif()
        endif()

        set(CHECK_CMAKE_CXX_STANDARD 17)
        if(NOT DEFINED IS_COMPILER_SUPPORTS_CXX17_STANDARD)
            set(IS_COMPILER_SUPPORTS_CHECK "IS_COMPILER_SUPPORTS_CXX${CHECK_CMAKE_CXX_STANDARD}_STANDARD")
            check_cxx_compiler_flag("-std=c++${CHECK_CMAKE_CXX_STANDARD}" COMPILER_SUPPORTS_CXX17_STANDARD)
            if(COMPILER_SUPPORTS_CXX17_STANDARD)
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL TRUE)
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL TRUE PARENT_SCOPE)
                developer_status_message("DEVEL" "  >> Compiler: ${CMAKE_CXX_COMPILER} supports CXX Standard '${CHECK_CMAKE_CXX_STANDARD}' ...")
            else()
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL FALSE)
                set(${IS_COMPILER_SUPPORTS_CHECK} BOOL FALSE PARENT_SCOPE)
            endif()
        endif()
    endif()

    ## Does it support the project C++ standard, ${CMAKE_CXX_STANDARD} = 20?
    set(IS_COMPILER_SUPPORTS_MIN_STANDARD "${IS_COMPILER_SUPPORTS_CXX${CMAKE_CXX_STANDARD}_STANDARD}")
    if(NOT IS_COMPILER_SUPPORTS_MIN_STANDARD)
        message(FATAL_ERROR ">> Compiler: '${CMAKE_CXX_COMPILER}' v'${CMAKE_CXX_COMPILER_VERSION}' doesn't support CXX Standard '${CMAKE_CXX_STANDARD}'! \n"
                             "  >> Project: '${${component_name}}' can't be built ...")
    else()
        message(STATUS ">> Compiler: '${CMAKE_CXX_COMPILER}' v'${CMAKE_CXX_COMPILER_VERSION}' supports the required CXX Standard '${CMAKE_CXX_STANDARD}' ...")
    endif()
endfunction()


#
#   Note: All macro definitions here
macro(set_variable_in_parent variable value)
    get_directory_property(has_parent PARENT_DIRECTORY)

    if(has_parent)
        set(${variable} "${value}" PARENT_SCOPE)
    else()
        set(${variable} "${value}")
    endif()
endmacro()

macro(setup_cmake target_name target_version)
    message(STATUS ">> Building ${${target_name}} v${${target_version}} ...")
    #   If building shared libraries or linking static libraries into shared ones
    if(TRANSFERBENCH_ENGINE_SHARED)
        set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Set position independent code for all targets ..." FORCE)
    endif()
    message(STATUS ">> Configuring CMake to use the following build tools...")
    check_compiler_requirements(${target_name})

    #
    find_program(CCACHE_PATH ccache)
    find_program(NINJA_PATH ninja)
    find_program(LD_LLD_PATH ld.lld)
    find_program(LD_MOLD_PATH ld.mold)

    if(NOT IS_LIGHTNING_CLANG_DEFAULT_COMPILER)
        if(CCACHE_PATH)
            set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PATH})
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PATH})
        else()
            message(WARNING ">> CCache was not found!")
        endif()
    endif()

    if(NINJA_PATH)
        set(CMAKE_GENERATOR Ninja)
    else()
        message(WARNING ">> Ninja was not found! Using default generator.")
    endif()

    #   Lets give priority to MOLD linker
    set(AMD_PROJECT_LINKER_OPTION "")
    if(LD_MOLD_PATH AND TRANSFERBENCH_LINKER_TRY_MOLD)
        set(CMAKE_LINKER ${LD_MOLD_PATH} CACHE STRING "Linker to use: ${LD_MOLD_PATH}")
        set(AMD_PROJECT_LINKER_OPTION "-fuse-ld=mold")
    #   Then LLD linker
    elseif(LD_LLD_PATH)
        set(CMAKE_LINKER ${LD_LLD_PATH} CACHE STRING "Linker to use: ${LD_LLD_PATH}")
        set(AMD_PROJECT_LINKER_OPTION "-fuse-ld=lld")
    else()
        message(WARNING ">> LLD linker was not found! Using default 'Gold' linker.")
    endif()

    if(LD_MOLD_PATH OR LD_LLD_PATH AND AMD_PROJECT_LINKER_OPTION)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${AMD_PROJECT_LINKER_OPTION}")
        message(STATUS ">> Using linker: '${CMAKE_LINKER}' with options: '${AMD_PROJECT_LINKER_OPTION}'")
    endif()
      

    #   CMake policies for the project
    foreach(_policy
        CMP0028 CMP0046 CMP0048 CMP0051 CMP0054
        CMP0056 CMP0063 CMP0065 CMP0074 CMP0075
        CMP0077 CMP0082 CMP0093 CMP0127 CMP0135)
        if(POLICY ${_policy})
            cmake_policy(SET ${_policy} NEW)
        endif()
    endforeach()

    set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "Disable deprecated warning messages" FORCE)
endmacro()

macro(add_build_definitions)
    if(NOT PROJECT_TARGET_VERSION)
        message(FATAL_ERROR ">> Project: 'PROJECT_TARGET_VERSION' was not defined!")
    endif()

    message(STATUS ">> Project: '${PROJECT_NAME}' v${${PROJECT_NAME}_VERSION} ...")
    set (CMAKE_RC_FLAGS "${CMAKE_RC_FLAGS} -DAMD_PROJECT_VERSION_MAJOR=${AMD_PROJECT_VERSION_MAJOR} 
                                           -DAMD_PROJECT_VERSION_MINOR=${AMD_PROJECT_VERSION_MINOR} 
                                           -DAMD_PROJECT_VERSION_MINOR=${AMD_PROJECT_VERSION_MINOR}")

    if (TRANSFERBENCH_ENGINE_HEADER_ONLY)
        add_compile_definitions(AMD_TRANSFERBENCH_ENGINE_HEADER_ONLY)
    endif()
    if (TRANSFERBENCH_ENGINE_STATIC)
        add_compile_definitions(AMD_TRANSFERBENCH_ENGINE_STATIC)
    endif()
    if (TRANSFERBENCH_ENGINE_SHARED)
        add_compile_definitions(AMD_TRANSFERBENCH_ENGINE_SHARED)
    endif()
endmacro()

macro(setup_compiler_init_flags)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-ftrivial-auto-var-init=zero HAS_TRIVIAL_AUTO_VAR_INIT_COMPILER)

    if(NOT COMPILER_INIT_FLAG)
        if(HAS_TRIVIAL_AUTO_VAR_INIT_COMPILER)
            message(STATUS ">> Compiler supports -ftrivial-auto-var-init")
            set(COMPILER_INIT_FLAG "-ftrivial-auto-var-init=zero" CACHE STRING "Using cache trivially-copyable automatic variable initialization.")
        else()
            message(STATUS ">> Compiler does not support -ftrivial-auto-var-init")
            set(COMPILER_INIT_FLAG " " CACHE STRING "Using cache trivially-copyable automatic variable initialization.")
        endif()
    endif()

    ##  Initialize automatic variables with either a pattern or with zeroes to increase program security by preventing
    ##  uninitialized memory disclosure and use. '-ftrivial-auto-var-init=[uninitialized|pattern|zero]' where
    ##  'uninitialized' is the default, 'pattern' initializes variables with a pattern, and 'zero' initializes variables
    ##  with zeroes.
    set(AMD_WORK_BENCH_COMMON_FLAGS "${AMD_WORK_BENCH_COMMON_FLAGS} ${COMPILER_INIT_FLAG}")
endmacro()

macro(setup_compression_flags)
    include(CheckCXXCompilerFlag)
    include(CheckLinkerFlag)
    check_cxx_compiler_flag(-gz=zstd ZSTD_AVAILABLE_COMPILER)
    check_linker_flag(CXX -gz=zstd ZSTD_AVAILABLE_LINKER)
    check_cxx_compiler_flag(-gz COMPRESS_AVAILABLE_COMPILER)
    check_linker_flag(CXX -gz COMPRESS_AVAILABLE_LINKER)

    # From cache
    if(NOT DEBUG_COMPRESSION_FLAG)
        if(ZSTD_AVAILABLE_COMPILER AND ZSTD_AVAILABLE_LINKER)
            message(STATUS ">> Compiler and Linker support ZSTD... using it.")
            set(DEBUG_COMPRESSION_FLAG "-gz=zstd" CACHE STRING "Using cache for debug info compression.")
        elseif(COMPRESS_AVAILABLE_COMPILER AND COMPRESS_AVAILABLE_LINKER)
            message(STATUS ">> Compiler and Linker support default compression... using it.")
            set(DEBUG_COMPRESSION_FLAG "-gz" CACHE STRING "Using cache for debug info compression.")
        endif()
    endif()

    set(AMD_WORK_BENCH_COMMON_FLAGS "${AMD_WORK_BENCH_COMMON_FLAGS} ${DEBUG_COMPRESSION_FLAG}")
endmacro()

macro(setup_default_compiler_flags target_name)
    #   Compiler specific flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_common_flag("-Wall" ${target_name})
        add_common_flag("-Wextra" ${target_name})
        add_common_flag("-Wno-unused-function" ${target_name})
        add_common_flag("-Wno-unused-variable" ${target_name})
        add_common_flag("-Wpedantic" ${target_name})

        if(TRANSFERBENCH_TREAT_WARNINGS_AS_ERRORS)
            add_common_flag("-Werror" ${target_name})
        endif()

        if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            add_common_flag("-rdynamic" ${target_name})
        endif()

        ##
        ## -fno-omit-frame-pointer -fno-strict-aliasing -fvisibility=hidden -fvisibility-inlines-hidden
        ## -fno-exceptions -fno-rtti
        add_cxx_flag("-fexceptions" ${target_name})
        add_cxx_flag("-frtti" ${target_name})
        add_cxx_flag("-fno-omit-frame-pointer" ${target_name})
        add_c_cxx_flag("-Wno-array-bounds" ${target_name})
        add_c_cxx_flag("-Wno-deprecated-declarations" ${target_name})
        add_c_cxx_flag("-Wno-unknown-pragmas" ${target_name})

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            add_c_cxx_flag("-Wno-restrict" ${target_name})
            add_c_cxx_flag("-Wno-stringop-overread" ${target_name})
            add_c_cxx_flag("-Wno-stringop-overflow" ${target_name})
            add_c_cxx_flag("-Wno-dangling-reference" ${target_name})

        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            add_c_cxx_flag("-Wno-unknown-warning-option" ${target_name})
        endif()

        if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_common_flag("-O1" ${target_name})

            if(TRANSFERBENCH_HARDENING_ENABLED)
                ##  Building with _FORTIFY_SOURCE=3 may impact the size and performance of the code. Since _FORTIFY_SOURCE=2 
                ##  generated only constant sizes, its overhead was negligible. However, _FORTIFY_SOURCE=3 may generate 
                ##  additional code to compute object sizes. These additions may also cause secondary effects, such as register 
                ##  pressure during code generation. Code size tends to increase the size of resultant binaries for the same reason.
                ##
                ##  _FORTIFY_SOURCE=3 has led to significant gains in security mitigation, but it may not be suitable for all
                ##  applications. We need a proper study of performance and code size to understand the magnitude of the impact 
                ##  created by _FORTIFY_SOURCE=3 additional runtime code generation, but the performance, and code size might well 
                ##  be worth the magnitude of the security benefits.  _FORTIFY_SOURCE requires compiling with optimization (-O).
                ##
                add_common_flag("-U_FORTIFY_SOURCE" ${target_name})
                add_common_flag("-D_FORTIFY_SOURCE=2" ${target_name})

                ##  Stack canary check for buffer overflow on the stack. 
                ##  Emit extra code to check for buffer overflows, such as stack smashing attacks. This is done by adding a guard 
                ##  variable to functions with vulnerable objects. This includes functions that call alloca, and functions with 
                ##  buffers larger than or equal to 8 bytes.
                ##  Only variables that are actually allocated on the stack are considered, optimized away variables or variables 
                ##  allocated in registers don’t count. 
                ##  'stack-protector-strong' is a stronger version of 'stack-protector', but includes additional functions to be 
                ##  protected — those that have local array definitions, or have references to local frame addresses. Only 
                ##  variables that are actually allocated on the stack are considered, optimized away variables or variables 
                ##  allocated in registers don’t count.
                ##
                add_common_flag("-fstack-protector-strong" ${target_name})
            endif()
        endif()            

        if(TRANSFERBENCH_COMPRESS_DEBUG_INFO)
            setup_compression_flags()
        endif()

        ##  Compiler initialization flags
        setup_compiler_init_flags()

        ## RelWithDebInfo builds, minimum debug info
        if (NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                add_c_cxx_flag("-g3" ${target_name})
            endif()

            ## Inline function debugg
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
                add_c_cxx_flag("-ginline-points" ${target_name})
                add_c_cxx_flag("-gstatement-frontiers" ${target_name})
            endif()
        endif()

    endif()

    ## TODO: Check if RPATH settings is needed
endmacro()

macro(developer_status_message message_mode message)
    #   Note:   This macro is used to print developer messages.
    has_build_debug_mode(HAS_DEBUG_MODE_ENABLED)
    if(HAS_DEBUG_MODE_ENABLED)
        #   Check for valid message mode
        #   Note:   We will use the 'STATUS' message mode as default if the user doesn't set it or
        if(NOT "${message_mode}" MATCHES "^(STATUS|WARNING|ERROR|DEBUG|FATAL_ERROR|DEVEL)$")
            message(WARNING "[DEVELOPER]: The '${message_mode}' message mode is not supported for message: '${message}' .")
        else()

            #   ${message_mode} doesn't work here. CMake interpreter sees it as a string; "STATUS", "WARNING"...
            if("${message_mode}" STREQUAL "STATUS")
                message(STATUS "[DEVELOPER]: ${message}")
            elseif("${message_mode}" STREQUAL "WARNING")
                message(WARNING "[DEVELOPER]: ${message}")
            elseif("${message_mode}" STREQUAL "ERROR")
                message(ERROR "[DEVELOPER]: ${message}")
            elseif("${message_mode}" STREQUAL "DEBUG")
                message(DEBUG "[DEVELOPER]: ${message}")
            elseif("${message_mode}" STREQUAL "FATAL_ERROR")
                message(FATAL_ERROR "[DEVELOPER]: ${message}")
            elseif("${message_mode}" STREQUAL "DEVEL")
                message(STATUS "[DEVELOPER]: ${message}")
            else()
                message(WARNING "[DEVELOPER]: ${message}, with invalid message mode: '${message_mode}'")
            endif()
        endif()
    endif()
endmacro()

