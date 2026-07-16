cmake_minimum_required(VERSION 3.22)


# VERSION SETTING

if (NOT DEFINED HECATE_VERSION_MAJOR)
  set(HECATE_VERSION_MAJOR 0)
endif ()

if (NOT DEFINED HECATE_VERSION_MINOR)
  set(HECATE_VERSION_MINOR 1)
endif ()

if (NOT DEFINED HECATE_VERSION_PATCH)
  set(HECATE_VERSION_PATCH 0)
endif ()

if (NOT DEFINED HECATE_VERSION_SUFFIX)
  set(HECATE_VERSION_SUFFIX git)
endif ()

if (NOT PACKAGE_VERSION)
  set(PACKAGE_VERSION
    "${HECATE_VERSION_MAJOR}.${HECATE_VERSION_MINOR}.${HECATE_VERSION_PATCH}${HECATE_VERSION_SUFFIX}")
endif()

# PROJECT SETTING

project(HECATE
  VERSION ${HECATE_VERSION_MAJOR}.${HECATE_VERSION_MINOR}.${HECATE_VERSION_PATCH}
  LANGUAGES CXX C)

option(BUILD_TESTING "Build Dacapo tests" OFF)
include(CTest)


# CXX BUILD SETTING

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

if (CMAKE_BUILD_TYPE AND
    NOT uppercase_CMAKE_BUILD_TYPE MATCHES "^(DEBUG|RELEASE|RELWITHDEBINFO|MINSIZEREL)$")
  message(FATAL_ERROR "Invalid value for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# set(CMAKE_BUILD_TYPE Debug)

# PACKAGE SETTING

option(HECATE_ENABLE_SEAL_RUNTIME "Build the SEAL HEVM runtime" OFF)
option(HECATE_ENABLE_HEAAN_RUNTIME "Build the HEaaN HEVM runtime" OFF)

find_package(MLIR REQUIRED CONFIG)
if (HECATE_ENABLE_SEAL_RUNTIME)
  find_package(SEAL 4.0 REQUIRED CONFIG)
endif()
if (HECATE_ENABLE_HEAAN_RUNTIME)
  find_package(HEaaN REQUIRED CONFIG)
endif()
find_package(OpenMP)

# Packaged MLIR installations may expose the TableGen tools by name instead of
# by absolute path. Resolve them here so Ninja does not mistake the executable
# name for a build target.
if (NOT IS_ABSOLUTE "${MLIR_TABLEGEN_EXE}")
  find_program(HECATE_MLIR_TABLEGEN
    NAMES ${MLIR_TABLEGEN_EXE} mlir-tblgen
    HINTS "${MLIR_INSTALL_PREFIX}/bin" "${LLVM_TOOLS_BINARY_DIR}"
          "${LLVM_INSTALL_PREFIX}/bin")
  if (HECATE_MLIR_TABLEGEN)
    set(MLIR_TABLEGEN_EXE "${HECATE_MLIR_TABLEGEN}")
    set(MLIR_TABLEGEN_TARGET "${HECATE_MLIR_TABLEGEN}")
  endif()
endif()

if (NOT IS_ABSOLUTE "${LLVM_TABLEGEN_EXE}")
  find_program(HECATE_LLVM_TABLEGEN
    NAMES ${LLVM_TABLEGEN_EXE} llvm-tblgen
    HINTS "${LLVM_TOOLS_BINARY_DIR}" "${LLVM_INSTALL_PREFIX}/bin")
  if (HECATE_LLVM_TABLEGEN)
    set(LLVM_TABLEGEN_EXE "${HECATE_LLVM_TABLEGEN}")
    set(LLVM_TABLEGEN_TARGET "${HECATE_LLVM_TABLEGEN}")
  endif()
endif()

# MLIR SETTING

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddHecate)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROTOBUF_INCLUDE_DIR})
# include_directories(${HEAAN_INCLUDE_DIR})

# Nix compiler wrappers inject dependency, standard-library, resource, and libc
# include paths at invocation time. CMake classifies them as implicit and omits
# them from compile_commands.json, but clangd does not execute the wrapper to
# recover those flags. Keep explicit copies in the compilation database.
if(DEFINED ENV{IN_NIX_SHELL})
  foreach(HECATE_TOOLING_INCLUDE_DIR IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
    add_compile_options(
      "$<$<COMPILE_LANGUAGE:CXX>:-isystem${HECATE_TOOLING_INCLUDE_DIR}>")
  endforeach()
  foreach(HECATE_TOOLING_INCLUDE_DIR IN LISTS CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES)
    add_compile_options(
      "$<$<COMPILE_LANGUAGE:C>:-isystem${HECATE_TOOLING_INCLUDE_DIR}>")
  endforeach()
endif()

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# HECATE DIRECTORY

add_compile_options(-fno-rtti)
add_compile_options(-fno-exceptions)
add_compile_options(-fPIC)
add_definitions (-DMEM_CHECK)

add_subdirectory (include/hecate)
add_subdirectory (lib)
add_subdirectory (tools)
if (BUILD_TESTING)
  add_subdirectory(test)
endif()

# get_target_property(HEAAN_INCLUDE_DIR HEaaN::HEaaN INTERFACE_INCLUDE_DIRECTORIES)
