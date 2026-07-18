cmake_minimum_required(VERSION 3.18)
project(cholgraph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# XLA FFI headers ship with jaxlib (header-only, nothing to link).
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import jax.ffi; print(jax.ffi.include_dir())"
    OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)

# CHOLMOD: default to the active Python environment's prefix (conda-forge
# suitesparse), overridable with -DCHOLMOD_PREFIX=...
if(NOT DEFINED CHOLMOD_PREFIX)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
        OUTPUT_VARIABLE CHOLMOD_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

find_library(CHOLMOD_LIBRARY cholmod PATHS ${CHOLMOD_PREFIX}/lib NO_DEFAULT_PATH)
find_path(CHOLMOD_INCLUDE_DIR cholmod.h
    PATHS ${CHOLMOD_PREFIX}/include/suitesparse ${CHOLMOD_PREFIX}/include
    NO_DEFAULT_PATH)

if(NOT CHOLMOD_LIBRARY OR NOT CHOLMOD_INCLUDE_DIR)
    message(FATAL_ERROR "CHOLMOD not found under ${CHOLMOD_PREFIX}. "
        "Install suitesparse (conda install -c conda-forge suitesparse) or set CHOLMOD_PREFIX.")
endif()

# Locate nanobind via its Python package.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
find_package(nanobind REQUIRED CONFIG)

nanobind_add_module(cholgraph_cpp src/cpp/cholgraph.cpp)

target_include_directories(cholgraph_cpp PRIVATE
    ${CHOLMOD_INCLUDE_DIR}
    ${XLA_FFI_INCLUDE_DIR})
target_link_libraries(cholgraph_cpp PRIVATE ${CHOLMOD_LIBRARY})

# conda's libcholmod has an @rpath install name; point the module at it.
set_target_properties(cholgraph_cpp PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CHOLMOD_PREFIX}/lib")

install(TARGETS cholgraph_cpp LIBRARY DESTINATION .)
