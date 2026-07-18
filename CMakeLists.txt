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

# --- CHOLMOD ---
# Prefer CMake's own package discovery first: modern SuiteSparse (>=6, as
# shipped by conda-forge, vcpkg, Homebrew, and a from-source `cmake --install`)
# exports a CHOLMODConfig.cmake with an imported target. This works
# regardless of where/how CHOLMOD was installed and needs no prefix-guessing.
# The exported target has been named both `SuiteSparse::CHOLMOD` and
# `CHOLMOD::CHOLMOD` across SuiteSparse/packaging versions, so check for both.
find_package(CHOLMOD CONFIG QUIET)

set(CHOLMOD_TARGET "")
if(TARGET SuiteSparse::CHOLMOD)
    set(CHOLMOD_TARGET SuiteSparse::CHOLMOD)
elseif(TARGET CHOLMOD::CHOLMOD)
    set(CHOLMOD_TARGET CHOLMOD::CHOLMOD)
endif()

if(NOT CHOLMOD_TARGET)
    # No CMake package found (e.g. an older distro package with no config
    # file). Fall back to guessing a prefix: default to the active Python
    # environment's prefix (conda-forge suitesparse), overridable with
    # -DCHOLMOD_PREFIX=... for non-Python-environment installs (system
    # packages, Homebrew, a custom install location).
    if(NOT DEFINED CHOLMOD_PREFIX)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
            OUTPUT_VARIABLE CHOLMOD_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()

    # conda-forge installs headers/libs under <prefix>/Library on Windows,
    # directly under <prefix> on Linux/macOS.
    if(WIN32)
        set(CHOLMOD_HINT_PREFIX "${CHOLMOD_PREFIX}/Library")
    else()
        set(CHOLMOD_HINT_PREFIX "${CHOLMOD_PREFIX}")
    endif()

    # HINTS (not PATHS+NO_DEFAULT_PATH): try the guessed prefix first, but
    # still fall through to CMake's normal system search (e.g. Debian/Ubuntu's
    # multiarch /usr/lib/x86_64-linux-gnu/, Homebrew's /opt/homebrew) if that
    # guess is wrong or CHOLMOD lives somewhere standard instead.
    find_library(CHOLMOD_LIBRARY NAMES cholmod libcholmod
        HINTS ${CHOLMOD_HINT_PREFIX}/lib)
    find_path(CHOLMOD_INCLUDE_DIR cholmod.h
        HINTS ${CHOLMOD_HINT_PREFIX}/include/suitesparse ${CHOLMOD_HINT_PREFIX}/include)

    if(NOT CHOLMOD_LIBRARY OR NOT CHOLMOD_INCLUDE_DIR)
        message(FATAL_ERROR
            "CHOLMOD not found. Checked for a CMake package config, the "
            "prefix ${CHOLMOD_HINT_PREFIX}, and standard system locations. "
            "Install suitesparse (conda install -c conda-forge suitesparse; "
            "apt install libsuitesparse-dev; brew install suitesparse; vcpkg "
            "install suitesparse) or set CHOLMOD_PREFIX / CMAKE_PREFIX_PATH.")
    endif()
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

target_include_directories(cholgraph_cpp PRIVATE ${XLA_FFI_INCLUDE_DIR})

if(CHOLMOD_TARGET)
    target_link_libraries(cholgraph_cpp PRIVATE ${CHOLMOD_TARGET})
else()
    target_include_directories(cholgraph_cpp PRIVATE ${CHOLMOD_INCLUDE_DIR})
    target_link_libraries(cholgraph_cpp PRIVATE ${CHOLMOD_LIBRARY})
endif()

# Embed an RPATH covering every library actually linked (CHOLMOD, and
# whatever BLAS/LAPACK/METIS it pulls in), wherever they really live --
# conda, Homebrew, apt, or vcpkg -- rather than hardcoding one guessed path.
set_target_properties(cholgraph_cpp PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS cholgraph_cpp LIBRARY DESTINATION .)