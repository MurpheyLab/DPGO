cmake_minimum_required(VERSION 3.3)

# PROJECT CONFIGURATION
project(MM-PGO LANGUAGES C CXX VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # We require C++ 17


# Set build type to 'RelWithDebInfo' if one was not specified by the user
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS RelWithDebInfo Release Debug MinSizeRel)
  message(STATUS "Setting build type to ${CMAKE_BUILD_TYPE}, as none was specified\n")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode\n")
endif()

message(STATUS "C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}\n")

# Directory for built libraries
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "The directory in which to place the MM-PGO library built by this project")
# Directory for built executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "The directory in which to place executables built by this project")
# Add the .cmake files to CMake module path (useful for finding other stuff)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE STRING "The CMake module path used for this project")

# BUILD CONFIGURATIONS
# Enable faster instruction sets (SIMD/AVX)
set(ENABLE_FAST_INSTRUCTIONS ON CACHE BOOL "Enable faster instruction sets (SIMD/AVX)")
# Enable OpenMP (if available)
set(ENABLE_OPENMP ON CACHE BOOL "Enable OpenMP (if available)")
# Enable code profiling using gperftools
set(ENABLE_PROFILING OFF CACHE BOOL "Enable code profiling using gperftools")

if(${ENABLE_FAST_INSTRUCTIONS})
message(STATUS "Enabling SIMD/AVX instruction sets")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
endif()

if(${ENABLE_PROFILING})
message(STATUS "Enabling code profiling using Google Performance Tools")
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} -lprofiler)
endif()

message(STATUS "")

# DPGO & DChordal DEPENDENCIES
find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)

# FIND ADDITIONAL LIBRARIES
# These next operations make use of the .cmake files shipped with Eigen3
find_package(SPQR REQUIRED)
find_package(Cholmod REQUIRED)

# PERFORMANCE IMPROVEMENTS
if(${ENABLE_OPENMP})
  find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
  message(STATUS "\nFound OpenMP! Turning on support for parallelization\n")
endif()
endif()

# FIND SPECTRA
# Spectra doesn't ship with a .cmake file, but it's header-only, so it suffices to simply determine it's top-level directory
set(SPECTRA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Spectra/include CACHE PATH "Path to top-level directory of Spectra template library")
message(STATUS "Spectra include directory: ${SPECTRA_INCLUDE_DIR}\n")

# Find Optimization library
set(OPTIMIZATION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Optimization/ CACHE PATH "Path to top-level directory of Optimization (the one containing CMakeLists.txt)")

# Run the Optimization package's CMake file
add_subdirectory(${OPTIMIZATION_DIR})

# Find Preconditioners library
set(PRECONDITIONERS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Preconditioners/ CACHE PATH "Path to top-level directory of Preconditioners library (the one containing CMakeLists.txt)")

# Run the Preconditioners package's CMake file
add_subdirectory(${PRECONDITIONERS_DIR})

# Find SESync library
set(SESync_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SESync/ CACHE PATH "Path to top-level directory of SESync (the one containing CMakeLists.txt)")

# Run the SESync package's CMake file
add_subdirectory(${SESync_DIR})

# Find PCM library
set(PCM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/PCM/ CACHE PATH "Path to top-level directory of DChordal (the one containing CMakeLists.txt)")

# Run the PCM package's CMake file
add_subdirectory(${PCM_DIR})

# Find DPGO library
set(DPGO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DPGO/ CACHE PATH "Path to top-level directory of DPGO (the one containing CMakeLists.txt)")

# Run the DPGO package's CMake file
add_subdirectory(${DPGO_DIR})

# Find DChordal library
set(DChordal_DIR ${CMAKE_CURRENT_SOURCE_DIR}/DChordal/ CACHE PATH "Path to top-level directory of DChordal (the one containing CMakeLists.txt)")

# Run the DChordal package's CMake file
add_subdirectory(${DChordal_DIR})

# Examples
add_subdirectory(examples)
