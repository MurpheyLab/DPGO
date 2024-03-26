# DPGO PROJECT
set(DPGO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DPGO_HDR_DIR ${DPGO_INCLUDE_DIR}/DPGO)
set(DPGO_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Set the include directories for this project
# Spectra is only used internally by the DPGO library
set(DPGO_PRIVATE_INCLUDES ${SPECTRA_INCLUDE_DIR} CACHE INTERNAL "")
# Set the include directories for this project
# The DPGO headers and and Eigen 3, SPQR, and Cholmod are all referenced by the header files of the DPGO library, hence must be PUBLICLY included (i.e. clients using the DPGO headers must also include these headers)
set(DPGO_INCLUDES ${DPGO_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${SPQR_INCLUDES} ${CHOLMOD_INCLUDES} ${YAML_CPP_INCLUDE_DIR} CACHE INTERNAL "")

# Get the set of DPGO header and source files
set(DPGO_HDRS
#${DPGO_HDR_DIR}/StiefelProduct.h
${DPGO_HDR_DIR}/SOdProduct.h
${DPGO_HDR_DIR}/RelativePoseMeasurement.h
${DPGO_HDR_DIR}/DPGO_types.h
${DPGO_HDR_DIR}/DPGO_utils.h
${DPGO_HDR_DIR}/DPGOProblem.h
${DPGO_HDR_DIR}/DPGOHash.h
${DPGO_HDR_DIR}/DPGOStar.h
${DPGO_HDR_DIR}/PCM.h
)

set(DPGO_SRCS
${DPGO_SOURCE_DIR}/DPGO_utils.cpp
${DPGO_SOURCE_DIR}/DPGOProblem.cpp
${DPGO_SOURCE_DIR}/DPGOHash.cpp
${DPGO_SOURCE_DIR}/DPGOStar.cpp
${DPGO_SOURCE_DIR}/PCM.cpp
${DPGO_SOURCE_DIR}/internal/traits.cpp
${DPGO_SOURCE_DIR}/internal/project_to_SOd.cpp
)

message(STATUS "Found DPGO include directory: ${DPGO_INCLUDE_DIR}")
message(STATUS "Found DPGO header files:\n ${DPGO_HDRS}\n")

# Build the DPGO library
add_library(DPGO SHARED ${DPGO_HDRS} ${DPGO_SRCS})
target_include_directories(DPGO PRIVATE ${DPGO_PRIVATE_INCLUDES})
target_include_directories(DPGO PUBLIC ${DPGO_INCLUDES})
target_link_libraries(DPGO Optimization PCM ${BLAS_LIBRARIES} ${CHOLMOD_LIBRARIES} ${SPQR_LIBRARIES} ${M} ${LAPACK_LIBRARIES} glog)

if(OPENMP_FOUND)
# Add additional compilation flags to enable OpenMP support
set_target_properties(DPGO PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
set_target_properties(DPGO PROPERTIES LINK_FLAGS "-fopenmp")
endif()

# EXPORT DPGO LIBRARY
# Add add entry for this project into CMake's package registry, so that this project can be found by other CMake projects
export(PACKAGE DPGO)
# Create a configuration file for this project, so that it can be imported by other CMake projects
export(TARGETS DPGO Optimization PCM FILE DPGOConfig.cmake)
