# DChordal PROJECT
set(DChordal_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DChordal_HDR_DIR ${DChordal_INCLUDE_DIR}/DChordal)
set(DChordal_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Set the include directories for this project
# Spectra is only used internally by the DChordal library
set(DChordal_PRIVATE_INCLUDES ${SPECTRA_INCLUDE_DIR} CACHE INTERNAL "")
# Set the include directories for this project
# The DChordal headers and and Eigen 3, SPQR, and Cholmod are all referenced by the header files of the DChordal library, hence must be PUBLICLY included (i.e. clients using the DChordal headers must also include these headers)
set(DChordal_INCLUDES ${DChordal_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${SPQR_INCLUDES} ${CHOLMOD_INCLUDES} CACHE INTERNAL "")

# Get the set of DChordal header and source files
set(DChordal_HDRS
#${DChordal_HDR_DIR}/StiefelProduct.h
${DChordal_HDR_DIR}/SOdProduct.h
${DChordal_HDR_DIR}/RelativePoseMeasurement.h
${DChordal_HDR_DIR}/DChordal_types.h
${DChordal_HDR_DIR}/DChordal_utils.h
${DChordal_HDR_DIR}/DChordalProblem.h
${DChordal_HDR_DIR}/DChordal.h
${DChordal_HDR_DIR}/DChordalReducedProblem.h
${DChordal_HDR_DIR}/DChordalReduced.h
)

set(DChordal_SRCS
${DChordal_SOURCE_DIR}/DChordal_utils.cpp
${DChordal_SOURCE_DIR}/DChordalProblem.cpp
${DChordal_SOURCE_DIR}/DChordal.cpp
${DChordal_SOURCE_DIR}/DChordalReducedProblem.cpp
${DChordal_SOURCE_DIR}/DChordalReduced.cpp
)

message(STATUS "Found DChordal include directory: ${DChordal_INCLUDE_DIR}")
message(STATUS "Found DChordal header files:\n ${DChordal_HDRS}\n")

# Build the DChordal library
add_library(DChordal SHARED ${DChordal_HDRS} ${DChordal_SRCS})
target_include_directories(DChordal PRIVATE ${DChordal_PRIVATE_INCLUDES})
target_include_directories(DChordal PUBLIC ${DChordal_INCLUDES})
target_link_libraries(DChordal DPGO SESync Optimization PCM ${BLAS_LIBRARIES} ${CHOLMOD_LIBRARIES} ${SPQR_LIBRARIES} ${M} ${LAPACK_LIBRARIES})

if(OPENMP_FOUND)
# Add additional compilation flags to enable OpenMP support
set_target_properties(DChordal PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
set_target_properties(DChordal PROPERTIES LINK_FLAGS "-fopenmp")
endif()

# EXPORT DChordal LIBRARY
# Add add entry for this project into CMake's package registry, so that this project can be found by other CMake projects
export(PACKAGE DChordal)
# Create a configuration file for this project, so that it can be imported by other CMake projects
export(TARGETS DChordal DPGO SESync Optimization PCM FILE DChordalConfig.cmake)
