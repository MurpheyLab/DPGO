## SE-Sync Project
set(SESync_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SESync_HDR_DIR ${SESync_INCLUDE_DIR}/SESync)
set(SESync_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Set the include directories for this project
# Spectra is only used internally by the SE-Sync library
set(SESync_PRIVATE_INCLUDES ${SPECTRA_INCLUDE_DIR} CACHE INTERNAL "")
# The SE-Sync headers and and Eigen 3, SPQR, and Cholmod are all referenced by the header files of the SE-Sync library, hence must be PUBLICLY included (i.e. clients using the SE-Sync headers must also include these headers)
set(SESync_INCLUDES ${SESync_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${SPQR_INCLUDES} ${CHOLMOD_INCLUDES} CACHE INTERNAL "")

# Get the set of SE-Sync header and source files
set(SESync_HDRS
${SESync_HDR_DIR}/StiefelProduct.h
${SESync_HDR_DIR}/RelativePoseMeasurement.h
${SESync_HDR_DIR}/SESync_types.h
${SESync_HDR_DIR}/SESync_utils.h
${SESync_HDR_DIR}/SESyncProblem.h
${SESync_HDR_DIR}/SESync.h
)

set(SESync_SRCS
${SESync_SOURCE_DIR}/StiefelProduct.cpp
${SESync_SOURCE_DIR}/SESync_utils.cpp
${SESync_SOURCE_DIR}/SESyncProblem.cpp
${SESync_SOURCE_DIR}/SESync.cpp
)

message(STATUS "Found SESync include directory: ${SESync_INCLUDE_DIR}")
message(STATUS "Found SESync header files:\n ${SESync_HDRS}\n")

# Build the SE-Sync library
add_library(SESync SHARED ${SESync_HDRS} ${SESync_SRCS})
target_include_directories(SESync PRIVATE ${SESync_PRIVATE_INCLUDES})
target_include_directories(SESync PUBLIC ${SESync_INCLUDES})
target_link_libraries(SESync Optimization ILDL ${BLAS_LIBRARIES} ${CHOLMOD_LIBRARIES} ${SPQR_LIBRARIES} ${M} ${LAPACK_LIBRARIES} ${GLOG_LIBRARIES})

if(OPENMP_FOUND)
# Add additional compilation flags to enable OpenMP support
set_target_properties(SESync PROPERTIES COMPILE_FLAGS ${OpenMP_CXX_FLAGS})
set_target_properties(SESync PROPERTIES LINK_FLAGS "-fopenmp")
endif()

export(PACKAGE SESync)
export(TARGETS SESync Optimization FILE SESyncConfig.cmake)
