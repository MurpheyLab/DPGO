set(PCM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)

add_subdirectory (fmc)

file(GLOB PCM_SRCS src/*.c src/*.cpp src/*.cc fmc/*.cpp)
file(GLOB PCM_HDRS include/PCM/*.h include/PCM/*.hpp fmc/*.h)
list(REMOVE_ITEM PCM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/fmc/main_fmc.cpp)

message(STATUS "PCM include directories: " ${PCM_INCLUDE_DIR})
message(STATUS "PCM header files: " ${PCM_HDRS})
message(STATUS "PCM source files: " ${PCM_SRCS})

add_library(PCM SHARED ${PCM_HDRS} ${PCM_SRCS})
target_include_directories(PCM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(PCM PUBLIC ${EIGEN3_INCLUDE_DIR} ${PCM_INCLUDE_DIR})

export(PACKAGE PCM)
export(TARGETS PCM FILE PCMConfig.cmake)
