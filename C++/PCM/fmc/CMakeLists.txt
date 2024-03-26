file(GLOB implem_files *.cpp)
file(GLOB header_files *.h)

set (CMAKE_CXX_FLAGS "-std=c++14 -O3 ${CMAKE_CXX_FLAGS}")

set (EXE_NAME ${POD_NAME}-fmc)

add_executable(${EXE_NAME} ${implem_files} ${header_files})

