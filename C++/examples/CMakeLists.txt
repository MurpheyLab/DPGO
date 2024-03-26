find_package(Boost 1.60 COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

add_executable(dist_pgo dist_pgo.cpp)
target_include_directories(dist_pgo PUBLIC ${Boost_INCLUDE_DIR})
target_link_libraries(dist_pgo DPGO DChordal SESync ${Boost_LIBRARIES})

add_executable(se_sync se_sync.cpp)
target_link_libraries(se_sync SESync)