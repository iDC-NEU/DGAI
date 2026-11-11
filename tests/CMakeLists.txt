set(CMAKE_CXX_STANDARD 17)

add_executable(build_disk_index build_disk_index.cpp)
target_link_libraries(build_disk_index ${PROJECT_NAME})

add_executable(build_pipnn_index build_pipnn_index.cpp)
target_link_libraries(build_pipnn_index ${PROJECT_NAME})

add_executable(build_memory_index build_memory_index.cpp)
target_link_libraries(build_memory_index ${PROJECT_NAME})

add_executable(overall_performance overall_performance.cpp)
target_link_libraries(overall_performance ${PROJECT_NAME})

add_executable(test_insert_search test_insert_search.cpp)
target_link_libraries(test_insert_search ${PROJECT_NAME})

# Portable deterministic split checks; no datasets or index files required.
add_executable(test_locality_split test_locality_split.cpp)
target_link_libraries(test_locality_split ${PROJECT_NAME})

add_executable(overall_perf_mem overall_perf_mem.cpp)
target_link_libraries(overall_perf_mem ${PROJECT_NAME})

add_executable(search_disk_index search_disk_index.cpp)
target_link_libraries(search_disk_index ${PROJECT_NAME})

add_executable(search_disk_index_mem search_disk_index_mem.cpp)
target_link_libraries(search_disk_index_mem ${PROJECT_NAME})

add_executable(search_disk_index_filtered search_disk_index_filtered.cpp)
target_link_libraries(search_disk_index_filtered ${PROJECT_NAME})

add_executable(pad_partition pad_partition.cpp)
target_link_libraries(pad_partition ${PROJECT_NAME} )

add_executable(normalize_data normalize_data.cpp)
target_link_libraries(normalize_data ${PROJECT_NAME})

add_executable(memnav_deep_update memnav_deep_update.cpp)
target_link_libraries(memnav_deep_update pipeann)
add_executable(test_memnav_safety test_memnav_safety.cpp)
target_link_libraries(test_memnav_safety pthread)

add_executable(test_memnav_online test_memnav_online.cpp)
target_link_libraries(test_memnav_online pthread)
