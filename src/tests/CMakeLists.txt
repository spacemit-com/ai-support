add_executable(json_test json_test.cc)
target_include_directories(json_test PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(json_test PUBLIC bianbuai)