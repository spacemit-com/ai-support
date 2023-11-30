add_executable(classification_demo image_classification_demo.cc)
target_include_directories(classification_demo PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(classification_demo PUBLIC bianbuai)

add_executable(detection_demo object_detection_demo.cc)
target_include_directories(detection_demo PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(detection_demo PUBLIC bianbuai)

add_executable(detection_stream_demo object_detection_stream_demo.cc)
target_include_directories(detection_stream_demo PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(detection_stream_demo PUBLIC bianbuai)

add_executable(json_demo json_demo.cc)
target_include_directories(json_demo PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(json_demo PUBLIC bianbuai)
