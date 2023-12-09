set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${OPENCV_LIB}:${ORT_HOME}/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")
#add_link_options("LINKER:-rpath-link,${OPENCV_LIB}:${ORT_HOME}/lib")
#add_link_options("LINKER:--as-needed")
link_directories(${OPENCV_LIB})

add_executable(classification_demo image_classification_demo.cc)
#target_include_directories(classification_demo PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(classification_demo PUBLIC bianbuai opencv_core opencv_imgcodecs)


add_executable(detection_demo object_detection_demo.cc)
#target_include_directories(detection_demo PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(detection_demo PUBLIC bianbuai opencv_core opencv_imgproc opencv_imgcodecs)

add_executable(detection_stream_demo object_detection_stream_demo.cc)
#target_include_directories(detection_stream_demo PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(detection_stream_demo PUBLIC bianbuai pthread opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs opencv_highgui)

add_executable(detection_video_demo object_detection_video_demo.cc)
#target_include_directories(detection_video_demo PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(detection_video_demo PUBLIC bianbuai opencv_core opencv_imgproc opencv_videoio opencv_highgui)

install(TARGETS detection_demo classification_demo detection_stream_demo detection_video_demo
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
