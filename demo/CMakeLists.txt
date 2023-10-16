#要求的Cmake最低版本
CMAKE_MINIMUM_REQUIRED( VERSION 3.10)

add_executable(classify_demo image_classify_demo.cc)
target_include_directories(classify_demo PUBLIC "${HOME_ROOTDIR}/include")
target_link_libraries(classify_demo PUBLIC bianbuai)

add_executable(detection_demo object_detection_demo.cc)
target_include_directories(detection_demo PUBLIC "${HOME_ROOTDIR}/include")
target_link_libraries(detection_demo PUBLIC bianbuai)