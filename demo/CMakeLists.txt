cmake_minimum_required(VERSION 3.10)

project(bianbuai-demo)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

# TODO: update cc files with '#ifndef NDEBUG'
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# To find OpenCV, one may need to set OpenCV_DIR variable to the
# absolute path to the directory containing OpenCVConfig.cmake file.
# Otherwise, try to set OPENCV_INC and OPENCV_LIB variables via the
# command line or GUI.
if ((NOT DEFINED OPENCV_INC OR OPENCV_INC STREQUAL "") OR ((NOT DEFINED OPENCV_LIB OR OPENCV_LIB STREQUAL "")))
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
if (OpenCV_FOUND)
  if (NOT DEFINED OPENCV_INC OR OPENCV_INC STREQUAL "")
    set(OPENCV_INC "${OpenCV_INCLUDE_DIRS}")
  endif()
  if (NOT DEFINED OPENCV_LIB OR OPENCV_LIB STREQUAL "")
    get_target_property(OpenCV_LIB_PATH opencv_core LOCATION)
    get_filename_component(OPENCV_LIB ${OpenCV_LIB_PATH} DIRECTORY)
    set(OPENCV_LIBS ${OpenCV_LIBS})
  endif()
endif()
endif()

# Check Required Env
if (NOT DEFINED BIANBUAI_HOME OR BIANBUAI_HOME STREQUAL "")
  message(FATAL_ERROR "Env 'BIANBUAI_HOME' not defined for platform ${CMAKE_GENERATOR_PLATFORM}")
endif()
if (NOT DEFINED OPENCV_INC OR OPENCV_INC STREQUAL "")
  message(FATAL_ERROR "OpenCV include dirs not found for platform ${CMAKE_GENERATOR_PLATFORM}")
endif()
if (NOT DEFINED OPENCV_LIB OR OPENCV_LIB STREQUAL "")
  message(FATAL_ERROR "OpenCV library dirs not found for platform ${CMAKE_GENERATOR_PLATFORM}")
endif()
if (NOT DEFINED ORT_HOME)
  if (EXISTS ${BIANBUAI_HOME}/lib/3rdparty/onnxruntime)
    set(ORT_HOME ${BIANBUAI_HOME}/lib/3rdparty/onnxruntime)
  else()
    message(FATAL_ERROR "Env 'ORT_HOME' not defined and OnnxRuntime library may not found for platform ${CMAKE_GENERATOR_PLATFORM}")
  endif()
endif()

include_directories(${OPENCV_INC} ${BIANBUAI_HOME}/include/bianbuai)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${BIANBUAI_HOME}/lib:${OPENCV_LIB}:${OPENCV_LIB}/3rdparty:${ORT_HOME}/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--as-needed")
link_directories(${BIANBUAI_HOME}/lib ${ORT_HOME}/lib ${OPENCV_LIB} ${OPENCV_LIB}/3rdparty)

add_executable(classification_demo image_classification_demo.cc)
target_link_libraries(classification_demo PUBLIC bianbuai opencv_core opencv_imgcodecs)

add_executable(detection_demo object_detection_demo.cc)
target_link_libraries(detection_demo PUBLIC bianbuai opencv_core opencv_imgproc opencv_imgcodecs)

add_executable(detection_video_demo object_detection_video_demo.cc)
target_link_libraries(detection_video_demo PUBLIC bianbuai opencv_core opencv_imgproc opencv_videoio opencv_highgui)

add_executable(detection_stream_demo object_detection_stream_demo.cc)
target_link_libraries(detection_stream_demo PUBLIC bianbuai pthread opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs opencv_highgui)

add_executable(estimation_demo pose_estimation_demo.cc)
target_link_libraries(estimation_demo PUBLIC bianbuai opencv_core opencv_imgproc opencv_imgcodecs)

add_executable(tracker_stream_demo pose_tracker_stream_demo.cc)
target_link_libraries(tracker_stream_demo PUBLIC bianbuai pthread opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs opencv_highgui)

install(TARGETS detection_demo classification_demo detection_stream_demo detection_video_demo estimation_demo
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
