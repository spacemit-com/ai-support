#要求的Cmake最低版本
cmake_minimum_required(VERSION 3.10)
 
#工程名称
project(bianbuai)
#c++标准
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

# TODO: update cc files with '#ifndef NDEBUG'
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

# Set if use spacemit-ep
option(ENABLE_SPACEMIT_EP "option to enable spacemit ep" OFF)

# To find OpenCV, you may need to set OpenCV_DIR variable to the
# absolute path to the directory containing OpenCVConfig.cmake file
# or OPENCV_INC and OPENCV_LIB variables via the command line or GUI.
find_package(OpenCV)
if(OpenCV_FOUND)
  if(NOT DEFINED OPENCV_INC OR OPENCV_INC STREQUAL "")
    set(OPENCV_INC ${OpenCV_INCLUDE_DIRS})
  endif()
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

option(Demo "option for Demo" ON)
if (Demo)
   add_subdirectory(${CMAKE_SOURCE_DIR}/demo)
endif()

option(TEST "option for Test" ON)
if (TEST)
   add_subdirectory(${CMAKE_SOURCE_DIR}/src/tests)
endif()