#要求的Cmake最低版本
cmake_minimum_required(VERSION 3.10)
 
#工程名称
project(bianbuai)
#设置编译器编译模式：
set(CMAKE_BUILD_TYPE ${BUILD_TYPE})
#c++标准
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -g")


if(DEBUG)
    message("DEBUG is On")
    add_definitions(-DDEBUG)
endif()

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${ORT_HOME}/include")
#include_directories("${ORT_HOME}/include/onnxruntime/core/session")

link_directories("${ORT_HOME}/lib")
link_libraries(onnxruntime)
link_libraries(${OpenCV_LIBS})

add_subdirectory(${CMAKE_SOURCE_DIR}/src)
option(Demo "option for Demo" ON)
if (Demo)
   add_subdirectory(${CMAKE_SOURCE_DIR}/demo)
endif()

option(TEST "option for Test" ON)

if (TEST)
   add_subdirectory(${CMAKE_SOURCE_DIR}/src/tests)
endif()

# 安装 include 文件
install(DIRECTORY include/ DESTINATION include/
        FILES_MATCHING PATTERN "*.h")