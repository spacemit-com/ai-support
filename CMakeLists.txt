#要求的Cmake最低版本
CMAKE_MINIMUM_REQUIRED( VERSION 3.10)
 
#工程名称
PROJECT(bianbuai)

#设置编译器编译模式：
set(CMAKE_BUILD_TYPE "Debug" )
#c++标准
set(CMAKE_CXX_STANDARD 11)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

set(HOME_ROOTDIR "${ROOT_DIR}/support")
set(ONNXRUNTIME_ROOTDIR "${ORT_ROOT_DIR}")


# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session") # Windows local install

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
link_libraries(onnxruntime)
link_libraries(${OpenCV_LIBS})

add_subdirectory(${HOME_ROOTDIR}/src)
option(Demo "option for Demo" ON)
if (Demo)
   add_subdirectory(${HOME_ROOTDIR}/demo)
endif()