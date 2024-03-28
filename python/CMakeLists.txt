cmake_minimum_required(VERSION 3.11)
project(support)

macro(FetchContent_MakeAvailable NAME)
    FetchContent_GetProperties(${NAME})
    if(NOT ${NAME}_POPULATED)
        FetchContent_Populate(${NAME})
        add_subdirectory(${${NAME}_SOURCE_DIR} ${${NAME}_BINARY_DIR})
    endif()
endmacro()

include(FetchContent)
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

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_parties/pybind11
)

FetchContent_MakeAvailable(pybind11)
link_directories(${BIANBUAI_HOME}/lib ${OPENCV_LIB} ${OPENCV_LIB}/3rdparty)
include_directories(${OPENCV_INC} ${BIANBUAI_HOME}/include)
pybind11_add_module(support wrapper.cc)
target_link_libraries(support PRIVATE bianbuai opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_highgui)
