if (NOT DEFINED OpenCV_SHARED OR OpenCV_SHARED STREQUAL "ON")
  message(FATAL_ERROR "Python binding requires static OpenCV libraries")
endif()

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.11
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_parties/pybind11
)

macro(FetchContent_MakeAvailable NAME)
    FetchContent_GetProperties(${NAME})
    if(NOT ${NAME}_POPULATED)
        FetchContent_Populate(${NAME})
        add_subdirectory(${${NAME}_SOURCE_DIR} ${${NAME}_BINARY_DIR})
    endif()
endmacro()
FetchContent_MakeAvailable(pybind11)

file(GLOB_RECURSE BIANBU_PYBIND_SRCS "${CMAKE_SOURCE_DIR}/python/*.cc")
list(APPEND BIANBU_PYBIND_SRCS ${BIANBU_SRC_FILES})

link_directories(${OPENCV_LIB} ${OPENCV_LIB}/3rdparty)
pybind11_add_module(bianbuai_pybind11_state ${BIANBU_PYBIND_SRCS})
target_include_directories(bianbuai_pybind11_state PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
target_include_directories(bianbuai_pybind11_state SYSTEM PRIVATE ${OPENCV_INC})
target_link_libraries(bianbuai_pybind11_state PRIVATE ${TARGET_SHARED_LINKER_FLAGS})
target_link_libraries(bianbuai_pybind11_state PRIVATE ${SPACEMITEP_LIB} onnxruntime ${OPENCV_LIBS})
