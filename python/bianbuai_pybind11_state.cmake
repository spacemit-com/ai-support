if (NOT DEFINED OpenCV_SHARED OR OpenCV_SHARED STREQUAL "ON")
  message(WARNING "Python binding suggests to link static OpenCV libraries")
endif()

file(GLOB_RECURSE BIANBU_PYBIND_SRCS "${CMAKE_SOURCE_DIR}/python/*.cc")
list(APPEND BIANBU_PYBIND_SRCS ${BIANBU_SRC_FILES})

pybind11_add_module(bianbuai_pybind11_state ${BIANBU_PYBIND_SRCS})
target_include_directories(bianbuai_pybind11_state PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR})
target_include_directories(bianbuai_pybind11_state SYSTEM PRIVATE ${OPENCV_INC})
target_include_directories(bianbuai_pybind11_state SYSTEM PRIVATE ${ORT_HOME}/include ${ORT_HOME}/include/onnxruntime)
target_link_libraries(bianbuai_pybind11_state PRIVATE ${HIDE_SYMBOLS_LINKER_FLAGS})
target_link_libraries(bianbuai_pybind11_state PRIVATE ${TARGET_SHARED_LINKER_FLAGS})
target_link_libraries(bianbuai_pybind11_state PRIVATE ${SPACEMITEP_LIB} onnxruntime ${OPENCV_LIBS})

install(TARGETS bianbuai_pybind11_state
        RUNTIME COMPONENT pybind11 DESTINATION bin
        LIBRARY COMPONENT pybind11 DESTINATION lib
        ARCHIVE COMPONENT pybind11 DESTINATION lib)