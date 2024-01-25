file(READ ${CMAKE_SOURCE_DIR}/VERSION_NUMBER VERSION_CONTENT)
string(STRIP "${VERSION_CONTENT}" VERSION_CONTENT)
project(bianbuai-support-library VERSION ${VERSION_CONTENT})

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")

if (WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

# TODO: update cc files with '#ifndef NDEBUG'
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

set(OPENCV_LIBS opencv_core opencv_imgproc)
# To find OpenCV, one may need to set OpenCV_DIR variable to the
# absolute path to the directory containing OpenCVConfig.cmake file.
# Otherwise, try to set OPENCV_INC and OPENCV_LIB variables via the
# command line or GUI.
if ((NOT DEFINED OPENCV_INC OR OPENCV_INC STREQUAL "") OR ((NOT DEFINED OPENCV_LIB OR OPENCV_LIB STREQUAL "")))
find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_LIBS})
if (OpenCV_FOUND)
  if (NOT DEFINED OPENCV_INC OR OPENCV_INC STREQUAL "")
    set(OPENCV_INC "${OpenCV_INCLUDE_DIRS}")
  endif()
  if (NOT DEFINED OPENCV_LIB OR OPENCV_LIB STREQUAL "")
    get_target_property(OpenCV_LIB_PATH opencv_core LOCATION)
    get_filename_component(OPENCV_LIB ${OpenCV_LIB_PATH} DIRECTORY)
  endif()
endif()
endif()

# Check Required Env
if (NOT DEFINED ORT_HOME OR ORT_HOME STREQUAL "")
  message(FATAL_ERROR "Env 'ORT_HOME' not defined for platform ${CMAKE_GENERATOR_PLATFORM}")
endif()
if (NOT DEFINED OPENCV_INC OR OPENCV_INC STREQUAL "")
  message(FATAL_ERROR "OpenCV include dirs not found for platform ${CMAKE_GENERATOR_PLATFORM}")
endif()
if (NOT DEFINED OPENCV_LIB OR OPENCV_LIB STREQUAL "")
  message(FATAL_ERROR "OpenCV library dirs not found for platform ${CMAKE_GENERATOR_PLATFORM}")
endif()

link_directories(${OPENCV_LIB})
link_directories(${ORT_HOME}/lib)

file(GLOB_RECURSE SRC_FILES "./*.cpp" "./*.c" "./*.cc")
add_library(bianbuai SHARED ${SRC_FILES})
set_target_properties(bianbuai PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

if (EXISTS "${ORT_HOME}/lib/libspacemit_ep.so")
  add_definitions(-DHAS_SPACEMIT_EP)
  set(SPACEMITEP_LIB "spacemit_ep")
endif()

target_include_directories(bianbuai PUBLIC  ${CMAKE_SOURCE_DIR}/include)
target_include_directories(bianbuai PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(bianbuai SYSTEM PUBLIC  ${OPENCV_INC})
target_include_directories(bianbuai SYSTEM PRIVATE ${ORT_HOME}/include ${ORT_HOME}/include/onnxruntime)

set(HIDE_SYMBOLS_LINKER_FLAGS "-Wl,--exclude-libs,ALL")
# Note: 'target_link_options' with 'PRIVATE' keyword would be cleaner
# but it's not available until CMake 3.13. Switch to 'target_link_options'
# once minimum CMake version is bumped up to 3.13 or above.
target_link_libraries(bianbuai PRIVATE ${HIDE_SYMBOLS_LINKER_FLAGS})

set(TARGET_SHARED_LINKER_FLAGS "-Wl,--as-needed")
# Note: As modern cmake suggests to give the preference to the property settings before the global ones,
# so instead of 'CMAKE_SHARED_LINKER_FLAGS' that has a global scope, the 'target_link_options' or
# 'target_link_libraries' should be used here:
#set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed")
target_link_libraries(bianbuai PRIVATE ${TARGET_SHARED_LINKER_FLAGS})
target_link_libraries(bianbuai PRIVATE ${SPACEMITEP_LIB} onnxruntime ${OPENCV_LIBS})

install(TARGETS bianbuai
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/task ${CMAKE_SOURCE_DIR}/include/utils
        DESTINATION include/bianbuai
        FILES_MATCHING PATTERN "*.h")

#install(DIRECTORY ${ORT_HOME}/include ${ORT_HOME}/lib
#        DESTINATION lib/3rdparty/onnxruntime
#        FILES_MATCHING PATTERN "*.h" PATTERN "*onnxruntime*.so*" PATTERN "*spacemit*.so*")
#if (EXISTS "${ORT_HOME}/bin")
#  install(DIRECTORY ${ORT_HOME}/bin DESTINATION lib/3rdparty/onnxruntime)
#endif()