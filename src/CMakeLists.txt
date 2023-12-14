link_directories(${OPENCV_LIB})
link_directories(${ORT_HOME}/lib)

file(GLOB_RECURSE SRC_FILES "./*.cpp" "./*.c" "./*.cc")
add_library(bianbuai SHARED ${SRC_FILES})
set_target_properties(bianbuai PROPERTIES VERSION 1.0.1 SOVERSION 1)

target_include_directories(bianbuai PUBLIC  ${CMAKE_SOURCE_DIR}/include)
target_include_directories(bianbuai PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(bianbuai SYSTEM PUBLIC  ${OPENCV_INC})
target_include_directories(bianbuai SYSTEM PRIVATE ${ORT_HOME}/include)

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
target_link_libraries(bianbuai PRIVATE onnxruntime opencv_core opencv_imgproc opencv_dnn)

install(TARGETS bianbuai
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

# ▒~I▒~E include ▒~V~G件
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/task ${CMAKE_SOURCE_DIR}/include/utils
        DESTINATION include/bianbuai
        FILES_MATCHING PATTERN "*.h")