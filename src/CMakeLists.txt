file(GLOB_RECURSE SRC_FILES "./*.cpp" "./*.c" "./*.cc")

set(onnxruntime "/home/gexy5/Documents/spacemit_ort.v0.8.0/onnxruntime/lib/libonnxruntime.so")
add_library(bianbuai SHARED ${SRC_FILES} )
target_include_directories(bianbuai PUBLIC "${CMAKE_SOURCE_DIR}/include"  "${CMAKE_SOURCE_DIR}")

install(TARGETS bianbuai
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)