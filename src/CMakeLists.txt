file(GLOB_RECURSE SRC_FILES "./*.cpp" "./*.c" "./*.cc")

add_library(bianbuai SHARED ${SRC_FILES})
target_include_directories(bianbuai PUBLIC "${CMAKE_SOURCE_DIR}")
target_include_directories(bianbuai PUBLIC "${CMAKE_SOURCE_DIR}/include")