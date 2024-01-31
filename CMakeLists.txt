cmake_minimum_required(VERSION 3.10)
 
project(bianbuai)
add_subdirectory(${CMAKE_SOURCE_DIR}/src)

option(DEMO "option for Demo" OFF)
if (DEMO)
  set(BIANBUAI_HOME ${CMAKE_SOURCE_DIR})  # useless but necessary
  add_subdirectory(${CMAKE_SOURCE_DIR}/demo)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/rootfs/usr/bin DESTINATION .)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/rootfs/usr/share DESTINATION .)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/rootfs/etc DESTINATION ..)
endif()
# always install demo project with test data
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/demo DESTINATION .)
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/data DESTINATION demo)

option(TEST "option for Test" ON)
if (TEST)
  add_subdirectory(${CMAKE_SOURCE_DIR}/src/tests)
endif()