cmake_minimum_required(VERSION 3.10)
 
project(bianbuai)
add_subdirectory(${CMAKE_SOURCE_DIR}/src)

option(DEMO "option for Demo" OFF)
option(XDG "option for XDG autostart support" ON)

if (DEMO)
  set(BIANBUAI_HOME ${CMAKE_SOURCE_DIR})  # useless but necessary
  add_subdirectory(${CMAKE_SOURCE_DIR}/demo)
endif()

if (DEMO OR XDG)
  # add resources for smoke test
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/rootfs/usr/share/ai-support DESTINATION share)
endif()

if (XDG)
  # add resources for xgd autostart
  install(PROGRAMS ${CMAKE_SOURCE_DIR}/rootfs/usr/bin/bianbu-ai-autotools DESTINATION bin)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/rootfs/usr/share/applications DESTINATION share)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/rootfs/usr/share/icons DESTINATION share)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/rootfs/etc DESTINATION ..)
endif()

# always install demo project with test data
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/demo DESTINATION .)
#install(DIRECTORY ${CMAKE_SOURCE_DIR}/data DESTINATION demo)

option(TEST "option for Test" ON)
if (TEST)
  add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
endif()
