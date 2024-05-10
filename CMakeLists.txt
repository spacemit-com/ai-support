cmake_minimum_required(VERSION 3.10)

file(READ ${CMAKE_SOURCE_DIR}/VERSION_NUMBER VERSION_CONTENT)
string(STRIP "${VERSION_CONTENT}" VERSION_CONTENT)
project(bianbuai-support-library VERSION ${VERSION_CONTENT})

option(PYTHON "Option for Python" OFF)
option(DEMO   "Option for Demo"   OFF)
option(XDG    "Option for XDG autostart support"  OFF)

if (PYTHON)
  add_subdirectory(${CMAKE_SOURCE_DIR}/pybind11)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/src)

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
