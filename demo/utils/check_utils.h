#ifndef SUPPORT_DEMO_UTILS_CHECK_UTILS_H_
#define SUPPORT_DEMO_UTILS_CHECK_UTILS_H_

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>  //for close

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

static bool checkImageFileExtension(const std::string& filename) {
  size_t pos = filename.rfind('.');
  if (filename.empty()) {
    std::cout << "[ ERROR ] The Image file path is empty" << std::endl;
    return false;
  }
  if (pos == std::string::npos) return false;
  std::string ext = filename.substr(pos + 1);
  if (ext == "jpeg" || ext == "jpg" || ext == "png") {
    return true;
  } else {
    return false;
  }
}

static bool isNumber(const std::string& str) {
  for (char const& c : str) {
    if (std::isdigit(c) == 0) return false;
  }
  return true;
}

static bool is_valid_camera(const std::string& path) {
  int fd = open(path.c_str(), O_RDWR);
  if (fd == -1) {
    std::cerr << "Cannot open " << path << std::endl;
    return false;
  }
  struct v4l2_capability cap;
  if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
    std::cerr << "Cannot query capabilities of " << path << std::endl;
    close(fd);
    return false;
  }
  close(fd);
  return (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) != 0;
}

#endif  // SUPPORT_DEMO_UTILS_CHECK_UTILS_H_
