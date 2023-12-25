#ifndef SUPPORT_DEMO_UTILS_CHECK_UTILS_H_
#define SUPPORT_DEMO_UTILS_CHECK_UTILS_H_

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

#endif  // SUPPORT_DEMO_UTILS_CHECK_UTILS_H_
