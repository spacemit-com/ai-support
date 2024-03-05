#ifndef SUPPORT_DEMO_UTILS_CHECK_UTILS_H_
#define SUPPORT_DEMO_UTILS_CHECK_UTILS_H_
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

static bool checkImageFileExtension(const std::string& filename) {
  size_t pos = filename.rfind('.');
  std::string ext;
  if (pos != std::string::npos) {
    ext = filename.substr(pos + 1);
  }
  if (ext == "jpeg" || ext == "jpg" || ext == "png") {
    return true;
  }
  std::cout << "[ ERROR ] The image file path " << filename
            << " is not correct. Make sure you "
               "are setting the path to an imgae file (.jpg/.jpeg/.png)"
            << std::endl;
  return false;
}

#endif  // SUPPORT_DEMO_UTILS_CHECK_UTILS_H_
