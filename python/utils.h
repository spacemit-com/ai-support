#ifndef SUPPORT_PYTHON_UTILS_H_
#define SUPPORT_PYTHON_UTILS_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "opencv2/opencv.hpp"
namespace py = pybind11;
struct Box {
  int x1;
  int y1;
  int x2;
  int y2;
  float score;
  const char *label_text;
  unsigned int label;
};

cv::Mat numpy_uint8_3c_to_cv_mat(const py::array_t<unsigned char> &input) {
  py::buffer_info buf = input.request();
  cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char *)buf.ptr);
  return mat;
}

#endif  // SUPPORT_PYTHON_UTILS_H_