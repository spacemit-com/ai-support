#ifndef SUPPORT_INCLUDE_UTILS_UTILS_H_
#define SUPPORT_INCLUDE_UTILS_UTILS_H_

#include <sys/stat.h>

#include <string>

#include "opencv2/opencv.hpp"

static bool exists_check(const std::string &name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

static void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                           int target_height, int target_width) {
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  mat_rs = cv::Mat(target_height, target_width, CV_8UC3, cv::Scalar(0, 0, 0));
  // scale ratio (new / old) new_shape(h,w)

  float w_r = static_cast<float>(target_width) / static_cast<float>(img_width);
  float h_r = (float)target_height / (float)img_height;
  float r = std::min(w_r, h_r);
  // compute padding
  int new_unpad_w = static_cast<int>((float)img_width * r);   // floor
  int new_unpad_h = static_cast<int>((float)img_height * r);  // floor
  int pad_w = target_width - new_unpad_w;                     // >=0
  int pad_h = target_height - new_unpad_h;                    // >=0

  int dw = pad_w / 2;
  int dh = pad_h / 2;

  // resize with unscaling
  cv::Mat new_unpad_mat;
  // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
  cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));

  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));
}

#endif  // SUPPORT_INCLUDE_UTILS_UTILS_H_
