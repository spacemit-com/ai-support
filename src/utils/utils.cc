#include "utils.h"

#include <math.h>  // for: exp

#include <cstdint>  // for: uint32_t
#include <fstream>  // for: ifstream
#include <string>
#include <vector>

#include "utils/utils.h"

std::vector<std::string> readLabels(const std::string& label_file_path) {
  std::vector<std::string> labels;
  std::string line;
  std::ifstream fp(label_file_path);
  while (std::getline(fp, line)) {
    labels.push_back(line);
  }
  return labels;
}

float sigmoid(float x) { return (1 / (1 + exp(-x))); }

float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = static_cast<uint32_t>((1 << 23) * (1.4426950409 * x + 126.93490512f));
  return v.f;
}

void resizeUnscale(const cv::Mat& mat, cv::Mat& mat_rs, int target_height,
                   int target_width) {
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  mat_rs = cv::Mat(target_height, target_width, CV_8UC3, cv::Scalar(0, 0, 0));
  // scale ratio (new / old) new_shape(h,w)

  float w_r = static_cast<float>(target_width) / static_cast<float>(img_width);
  float h_r =
      static_cast<float>(target_height) / static_cast<float>(img_height);
  float r = std::min(w_r, h_r);
  // compute padding
  int new_unpad_w =
      static_cast<int>(static_cast<float>(img_width) * r);  // floor
  int new_unpad_h =
      static_cast<int>(static_cast<float>(img_height) * r);  // floor
  int pad_w = target_width - new_unpad_w;                    // >=0
  int pad_h = target_height - new_unpad_h;                   // >=0

  int dw = pad_w / 2;
  int dh = pad_h / 2;

  // resize with unscaling
  cv::Mat new_unpad_mat;
  // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
  cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));

  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));
}
