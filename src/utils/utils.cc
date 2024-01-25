#include "utils/utils.h"

#include <cmath>
#include <cstdint>  // for: uint32_t
#include <fstream>
#include <string>
#include <vector>

#include "include/utils/utils.h"
#include "src/utils/json.hpp"
#include "src/utils/utils.h"
using json = nlohmann::json;

bool checkLabelFileExtension(const std::string& filename) {
  size_t pos = filename.rfind('.');
  if (filename.empty()) {
    std::cout << "[ ERROR ] The Label file path is empty" << std::endl;
    return false;
  }
  if (pos == std::string::npos) return false;
  std::string ext = filename.substr(pos + 1);
  if (ext == "txt") {
    return true;
  } else {
    return false;
  }
}

std::vector<std::string> readLabels(const std::string& labelFilepath) {
  std::vector<std::string> labels;
  std::string line;
  std::ifstream fp(labelFilepath);
  while (std::getline(fp, line)) {
    labels.push_back(line);
  }
  return labels;
}

bool checkModelFileExtension(const std::string& filename) {
  size_t pos = filename.rfind('.');
  if (filename.empty()) {
    std::cout << "[ ERROR ] The Model file path is empty" << std::endl;
    return false;
  }
  if (pos == std::string::npos) return false;
  std::string ext = filename.substr(pos + 1);
  if (ext == "onnx") {
    return true;
  } else {
    return false;
  }
}

int checkConfigFileExtension(const std::string& filename) {
  size_t pos = filename.rfind('.');
  if (filename.empty()) {
    std::cout << "[ ERROR ] The Config file path is empty" << std::endl;
    return false;
  }
  if (pos == std::string::npos) return false;
  std::string ext = filename.substr(pos + 1);
  if (ext == "json") {
    return true;
  } else {
    return false;
  }
}

int configCheck(const json& config) {
  if (!config.contains("model_path") || !config.contains("label_path")) {
    return 1;
  } else if (!checkModelFileExtension(config["model_path"]) ||
             !checkLabelFileExtension(config["label_path"])) {
    return 1;
  } else if (!exists_check(config["model_path"]) ||
             !exists_check(config["label_path"])) {
    return 1;
  } else {
    return 0;
  }
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

bool exists_check(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

void resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs, int target_height,
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
