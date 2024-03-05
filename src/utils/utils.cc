#include "utils/utils.h"

#include <cmath>
#include <cstdint>  // for: uint32_t
#include <fstream>  // for ifstream
#include <iostream>
#include <string>
#include <vector>

#include "include/utils/utils.h"
#include "src/utils/json.hpp"
#include "src/utils/utils.h"
using json = nlohmann::json;

bool startsWith(const std::string& str, const std::string& prefix) {
  return (str.rfind(prefix, 0) == 0);
}

bool endsWith(const std::string& str, const std::string& suffix) {
  if (suffix.length() > str.length()) {
    return false;
  }
  return (str.rfind(suffix) == (str.length() - suffix.length()));
}

std::vector<std::string> readLabels(const std::string& label_file_path) {
  std::vector<std::string> labels;
  std::string line;
  std::ifstream fp(label_file_path);
  while (std::getline(fp, line)) {
    labels.push_back(line);
  }
  return labels;
}

bool checkFileExtension(const std::string& filename, const std::string& ext) {
  if (endsWith(filename, ext)) {
    return true;
  }
  std::cout << "[ ERROR ] The file path " << filename
            << " is not correct. Make sure you "
               "are setting the path to a file ("
            << ext << ")" << std::endl;
  return false;
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

bool existsCheck(const std::string& name) {
  struct stat buffer;
  if (stat(name.c_str(), &buffer) == 0) {
    return true;
  } else {
    std::cout << "[ ERROR ] The file " << name
              << " does not exist. Make sure you are "
                 "setting the correct path to the file"
              << std::endl;
    return false;
  }
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

int configToOption(const std::string& config_file_path,
                   ImageClassificationOption& option) {
  if (!checkFileExtension(config_file_path, ".json") &&
      !existsCheck(config_file_path)) {
    return 1;
  }
  std::ifstream f(config_file_path);
  json config = json::parse(f);
  std::string model_path = config["model_path"];
  option.model_path = model_path;
  std::string label_path = config["label_path"];
  option.label_path = label_path;
  if (config.contains("intra_threads_num")) {
    option.intra_threads_num = config["intra_threads_num"];
  }
  if (config.contains("inter_threads_num")) {
    option.inter_threads_num = config["inter_threads_num"];
  }
  return 0;
}

int configToOption(const std::string& config_file_path,
                   ObjectDetectionOption& option) {
  if (!checkFileExtension(config_file_path, "json") &&
      !existsCheck(config_file_path)) {
    return 1;
  }
  std::ifstream f(config_file_path);
  json config = json::parse(f);
  std::string model_path = config["model_path"];
  option.model_path = model_path;
  std::string label_path = config["label_path"];
  option.label_path = label_path;
  if (config.contains("intra_threads_num")) {
    option.intra_threads_num = config["intra_threads_num"];
  }
  if (config.contains("inter_threads_num")) {
    option.inter_threads_num = config["inter_threads_num"];
  }
  if (config.contains("score_threshold")) {
    option.score_threshold = config["score_threshold"];
  }
  if (config.contains("nms_threshold")) {
    option.nms_threshold = config["nms_threshold"];
  }
  if (config.contains("class_name_whitelist")) {
    option.class_name_whitelist =
        config["class_name_whitelist"].get<std::vector<int>>();
  }
  if (config.contains("class_name_blacklist")) {
    option.class_name_blacklist =
        config["class_name_blacklist"].get<std::vector<int>>();
  }
  return 0;
}

int configToOption(const std::string& config_file_path,
                   PoseEstimationOption& option) {
  if (!checkFileExtension(config_file_path, ".json") &&
      !existsCheck(config_file_path)) {
    return 1;
  }
  std::ifstream f(config_file_path);
  json config = json::parse(f);
  std::string model_path = config["model_path"];
  option.model_path = model_path;
  if (config.contains("intra_threads_num")) {
    option.intra_threads_num = config["intra_threads_num"];
  }
  if (config.contains("inter_threads_num")) {
    option.inter_threads_num = config["inter_threads_num"];
  }
  return 0;
}
