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

int getConfig(const std::string& config_file_path, json& config) {
  std::ifstream f(config_file_path);
  try {
    config = json::parse(f);
  } catch (json::parse_error& ex) {
    std::cout << "[ ERROR ]  Init fail, parse json config file fail"
              << std::endl;
    return 0;
  }
  return 1;
}

int configToOption(const std::string& config_file_path,
                   ImageClassificationOption& option) {
  json config;
  if (!getConfig(config_file_path, config)) {
    return -1;
  }
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
  json config;
  if (!getConfig(config_file_path, config)) {
    return -1;
  }
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
  json config;
  if (!getConfig(config_file_path, config)) {
    return -1;
  }
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
