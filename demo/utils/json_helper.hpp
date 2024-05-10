#ifndef SUPPORT_DEMO_UTILS_JSON_HELPER_HPP_
#define SUPPORT_DEMO_UTILS_JSON_HELPER_HPP_

#include <fstream>
#include <iostream>
#include <string>

#include "json.hpp"
#include "task/vision/image_classification_types.h"
#include "task/vision/object_detection_types.h"
#include "task/vision/pose_estimation_types.h"

using json = nlohmann::json;

static int getConfig(const char* config_file_path, json& config) {
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

int configToOption(const char* config_file_path,
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

int configToOption(const char* config_file_path,
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

int configToOption(const char* config_file_path, PoseEstimationOption& option) {
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

#endif  // SUPPORT_DEMO_UTILS_JSON_HELPER_HPP_
