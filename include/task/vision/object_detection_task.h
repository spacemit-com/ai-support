#ifndef SUPPORT_INCLUDE_TASK_VISION_OBJECT_DETECTION_TASK_H_
#define SUPPORT_INCLUDE_TASK_VISION_OBJECT_DETECTION_TASK_H_

#include <memory>  // for: shared_ptr
#include <string>
#include <vector>  //for: vector

#include "opencv2/opencv.hpp"
#include "task/vision/object_detection_types.h"

class ObjectDetectionTask {
 public:
  explicit ObjectDetectionTask(const std::string &config_file_path);
  explicit ObjectDetectionTask(const ObjectDetectionOption &option);
  ~ObjectDetectionTask() = default;
  ObjectDetectionResult Detect(const cv::Mat &img_raw);
  ObjectDetectionResult Detect(
      const std::vector<std::vector<float>> &input_tensors,
      const int img_height, const int img_width);
  std::vector<std::vector<float>> Process(const cv::Mat &img_raw);
  int getInitFlag();

 private:
  class impl;
  std::shared_ptr<impl> pimpl_;
  int init_flag_;
};

#endif  // SUPPORT_INCLUDE_TASK_VISION_OBJECT_DETECTION_TASK_H_
