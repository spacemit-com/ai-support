#ifndef SUPPORT_INCLUDE_TASK_VISION_OBJECT_DETECTION_TASK_H_
#define SUPPORT_INCLUDE_TASK_VISION_OBJECT_DETECTION_TASK_H_

#include <memory>  // for: shared_ptr
#include <string>
#include <vector>  //for: vector

#include "opencv2/opencv.hpp"
#include "task/vision/object_detection_types.h"

class objectDetectionTask {
 public:
  objectDetectionTask(const std::string &filePath,
                      const std::string &labelFilepath,
                      const bool &disable_spacemit_ep = false,
                      const int &intra_threads_num = 4,
                      const float &score_threshold = -1.f,
                      const float &nms_threshold = -1.f);
  objectDetectionTask(const std::string &filePath);
  ~objectDetectionTask() = default;
  ObjectDetectionResult Detect(const cv::Mat &raw_img);
  ObjectDetectionResult Detect(
      const std::vector<std::vector<float>> &input_tensors,
      const int img_height, const int img_width);
  std::vector<std::vector<float>> Process(const cv::Mat &img_raw);

 private:
  class impl;
  std::shared_ptr<impl> pimpl_;
};

#endif  // SUPPORT_INCLUDE_TASK_VISION_OBJECT_DETECTION_TASK_H_
