#ifndef SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TASK_H_
#define SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TASK_H_

#include <memory>  // for: shared_ptr

#include "opencv2/opencv.hpp"
#include "task/vision/image_classification_types.h"

class ImageClassificationTask {
 public:
  explicit ImageClassificationTask(const ImageClassificationOption& option);
  ~ImageClassificationTask() = default;
  ImageClassificationResult Classify(const cv::Mat& img_raw);
  int getInitFlag();

 private:
  class impl;
  std::shared_ptr<impl> pimpl_;
  int init_flag_;
};

#endif  // SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TASK_H_
