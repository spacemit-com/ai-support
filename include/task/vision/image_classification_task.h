#ifndef SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TASK_H_
#define SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TASK_H_

#include <memory>  // for: shared_ptr
#include <string>

#include "opencv2/opencv.hpp"
#include "task/vision/image_classification_types.h"

class imageClassificationTask {
 public:
  imageClassificationTask(const std::string& filePath,
                          const std::string& labelFilepath,
                          const bool& disable_spacemit_ep = false,
                          const int& intra_threads_num = 4);
  ~imageClassificationTask() = default;
  ImageClassificationResult Classify(const cv::Mat& img_raw);

 private:
  class impl;
  std::shared_ptr<impl> pimpl_;
};

#endif  // SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TASK_H_
