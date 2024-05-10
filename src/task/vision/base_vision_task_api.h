#ifndef _BASE_VISION_TASK_API_H_
#define _BASE_VISION_TASK_API_H_

#include "opencv2/opencv.hpp"
#include "src/task/core/base_task_api.h"
#include "task/vision/object_detection_types.h"

template <class OutputType>
class BaseVisionTaskApi : public BaseTaskApi<OutputType, const cv::Mat&> {
 public:
  BaseVisionTaskApi() : BaseTaskApi<OutputType, const cv::Mat&>() {}
  ~BaseVisionTaskApi() {}
  // BaseVisionTaskApi is neither copyable nor movable.
  BaseVisionTaskApi(const BaseVisionTaskApi&) = delete;
  BaseVisionTaskApi& operator=(const BaseVisionTaskApi&) = delete;

 protected:
  virtual void Preprocess(const cv::Mat& img_raw) = 0;
};

#endif
