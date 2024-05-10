#ifndef SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TASK_H_
#define SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TASK_H_

#include <memory>  // for: shared_ptr

#include "opencv2/opencv.hpp"
#include "task/vision/object_detection_types.h" // for: Boxi
#include "task/vision/pose_estimation_types.h"

class PoseEstimationTask {
 public:
  explicit PoseEstimationTask(const PoseEstimationOption &option);
  ~PoseEstimationTask() = default;
  int getInitFlag();
  PoseEstimationResult Estimate(const cv::Mat &img_raw, const Boxi &box);

 private:
  class impl;
  std::shared_ptr<impl> pimpl_;
  int init_flag_;
};

#endif  // SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TASK_H_
