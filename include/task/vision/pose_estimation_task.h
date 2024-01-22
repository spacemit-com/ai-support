#ifndef SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TASK_H_
#define SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TASK_H_

#include <memory>  // for: shared_ptr
#include <string>

#include "opencv2/opencv.hpp"
#include "task/vision/object_detection_types.h"
#include "task/vision/pose_estimation_types.h"

class poseEstimationTask {
 public:
  poseEstimationTask(const std::string &filePath,
                     const bool &disable_spacemit_ep = false,
                     const int &intra_threads_num = 4);
  poseEstimationTask(const std::string &filePath);
  ~poseEstimationTask() = default;
  PoseEstimationResult Estimate(const cv::Mat &raw_img, const Boxi &box);

 private:
  class impl;
  std::shared_ptr<impl> pimpl_;
};

#endif  // SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TASK_H_
