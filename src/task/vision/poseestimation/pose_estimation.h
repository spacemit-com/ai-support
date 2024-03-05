#ifndef SUPPORT_SRC_TASK_VISION_POSEESTIMATION_POSE_ESTIMATION_H_
#define SUPPORT_SRC_TASK_VISION_POSEESTIMATION_POSE_ESTIMATION_H_

#include <iostream>
#include <string>
#include <utility>  // for pair<>
#include <vector>

#include "opencv2/opencv.hpp"
#include "src/core/engine.h"
#include "src/processor/estimation_postprocessor.h"
#include "src/processor/estimation_preprocessor.h"
#include "src/task/vision/base_vision_task_api.h"
#include "src/utils/cv2_utils.h"
#include "task/vision/pose_estimation_types.h"

class PoseEstimation : public BaseVisionTaskApi<PoseEstimationResult> {
 public:
  PoseEstimation() : BaseVisionTaskApi<PoseEstimationResult>() {
    init_flag_ = -1;
  }
  ~PoseEstimation() {}
  PoseEstimationResult Estimate(const cv::Mat &img_raw, const Boxi &box);
  int InitFromOption(const PoseEstimationOption &option);

 protected:
  void Preprocess(const cv::Mat &img_raw) override;
  PoseEstimationResult Postprocess() override;

 private:
  std::string instance_name_;
  PoseEstimationOption option_;
  std::vector<std::vector<float>> input_tensors_;
  EstimationPreprocessor processor_;
  EstimationPostprocessor postprocessor_;
  std::vector<PosePoint> result_points_;
  PoseEstimationResult result_;
  Boxi box_;
  int init_flag_;
  std::pair<cv::Mat, cv::Mat> crop_result_pair_;
};

#endif  // SUPPORT_SRC_TASK_VISION_POSEESTIMATION_POSE_ESTIMATION_H_
