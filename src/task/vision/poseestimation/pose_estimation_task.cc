#include "task/vision/pose_estimation_task.h"

#include "include/utils/utils.h"
#include "src/task/vision/poseestimation/pose_estimation.h"
#include "src/utils/utils.h"

class PoseEstimationTask::impl {
 public:
  std::unique_ptr<PoseEstimation> poseestimation_;
};

PoseEstimationTask::PoseEstimationTask(const PoseEstimationOption &option)
    : pimpl_(std::make_unique<impl>()) {
  init_flag_ = -1;
  pimpl_->poseestimation_ =
      std::unique_ptr<PoseEstimation>(new PoseEstimation());
  init_flag_ = pimpl_->poseestimation_->InitFromOption(option);
}

PoseEstimationTask::PoseEstimationTask(const std::string &config_file_path)
    : pimpl_(std::make_unique<impl>()) {
  init_flag_ = -1;
  pimpl_->poseestimation_ =
      std::unique_ptr<PoseEstimation>(new PoseEstimation());
  PoseEstimationOption option;
  if (!configToOption(config_file_path, option)) {
    init_flag_ = pimpl_->poseestimation_->InitFromOption(option);
  }
}

int PoseEstimationTask::getInitFlag() { return init_flag_; }

PoseEstimationResult PoseEstimationTask::Estimate(const cv::Mat &img_raw,
                                                  const Boxi &box) {
  return pimpl_->poseestimation_->Estimate(img_raw, box);
}
