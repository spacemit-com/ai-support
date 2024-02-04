#include "include/utils/utils.h"
#include "src/task/vision/poseestimation/pose_estimation.h"
#include "src/utils/utils.h"
#include "task/vision/pose_estimation_task.h"

class poseEstimationTask::impl {
 public:
  std::unique_ptr<PoseEstimation> poseestimation_;
};

poseEstimationTask::poseEstimationTask(const std::string &filePath)
    : pimpl_(std::make_unique<impl>()) {
  init_flag_ = -1;
  pimpl_->poseestimation_ =
      std::unique_ptr<PoseEstimation>(new PoseEstimation());
  if (filePath.length() > 4) {
    std::string suffixStr = filePath.substr(filePath.length() - 4, 4);
    if (strcmp(suffixStr.c_str(), "onnx") == 0) {
      if (!checkModelFileExtension(filePath)) {
        std::cout << "[ ERROR ] The ModelFilepath is not correct. Make sure "
                     "you are setting the path to an onnx model file (.onnx)"
                  << std::endl;
      } else if (!exists_check(filePath)) {
        std::cout << "[ ERROR ] The File does not exist. Make sure you are "
                     "setting the correct path to the file"
                  << std::endl;
      } else {
        init_flag_ = pimpl_->poseestimation_->InitFromCommand(filePath);
        if (init_flag_ != 0) {
          std::cout << "[Error] Init fail" << std::endl;
        }
      }
    } else if (strcmp(suffixStr.c_str(), "json") == 0) {
      if (!checkConfigFileExtension(filePath)) {
        std::cout << "[ ERROR ] The ConfigFilepath is not correct. Make sure "
                     "you are setting the path to an json file (.json)"
                  << std::endl;
      } else if (!exists_check(filePath)) {
        std::cout << "[ ERROR ] The File does not exist. Make sure you are "
                     "setting the correct path to the file"
                  << std::endl;
      } else {
        init_flag_ = pimpl_->poseestimation_->InitFromConfig(filePath);
        if (init_flag_ != 0) {
          std::cout << "[Error] Init fail" << std::endl;
        }
      }
    } else {
      std::cout << "[ ERROR ] Unsupport file" << std::endl;
    }
  } else {
    std::cout << "[ ERROR ] Unsupport filepath" << std::endl;
  }
}

int poseEstimationTask::getInitFlag() { return init_flag_; }

PoseEstimationResult poseEstimationTask::Estimate(const cv::Mat &raw_img,
                                                  const Boxi &box) {
  return pimpl_->poseestimation_->Estimate(raw_img, box);
}
