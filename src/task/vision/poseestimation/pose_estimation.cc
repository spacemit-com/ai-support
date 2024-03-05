#include "src/task/vision/poseestimation/pose_estimation.h"

#include <chrono>

#include "src/utils/json.hpp"
#include "utils/time.h"

using json = nlohmann::json;

PoseEstimationResult PoseEstimation::Estimate(const cv::Mat &img_raw,
                                              const Boxi &box) {
  result_points_.clear();
  input_tensors_.clear();
  box_ = box;
  {
#ifdef DEBUG
    std::cout << "|-- Preprocess" << std::endl;
    TimeWatcher t("|--");
#endif
    Preprocess(img_raw);
  }
  return Postprocess();
}

void PoseEstimation::Preprocess(const cv::Mat &img_raw) {
  if (init_flag_ != 0) {
    return;
  }
  if (option_.model_path.find("rtmpose") != option_.model_path.npos) {
    processor_.Preprocess(img_raw, box_, input_tensors_, crop_result_pair_,
                          CHW);
  } else {
    std::cout << "[ ERROR ] Unsupported model" << std::endl;
  }
}

PoseEstimationResult PoseEstimation::Postprocess() {
  if (init_flag_ != 0) {
    std::cout << "[ ERROR ] Init fail" << std::endl;
    result_.result_points = result_points_;
    result_.timestamp = std::chrono::steady_clock::now();
    return result_;
  }
  if (option_.model_path.find("rtmpose") != option_.model_path.npos) {
    postprocessor_.Postprocess(Infer(input_tensors_), crop_result_pair_,
                               result_points_);
  } else {
    std::cout << "[ ERROR ] Unsupported model" << std::endl;
  }
  result_.result_points = result_points_;
  result_.timestamp = std::chrono::steady_clock::now();
  return result_;
}

int PoseEstimation::InitFromOption(const PoseEstimationOption &option) {
  init_flag_ = 1;
  instance_name_ = "pose-estimation-inference";
  option_ = option;
  init_flag_ =
      GetEngine()->Init(instance_name_, option_.model_path,
                        option.intra_threads_num, option.inter_threads_num);
  return init_flag_;
}
