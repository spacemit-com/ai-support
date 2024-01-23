#include "src/task/vision/poseestimation/pose_estimation.h"

#include <chrono>
#include <fstream>

#include "src/utils/json.hpp"
#include "utils/time.h"

using json = nlohmann::json;

void PoseEstimation::Preprocess(std::vector<std::vector<float>> &input_tensors,
                                const cv::Mat &img_raw) {
  Boxi box;
  processor_.Preprocess(img_raw, box, inputDims_, input_tensors_,
                        crop_result_pair_, CHW);
}

PoseEstimationResult PoseEstimation::Estimate(const cv::Mat &raw_img,
                                              const Boxi &box) {
  if (initFlag_ != 0) {
    std::cout << "[ ERROR ] Init fail return empty result" << std::endl;
    PoseEstimationResult empty_result;
    return empty_result;
  }
  if (modelFilepath_.find("rtmpose") != modelFilepath_.npos) {
    return EstimateRtmPose(raw_img, box);
  } else {
    std::cout << "[ ERROR ] Unsupported model" << std::endl;
    return result_;
  }
}

PoseEstimationResult PoseEstimation::EstimateRtmPose(const cv::Mat &raw_img,
                                                     const Boxi &box) {
  result_points_.clear();
  input_tensors_.clear();
  {
#ifdef DEBUG
    std::cout << "|-- Preprocess" << std::endl;
    TimeWatcher t("|--");
#endif
    processor_.Preprocess(raw_img, box, inputDims_, input_tensors_,
                          crop_result_pair_, CHW);
  }
  postprocessor_.Postprocess(Infer(input_tensors_), crop_result_pair_,
                             result_points_);

  result_.result_points = result_points_;
  result_.timestamp = std::chrono::high_resolution_clock::now();
  return result_;
}

PoseEstimationResult PoseEstimation::Postprocess() {
  postprocessor_.Postprocess(Infer(input_tensors_), crop_result_pair_,
                             result_points_);

  result_.result_points = result_points_;
  return result_;
}

int PoseEstimation::InitFromCommand(const std::string &modelFilepath,
                                    const bool disable_spacemit_ep,
                                    const int intra_threads_num) {
  instanceName_ = "pose-estimation-inference";
  modelFilepath_ = modelFilepath;
  initFlag_ = GetEngine()->Init(instanceName_, modelFilepath_,
                                disable_spacemit_ep, intra_threads_num);
  inputDims_ = GetEngine()->GetInputDims();
  return initFlag_;
}

int PoseEstimation::InitFromConfig(const std::string &configFilepath) {
  std::ifstream f(configFilepath);
  json config = json::parse(f);
  if (configCheck(config)) {
    initFlag_ = 1;
    std::cout << "[ ERROR ] Config check fail" << std::endl;
    return initFlag_;
  }
  modelFilepath_ = config["model_path"];
  initFlag_ = GetEngine()->Init(config);
  inputDims_ = GetEngine()->GetInputDims();
  return initFlag_;
}
