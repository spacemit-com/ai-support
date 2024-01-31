#include "src/task/vision/poseestimation/pose_estimation.h"

#include <chrono>
#include <fstream>

#include "src/utils/json.hpp"
#include "utils/time.h"

using json = nlohmann::json;

PoseEstimationResult PoseEstimation::Estimate(const cv::Mat &raw_img,
                                              const Boxi &box) {
  result_points_.clear();
  input_tensors_.clear();
  box_ = box;
  {
#ifdef DEBUG
    std::cout << "|-- Preprocess" << std::endl;
    TimeWatcher t("|--");
#endif
    Preprocess(raw_img);
  }
  return Postprocess();
}

void PoseEstimation::Preprocess(const cv::Mat &img_raw) {
  if (initFlag_ != 0) {
    std::cout << "[ ERROR ] Init fail" << std::endl;
    return;
  }
  if (modelFilepath_.find("rtmpose") != modelFilepath_.npos) {
    processor_.Preprocess(img_raw, box_, input_tensors_, crop_result_pair_,
                          CHW);
  } else {
    std::cout << "[ ERROR ] Unsupported model" << std::endl;
  }
}

PoseEstimationResult PoseEstimation::Postprocess() {
  if (initFlag_ != 0) {
    std::cout << "[ ERROR ] Init fail" << std::endl;
    result_.result_points = result_points_;
    result_.timestamp = std::chrono::steady_clock::now();
    return result_;
  }
  if (modelFilepath_.find("rtmpose") != modelFilepath_.npos) {
    postprocessor_.Postprocess(Infer(input_tensors_), crop_result_pair_,
                               result_points_);
  } else {
    std::cout << "[ ERROR ] Unsupported model" << std::endl;
  }
  result_.result_points = result_points_;
  result_.timestamp = std::chrono::steady_clock::now();
  return result_;
}

int PoseEstimation::InitFromCommand(const std::string &modelFilepath) {
  instanceName_ = "pose-estimation-inference";
  modelFilepath_ = modelFilepath;
  initFlag_ = GetEngine()->Init(instanceName_, modelFilepath_);
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
