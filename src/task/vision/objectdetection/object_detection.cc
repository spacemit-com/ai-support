#include "src/task/vision/objectdetection/object_detection.h"

#include <chrono>
#include <fstream>

#include "src/utils/json.hpp"
#include "utils/time.h"
using json = nlohmann::json;

std::vector<std::vector<float>> ObjectDetection::Process(
    const cv::Mat &raw_img) {
  input_tensors_.clear();
  if (initFlag_ != 0) {
    std::cout << "[ ERROR ] Init fail return empty tensors" << std::endl;
    return input_tensors_;
  }
  if (modelFilepath_.find("yolov4") != modelFilepath_.npos) {
    preprocessor_.Preprocess(raw_img, inputDims_, input_tensors_, HWC);
  } else if (modelFilepath_.find("yolov6") != modelFilepath_.npos) {
    preprocessor_.Preprocess(raw_img, inputDims_, input_tensors_, CHW);
  } else if (modelFilepath_.find("nanodet-plus") != modelFilepath_.npos) {
    preprocessor_.PreprocessNanoDetPlus(raw_img, inputDims_, input_tensors_);
  } else if (modelFilepath_.find("rtmdet") != modelFilepath_.npos) {
    preprocessor_.Preprocess(raw_img, inputDims_, input_tensors_, CHW);
  } else {
    std::cout << "[ ERROR ] Unsupported model return empty tensors"
              << std::endl;
    return input_tensors_;
  }
  return input_tensors_;
}

ObjectDetectionResult ObjectDetection::Detect(const cv::Mat &raw_img) {
  result_boxes_.clear();
  input_tensors_.clear();
  Preprocess(raw_img);
  return Postprocess();
}

ObjectDetectionResult ObjectDetection::Detect(
    const std::vector<std::vector<float>> &input_tensors, const int img_height,
    const int img_width) {
  result_boxes_.clear();
  input_tensors_ = input_tensors;
  if (initFlag_ != 0) {
    std::cout << "[ ERROR ] Init fail return empty result" << std::endl;
    result_.result_bboxes = result_boxes_;
    result_.timestamp = std::chrono::steady_clock::now();
    return result_;
  }
  if (modelFilepath_.find("yolov4") != modelFilepath_.npos) {
    postprocessor_.Postprocess(Infer(input_tensors_), result_boxes_, inputDims_,
                               img_height, img_width, labels_, score_threshold_,
                               nms_threshold_);
  } else if (modelFilepath_.find("yolov6") != modelFilepath_.npos) {
    postprocessor_.PostprocessYolov6(Infer(input_tensors_), result_boxes_,
                                     inputDims_, img_height, img_width, labels_,
                                     score_threshold_);
  } else if (modelFilepath_.find("nanodet-plus") != modelFilepath_.npos) {
    postprocessor_.PostprocessNanoDetPlus(
        Infer(input_tensors_), result_boxes_, inputDims_, img_height, img_width,
        labels_, score_threshold_, nms_threshold_);
  } else if (modelFilepath_.find("rtmdet") != modelFilepath_.npos) {
    postprocessor_.PostprocessRtmDet(Infer(input_tensors_), result_boxes_,
                                     inputDims_, img_height, img_width, labels_,
                                     score_threshold_, nms_threshold_);
  } else {
    std::cout << "[ ERROR ] Unsupported model return empty result" << std::endl;
  }
  result_.result_bboxes = result_boxes_;
  result_.timestamp = std::chrono::steady_clock::now();
  return result_;
}

void ObjectDetection::Preprocess(const cv::Mat &raw_img) {
  if (initFlag_ != 0) {
    std::cout << "[ ERROR ] Init fail" << std::endl;
    return;
  }
  img_height_ = raw_img.rows;
  img_width_ = raw_img.cols;
  if (modelFilepath_.find("yolov4") != modelFilepath_.npos) {
    preprocessor_.Preprocess(raw_img, inputDims_, input_tensors_, HWC);
  } else if (modelFilepath_.find("yolov6") != modelFilepath_.npos) {
    preprocessor_.Preprocess(raw_img, inputDims_, input_tensors_, CHW);
  } else if (modelFilepath_.find("nanodet-plus") != modelFilepath_.npos) {
    preprocessor_.PreprocessNanoDetPlus(raw_img, inputDims_, input_tensors_);
  } else if (modelFilepath_.find("rtmdet") != modelFilepath_.npos) {
    preprocessor_.Preprocess(raw_img, inputDims_, input_tensors_, CHW);
  } else {
    std::cout << "[ ERROR ] Unsupported model" << std::endl;
  }
}

ObjectDetectionResult ObjectDetection::Postprocess() {
  if (initFlag_ != 0) {
    std::cout << "[ ERROR ] Init fail return empty result" << std::endl;
    result_.result_bboxes = result_boxes_;
    result_.timestamp = std::chrono::steady_clock::now();
    return result_;
  }
  if (modelFilepath_.find("yolov4") != modelFilepath_.npos) {
    postprocessor_.Postprocess(Infer(input_tensors_), result_boxes_, inputDims_,
                               img_height_, img_width_, labels_,
                               score_threshold_, nms_threshold_);
  } else if (modelFilepath_.find("yolov6") != modelFilepath_.npos) {
    postprocessor_.PostprocessYolov6(Infer(input_tensors_), result_boxes_,
                                     inputDims_, img_height_, img_width_,
                                     labels_, score_threshold_);
  } else if (modelFilepath_.find("nanodet-plus") != modelFilepath_.npos) {
    postprocessor_.PostprocessNanoDetPlus(
        Infer(input_tensors_), result_boxes_, inputDims_, img_height_,
        img_width_, labels_, score_threshold_, nms_threshold_);
  } else if (modelFilepath_.find("rtmdet") != modelFilepath_.npos) {
    postprocessor_.PostprocessRtmDet(Infer(input_tensors_), result_boxes_,
                                     inputDims_, img_height_, img_width_,
                                     labels_, score_threshold_, nms_threshold_);
  } else {
    std::cout << "[ ERROR ] Unsupported model return empty result" << std::endl;
  }
  if (!class_name_blacklist_.empty()) {
    for (int i = 0; i < static_cast<int>(class_name_blacklist_.size()); i++) {
      for (int j = 0; j < static_cast<int>(result_boxes_.size()); j++) {
        if (class_name_blacklist_[i] ==
            static_cast<int>(result_boxes_[j].label)) {
          result_boxes_[j].flag = false;
        }
      }
    }
  }
  if (!class_name_whitelist_.empty()) {
    for (int j = 0; j < static_cast<int>(result_boxes_.size()); j++) {
      result_boxes_[j].flag = false;
    }
    for (int i = 0; i < static_cast<int>(class_name_whitelist_.size()); i++) {
      for (int j = 0; j < static_cast<int>(result_boxes_.size()); j++) {
        if (class_name_whitelist_[i] ==
            static_cast<int>(result_boxes_[j].label)) {
          result_boxes_[j].flag = true;
        }
      }
    }
  }
  result_.result_bboxes = result_boxes_;
  result_.timestamp = std::chrono::steady_clock::now();
  return result_;
}

int ObjectDetection::InitFromCommand(const std::string &modelFilepath,
                                     const std::string &labelFilepath) {
  instanceName_ = "object-detection-inference";
  modelFilepath_ = modelFilepath;
  labelFilepath_ = labelFilepath;
  score_threshold_ = -1.f;
  nms_threshold_ = -1.f;
  initFlag_ = GetEngine()->Init(instanceName_, modelFilepath_);
  inputDims_ = GetEngine()->GetInputDims();
  labels_ = readLabels(labelFilepath_);
  return initFlag_;
}

int ObjectDetection::InitFromConfig(const std::string &configFilepath) {
  std::ifstream f(configFilepath);
  json config = json::parse(f);
  if (configCheck(config)) {
    initFlag_ = 1;
    std::cout << "[ ERROR ] Config check fail" << std::endl;
    return initFlag_;
  }
  modelFilepath_ = config["model_path"];
  labelFilepath_ = config["label_path"];
  if (config.contains("score_threshold")) {
    score_threshold_ = config["score_threshold"];
  } else {
    score_threshold_ = -1.f;
  }
  if (config.contains("nms_threshold")) {
    nms_threshold_ = config["nms_threshold"];
  } else {
    nms_threshold_ = -1.f;
  }
  if (config.contains("class_name_whitelist")) {
    class_name_whitelist_ =
        config["class_name_whitelist"].get<std::vector<int>>();
  }
  if (config.contains("class_name_blacklist")) {
    class_name_blacklist_ =
        config["class_name_blacklist"].get<std::vector<int>>();
  }
  labels_ = readLabels(labelFilepath_);
  initFlag_ = GetEngine()->Init(config);
  inputDims_ = GetEngine()->GetInputDims();
  return initFlag_;
}
