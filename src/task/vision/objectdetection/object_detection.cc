#include "src/task/vision/objectdetection/object_detection.h"

#include <fstream>

#include "src/utils/json.hpp"
#include "utils/time.h"
using json = nlohmann::json;

void ObjectDetection::Preprocess(std::vector<std::vector<float>> &input_tensors,
                                 const cv::Mat &img_raw) {
  processor_.Preprocess(img_raw, inputDims_, input_tensors_, CHW);
}

ObjectDetectionResult ObjectDetection::Detect(const cv::Mat &raw_img) {
  if (initFlag_ != 0) {
    std::cout << "[ ERROR ] Init fail return empty result" << std::endl;
    ObjectDetectionResult empty_result;
    return empty_result;
  }
  if (modelFilepath_.find("yolov4") != modelFilepath_.npos) {
    return DetectYolov4(raw_img);
  } else if (modelFilepath_.find("yolov6") != modelFilepath_.npos) {
    return DetectYolov6(raw_img);
  } else if (modelFilepath_.find("nanodet-plus") != modelFilepath_.npos) {
    return DetectNanoDet(raw_img);
  } else {
    std::cout << "[ ERROR ] Unsupported model" << std::endl;
    return result_;
  }
}

ObjectDetectionResult ObjectDetection::DetectNanoDet(const cv::Mat &raw_img) {
  result_boxes_.clear();
  input_tensors_.clear();
  img_height_ = raw_img.rows;
  img_width_ = raw_img.cols;

  {
#ifdef DEBUG
    std::cout << "|-- Preprocess " << std::endl;
    TimeWatcher t("|--");
#endif
    processor_.PreprocessNanoDet(raw_img, inputDims_, input_tensors_);
  }

  postprocessor_.PostprocessNanoDet(Infer(input_tensors_), result_boxes_,
                                    inputDims_, img_height_, img_width_,
                                    labels_);

  result_.result_bboxes = result_boxes_;
  return result_;
}

ObjectDetectionResult ObjectDetection::DetectYolov6(const cv::Mat &raw_img) {
  result_boxes_.clear();
  input_tensors_.clear();
  img_height_ = raw_img.rows;
  img_width_ = raw_img.cols;
  {
#ifdef DEBUG
    std::cout << "|-- Preprocess" << std::endl;
    TimeWatcher t("|--");
#endif
    processor_.Preprocess(raw_img, inputDims_, input_tensors_, CHW);
  }
  postprocessor_.PostprocessYolov6(Infer(input_tensors_), result_boxes_,
                                   inputDims_, img_height_, img_width_,
                                   labels_);

  result_.result_bboxes = result_boxes_;
  return result_;
}

ObjectDetectionResult ObjectDetection::DetectYolov4(const cv::Mat &raw_img) {
  result_boxes_.clear();
  input_tensors_.clear();
  img_height_ = raw_img.rows;
  img_width_ = raw_img.cols;
  {
#ifdef DEBUG
    std::cout << "|-- Preprocess" << std::endl;
    TimeWatcher t("|--");
#endif
    processor_.Preprocess(raw_img, inputDims_, input_tensors_, HWC);
  }
  postprocessor_.Postprocess(Infer(input_tensors_), result_boxes_, inputDims_,
                             img_height_, img_width_, labels_);

  result_.result_bboxes = result_boxes_;
  return result_;
}
ObjectDetectionResult ObjectDetection::Postprocess() {
  postprocessor_.Postprocess(Infer(input_tensors_), result_boxes_, inputDims_,
                             img_height_, img_width_, labels_);

  result_.result_bboxes = result_boxes_;
  return result_;
}

int ObjectDetection::InitFromCommand(const std::string &modelFilepath,
                                     const std::string &labelFilepath) {
  instanceName_ = "object-detection-inference";
  modelFilepath_ = modelFilepath;
  labelFilepath_ = labelFilepath;
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
  labels_ = readLabels(labelFilepath_);
  initFlag_ = GetEngine()->Init(config);
  inputDims_ = GetEngine()->GetInputDims();
  return initFlag_;
}
