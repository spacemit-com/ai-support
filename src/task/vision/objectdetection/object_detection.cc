#include "object_detection.h"

#include <chrono>
#include <iostream>

#include "src/utils/utils.h"
#include "utils/time.h"

std::vector<std::vector<float>> ObjectDetection::Process(
    const cv::Mat &img_raw) {
  input_tensors_.clear();
  if (init_flag_ != 0) {
    std::cout << "[ ERROR ] Init fail return empty tensors" << std::endl;
    return input_tensors_;
  }
  if (option_.model_path.find("yolov4") != option_.model_path.npos) {
    preprocessor_.Preprocess(img_raw, input_dims_, input_tensors_, HWC);
  } else if (option_.model_path.find("yolov6") != option_.model_path.npos) {
    preprocessor_.Preprocess(img_raw, input_dims_, input_tensors_, CHW);
  } else if (option_.model_path.find("nanodet-plus") !=
             option_.model_path.npos) {
    preprocessor_.PreprocessNanoDetPlus(img_raw, input_dims_, input_tensors_);
  } else if (option_.model_path.find("rtmdet") != option_.model_path.npos) {
    preprocessor_.Preprocess(img_raw, input_dims_, input_tensors_, CHW);
  } else {
    std::cout << "[ ERROR ] Unsupported model return empty tensors"
              << std::endl;
    return input_tensors_;
  }
  return input_tensors_;
}

ObjectDetectionResult ObjectDetection::Detect(const cv::Mat &img_raw) {
  result_boxes_.clear();
  input_tensors_.clear();
  Preprocess(img_raw);
  return Postprocess();
}

ObjectDetectionResult ObjectDetection::Detect(
    const std::vector<std::vector<float>> &input_tensors, const int img_height,
    const int img_width) {
  result_boxes_.clear();
  input_tensors_ = input_tensors;
  if (init_flag_ != 0) {
    std::cout << "[ ERROR ] Init fail return empty result" << std::endl;
    result_.result_bboxes = result_boxes_;
    result_.timestamp = std::chrono::steady_clock::now();
    return result_;
  }
  if (option_.model_path.find("yolov4") != option_.model_path.npos) {
    postprocessor_.Postprocess(Infer(input_tensors_), result_boxes_,
                               input_dims_, img_height, img_width, labels_,
                               option_.score_threshold, option_.nms_threshold);
  } else if (option_.model_path.find("yolov6") != option_.model_path.npos) {
    postprocessor_.PostprocessYolov6(Infer(input_tensors_), result_boxes_,
                                     input_dims_, img_height, img_width,
                                     labels_, option_.score_threshold);
  } else if (option_.model_path.find("nanodet-plus") !=
             option_.model_path.npos) {
    postprocessor_.PostprocessNanoDetPlus(
        Infer(input_tensors_), result_boxes_, input_dims_, img_height,
        img_width, labels_, option_.score_threshold, option_.nms_threshold);
  } else if (option_.model_path.find("rtmdet") != option_.model_path.npos) {
    postprocessor_.PostprocessRtmDet(
        Infer(input_tensors_), result_boxes_, input_dims_, img_height,
        img_width, labels_, option_.score_threshold, option_.nms_threshold);
  } else {
    std::cout << "[ ERROR ] Unsupported model return empty result" << std::endl;
  }
  result_.result_bboxes = result_boxes_;
  result_.timestamp = std::chrono::steady_clock::now();
  return result_;
}

void ObjectDetection::Preprocess(const cv::Mat &img_raw) {
  if (init_flag_ != 0) {
    return;
  }
  img_height_ = img_raw.rows;
  img_width_ = img_raw.cols;
  if (option_.model_path.find("yolov4") != option_.model_path.npos) {
    preprocessor_.Preprocess(img_raw, input_dims_, input_tensors_, HWC);
  } else if (option_.model_path.find("yolov6") != option_.model_path.npos) {
    preprocessor_.Preprocess(img_raw, input_dims_, input_tensors_, CHW);
  } else if (option_.model_path.find("nanodet-plus") !=
             option_.model_path.npos) {
    preprocessor_.PreprocessNanoDetPlus(img_raw, input_dims_, input_tensors_);
  } else if (option_.model_path.find("rtmdet") != option_.model_path.npos) {
    preprocessor_.Preprocess(img_raw, input_dims_, input_tensors_, CHW);
  } else {
    std::cout << "[ ERROR ] Unsupported model" << std::endl;
  }
}

ObjectDetectionResult ObjectDetection::Postprocess() {
  if (init_flag_ != 0) {
    std::cout << "[ ERROR ] Init fail return empty result" << std::endl;
    result_.result_bboxes = result_boxes_;
    result_.timestamp = std::chrono::steady_clock::now();
    return result_;
  }
  if (option_.model_path.find("yolov4") != option_.model_path.npos) {
    postprocessor_.Postprocess(Infer(input_tensors_), result_boxes_,
                               input_dims_, img_height_, img_width_, labels_,
                               option_.score_threshold, option_.nms_threshold);
  } else if (option_.model_path.find("yolov6") != option_.model_path.npos) {
    postprocessor_.PostprocessYolov6(Infer(input_tensors_), result_boxes_,
                                     input_dims_, img_height_, img_width_,
                                     labels_, option_.score_threshold);
  } else if (option_.model_path.find("nanodet-plus") !=
             option_.model_path.npos) {
    postprocessor_.PostprocessNanoDetPlus(
        Infer(input_tensors_), result_boxes_, input_dims_, img_height_,
        img_width_, labels_, option_.score_threshold, option_.nms_threshold);
  } else if (option_.model_path.find("rtmdet") != option_.model_path.npos) {
    postprocessor_.PostprocessRtmDet(
        Infer(input_tensors_), result_boxes_, input_dims_, img_height_,
        img_width_, labels_, option_.score_threshold, option_.nms_threshold);
  } else {
    std::cout << "[ ERROR ] Unsupported model return empty result" << std::endl;
  }
  ApllyList();
  result_.result_bboxes = result_boxes_;
  result_.timestamp = std::chrono::steady_clock::now();
  return result_;
}

void ObjectDetection::ApllyList() {
  if (option_.class_name_blacklist.empty() &&
      option_.class_name_whitelist.empty()) {
    return;
  }
  for (auto it = result_boxes_.begin(); it != result_boxes_.end();) {
    if (!class_name_list_[static_cast<int>(it->label)]) {
      it = result_boxes_.erase(it);
    } else {
      ++it;
    }
  }
}

int ObjectDetection::InitFromOption(const ObjectDetectionOption &option) {
  init_flag_ = 1;
  option_ = option;
  instance_name_ = "object-detection-inference";
  labels_ = readLabels(option_.label_path);
  if (labels_.empty()) {
    std::cout << "[ ERROR ] label file is empty, init fail" << std::endl;
    return init_flag_;
  }
  int label_size = labels_.size();
  if (!option_.class_name_whitelist.empty()) {
    std::vector<int> list(label_size, 0);
    class_name_list_ = list;
    for (size_t i = 0; i < option_.class_name_whitelist.size(); i++) {
      if (option_.class_name_whitelist[i] < label_size &&
          option_.class_name_whitelist[i] >= 0) {
        class_name_list_[option_.class_name_whitelist[i]] = 1;
      }
    }
  } else {
    std::vector<int> list(label_size, 1);
    class_name_list_ = list;
  }
  if (!option_.class_name_blacklist.empty()) {
    for (size_t i = 0; i < option_.class_name_blacklist.size(); i++) {
      if (option_.class_name_blacklist[i] < label_size &&
          option_.class_name_blacklist[i] >= 0) {
        class_name_list_[option_.class_name_blacklist[i]] = 0;
      }
    }
  }
  init_flag_ =
      GetEngine()->Init(instance_name_, option_.model_path,
                        option.intra_threads_num, option.inter_threads_num);
  if (!init_flag_) {
    input_dims_ = GetEngine()->GetInputDims();
  }
  return init_flag_;
}
