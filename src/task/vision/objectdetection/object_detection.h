#ifndef SUPPORT_SRC_TASK_VISION_OBJECTDETECTION_OBJECT_DETECTION_H_
#define SUPPORT_SRC_TASK_VISION_OBJECTDETECTION_OBJECT_DETECTION_H_

#include <iostream>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "src/core/engine.h"
#include "src/processor/detection_postprocessor.h"
#include "src/processor/detection_preprocessor.h"
#include "src/task/vision/base_vision_task_api.h"
#include "src/utils/utils.h"
#include "task/vision/object_detection_types.h"

class ObjectDetection : public BaseVisionTaskApi<ObjectDetectionResult> {
 public:
  ObjectDetection() : BaseVisionTaskApi<ObjectDetectionResult>() {
    init_flag_ = -1;
  }
  ~ObjectDetection() {}
  ObjectDetectionResult Detect(const cv::Mat &img_raw);
  int InitFromOption(const ObjectDetectionOption &option);
  std::vector<std::vector<float>> Process(const cv::Mat &img_raw);
  ObjectDetectionResult Detect(
      const std::vector<std::vector<float>> &input_tensors,
      const int img_height, const int img_width);

 protected:
  void ApllyList();
  void Preprocess(const cv::Mat &img_raw) override;
  ObjectDetectionResult Postprocess() override;

 private:
  std::string instance_name_;
  std::vector<std::string> labels_;
  ObjectDetectionOption option_;
  std::vector<std::vector<int64_t>> input_dims_;
  std::vector<std::vector<float>> input_tensors_;
  DetectionPreprocessor preprocessor_;
  DetectionPostprocessor postprocessor_;
  std::vector<Boxi> result_boxes_;
  ObjectDetectionResult result_;
  int img_height_;
  int img_width_;
  int init_flag_;
  std::vector<int> class_name_list_;
};

#endif  // SUPPORT_SRC_TASK_VISION_OBJECTDETECTION_OBJECT_DETECTION_H_
