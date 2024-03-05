#ifndef SUPPORT_SRC_TASK_VISION_ImageClassification_IMAGE_CLASSIFICATION_H_
#define SUPPORT_SRC_TASK_VISION_ImageClassification_IMAGE_CLASSIFICATION_H_

#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "src/core/engine.h"
#include "src/processor/classification_postprocessor.h"
#include "src/processor/classification_preprocessor.h"
#include "src/task/vision/base_vision_task_api.h"
#include "src/utils/utils.h"
#include "task/vision/image_classification_types.h"

class ImageClassification
    : public BaseVisionTaskApi<ImageClassificationResult> {
 public:
  ImageClassification() : BaseVisionTaskApi<ImageClassificationResult>() {
    init_flag_ = -1;
  }
  ~ImageClassification() {}
  int InitFromOption(const ImageClassificationOption& option);
  ImageClassificationResult Classify(const cv::Mat& img_raw);

 protected:
  void Preprocess(const cv::Mat& img_raw) override;
  ImageClassificationResult Postprocess() override;

 private:
  ClassificationPreprocessor preprocessor_;
  ClassificationPostprocessor postprocessor_;
  ImageClassificationOption option_;
  std::string instance_name_;
  cv::Mat img_raw_;
  std::vector<std::string> labels_;
  std::vector<Ort::Value> output_tensors_;
  std::vector<std::vector<float>> input_tensors_;
  int init_flag_;
};

#endif  // SUPPORT_SRC_TASK_VISION_ImageClassification_IMAGE_CLASSIFICATION_H_
