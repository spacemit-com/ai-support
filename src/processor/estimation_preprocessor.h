#ifndef SUPPORT_SRC_PROCESSOR_ESTIMATION_PREPROCESSOR_H_
#define SUPPORT_SRC_PROCESSOR_ESTIMATION_PREPROCESSOR_H_

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"
#include "src/processor/processor.h"
#include "task/vision/object_detection_types.h"

class EstimationPreprocessor : public Preprocessor {
 public:
  EstimationPreprocessor() {}
  ~EstimationPreprocessor() {}
  // Function to validate the input image file extension.
  void Preprocess(const cv::Mat& mat, const Boxi& box,
                  std::vector<std::vector<float>>& input_tensor_values,
                  std::pair<cv::Mat, cv::Mat>& crop_result_pair,
                  unsigned int data_format);

 protected:
  std::pair<cv::Mat, cv::Mat> CropImageByDetectBox(const cv::Mat& input_image,
                                                   const Boxi& box);
};

#endif  // SUPPORT_SRC_PROCESSOR_ESTIMATION_PREPROCESSOR_H_
