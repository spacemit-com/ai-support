#ifndef SUPPORT_SRC_PROCESSOR_CLASSIFICATION_PREPROCESSOR_H_
#define SUPPORT_SRC_PROCESSOR_CLASSIFICATION_PREPROCESSOR_H_

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "src/processor/processor.h"

class ClassificationPreprocessor : public Preprocessor {
 public:
  ClassificationPreprocessor() {}
  ~ClassificationPreprocessor() {}
  void Preprocess(const cv::Mat& imageBGR,
                  std::vector<std::vector<int64_t>> inputDims,
                  std::vector<std::vector<float>>& input_tensor_values);
};

#endif  // SUPPORT_SRC_PROCESSOR_CLASSIFICATION_PREPROCESSOR_H_
