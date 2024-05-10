#ifndef SUPPORT_SRC_PROCESSOR_CLASSIFICATION_PREPROCESSOR_H_
#define SUPPORT_SRC_PROCESSOR_CLASSIFICATION_PREPROCESSOR_H_

#include <vector>

#include "opencv2/opencv.hpp"
#include "processor.h"

class ClassificationPreprocessor : public Preprocessor {
 public:
  ClassificationPreprocessor() {}
  ~ClassificationPreprocessor() {}
  void Preprocess(const cv::Mat& imageBGR,
                  std::vector<std::vector<int64_t>> inputDims,
                  std::vector<std::vector<float>>& input_tensor_values);
};

#endif  // SUPPORT_SRC_PROCESSOR_CLASSIFICATION_PREPROCESSOR_H_
