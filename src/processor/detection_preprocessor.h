#ifndef SUPPORT_SRC_PROCESSOR_DETECTION_PREPROCESSOR_H_
#define SUPPORT_SRC_PROCESSOR_DETECTION_PREPROCESSOR_H_

#include <chrono>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "src/processor/processor.h"
#include "src/utils/cv2_utils.h"
#include "src/utils/nms_utils.h"

class DetectionPreprocessor : public Preprocessor {
 public:
  DetectionPreprocessor() {}
  ~DetectionPreprocessor() {}
  // Function to validate the input image file extension.
  void Preprocess(const cv::Mat& mat,
                  std::vector<std::vector<int64_t>>& input_node_dims,
                  std::vector<std::vector<float>>& input_tensor_values,
                  unsigned int data_format);
  void PreprocessNanoDetPlus(
      const cv::Mat& mat, std::vector<std::vector<int64_t>>& input_node_dims,
      std::vector<std::vector<float>>& input_tensor_values);
};

#endif  // SUPPORT_SRC_PROCESSOR_DETECTION_PREPROCESSOR_H_
