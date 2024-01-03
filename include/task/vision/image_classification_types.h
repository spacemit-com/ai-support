#ifndef SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TYPES_H_
#define SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TYPES_H_

#include <chrono>
#include <string>

struct ImageClassificationResult {
  std::string label_text;
  int label;
  float score;
  std::chrono::_V2::system_clock::time_point timestamp;
};
#endif  // SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TYPES_H_
