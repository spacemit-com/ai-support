#ifndef SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TYPES_H_
#define SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TYPES_H_

#include <chrono>
#include <string>

struct ImageClassificationResult {
  std::string label_text;
  int label;
  float score;
};

struct ImageClassificationOption {
  std::string model_path;
  std::string label_path;
  int intra_threads_num = 2;
  int inter_threads_num = 2;
};

#endif  // SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TYPES_H_
