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
  ImageClassificationOption()
      : model_path(""),
        label_path(""),
        intra_threads_num(2),
        inter_threads_num(2) {}
  ImageClassificationOption(const std::string mp, const std::string lp,
                            const int atm, const int etm)
      : model_path(mp),
        label_path(lp),
        intra_threads_num(atm),
        inter_threads_num(etm) {}
};

#endif  // SUPPORT_INCLUDE_TASK_VISION_IMAGE_CLASSIFICATION_TYPES_H_
