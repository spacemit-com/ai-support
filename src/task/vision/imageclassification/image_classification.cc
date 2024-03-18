#include "src/task/vision/imageclassification/image_classification.h"

#include <iostream>

#include "utils/time.h"
#include "utils/utils.h"

int ImageClassification::InitFromOption(
    const ImageClassificationOption &option) {
  option_ = option;
  init_flag_ = 1;
  instance_name_ = "image-classification-inference";
  labels_ = readLabels(option_.label_path);
  if (labels_.empty()) {
    std::cout << "[ ERROR ] label file is empty, init fail" << std::endl;
    return init_flag_;
  }
  init_flag_ =
      GetEngine()->Init(instance_name_, option_.model_path,
                        option.intra_threads_num, option.inter_threads_num);
  return init_flag_;
}

void ImageClassification::Preprocess(const cv::Mat &img_raw) {
  auto input_dims = GetInputShape();
  preprocessor_.Preprocess(img_raw, input_dims, input_tensors_);
}

ImageClassificationResult ImageClassification::Postprocess() {
  return postprocessor_.Postprocess(Infer(input_tensors_), labels_);
}

ImageClassificationResult ImageClassification::Classify(
    const cv::Mat &img_raw) {
  if (init_flag_ != 0) {
    std::cout << "[ ERROR ] Init fail return empty result" << std::endl;
    ImageClassificationResult empty_result{"", -1, .0f};
    return empty_result;
  } else {
    img_raw_ = img_raw;
    {
#ifdef DEBUG
      std::cout << "|-- Preprocess" << std::endl;
      TimeWatcher t("|--");
#endif
      Preprocess(img_raw_);
    }
    return Postprocess();
  }
}
