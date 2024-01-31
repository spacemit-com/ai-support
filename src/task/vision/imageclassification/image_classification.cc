#include "src/task/vision/imageclassification/image_classification.h"

#include "utils/time.h"

int imageClassification::Init(const std::string modelFilepath,
                              const std::string labelFilepath) {
  instanceName_ = "image-classification-inference";
  modelFilepath_ = modelFilepath;
  labelFilepath_ = labelFilepath;
  labels_ = readLabels(labelFilepath_);
  initFlag_ = GetEngine()->Init(instanceName_, modelFilepath_);
  return initFlag_;
}

void imageClassification::Preprocess(const cv::Mat &img_raw) {
  auto input_dims = GetInputShape();
  preprocessor_.Preprocess(img_raw, input_dims, input_tensors_);
}

ImageClassificationResult imageClassification::Postprocess() {
  return postprocessor_.Postprocess(Infer(input_tensors_), labels_);
}

ImageClassificationResult imageClassification::Classify(
    const cv::Mat &img_raw) {
  if (initFlag_ != 0) {
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
