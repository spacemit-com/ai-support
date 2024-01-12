#include "task/vision/image_classification_task.h"

#include "include/utils/utils.h"
#include "src/task/vision/imageclassification/image_classification.h"
#include "src/utils/utils.h"

class imageClassificationTask::impl {
 public:
  std::unique_ptr<imageClassification> imageclassification_;
};

imageClassificationTask::imageClassificationTask(
    const std::string& filePath, const std::string& labelFilepath,
    const bool& disable_spacemit_ep, const int& intra_threads_num)
    : pimpl_(std::make_unique<impl>()) {
  pimpl_->imageclassification_ =
      std::unique_ptr<imageClassification>(new imageClassification());
  if (!checkLabelFileExtension(labelFilepath)) {
    std::cout << "[ ERROR ] The LabelFilepath is not set correctly and the "
                 "labels file should end with extension .txt"
              << std::endl;
  } else if (!checkModelFileExtension(filePath)) {
    std::cout << "[ ERROR ] The ModelFilepath is not correct. Make sure you "
                 "are setting the path to an onnx model file (.onnx)"
              << std::endl;
  } else if (!exists_check(filePath) || !exists_check(labelFilepath)) {
    std::cout << "[ ERROR ] The File does not exist. Make sure you are setting "
                 "the correct path to the file"
              << std::endl;
  } else {
    int flag = pimpl_->imageclassification_->Init(
        filePath, labelFilepath, disable_spacemit_ep, intra_threads_num);
  }
}

ImageClassificationResult imageClassificationTask::Classify(
    const cv::Mat& img_raw) {
  return pimpl_->imageclassification_->Classify(img_raw);
}