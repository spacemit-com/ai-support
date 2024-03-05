#include "task/vision/image_classification_task.h"

#include "include/utils/utils.h"
#include "src/task/vision/imageclassification/image_classification.h"
#include "src/utils/utils.h"

class ImageClassificationTask::impl {
 public:
  std::unique_ptr<ImageClassification> imageclassification_;
};

ImageClassificationTask::ImageClassificationTask(
    const std::string& config_file_path)
    : pimpl_(std::make_unique<impl>()) {
  init_flag_ = -1;
  pimpl_->imageclassification_ =
      std::unique_ptr<ImageClassification>(new ImageClassification());
  ImageClassificationOption option;
  if (!configToOption(config_file_path, option)) {
    init_flag_ = pimpl_->imageclassification_->InitFromOption(option);
  }
}

ImageClassificationTask::ImageClassificationTask(
    const ImageClassificationOption& option)
    : pimpl_(std::make_unique<impl>()) {
  init_flag_ = -1;
  pimpl_->imageclassification_ =
      std::unique_ptr<ImageClassification>(new ImageClassification());
  init_flag_ = pimpl_->imageclassification_->InitFromOption(option);
}

int ImageClassificationTask::getInitFlag() { return init_flag_; }

ImageClassificationResult ImageClassificationTask::Classify(
    const cv::Mat& img_raw) {
  return pimpl_->imageclassification_->Classify(img_raw);
}
