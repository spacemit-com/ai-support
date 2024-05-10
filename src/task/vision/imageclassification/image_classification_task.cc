#include "task/vision/image_classification_task.h"

#include "image_classification.h"

class ImageClassificationTask::impl {
 public:
  std::unique_ptr<ImageClassification> imageclassification_;
};

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
