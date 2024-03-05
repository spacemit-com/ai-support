#include "task/vision/object_detection_task.h"

#include "include/utils/utils.h"
#include "src/task/vision/objectdetection/object_detection.h"
#include "src/utils/utils.h"

class ObjectDetectionTask::impl {
 public:
  std::unique_ptr<ObjectDetection> objectdetection_;
};

ObjectDetectionTask::ObjectDetectionTask(const ObjectDetectionOption &option)
    : pimpl_(std::make_unique<impl>()) {
  init_flag_ = -1;
  pimpl_->objectdetection_ =
      std::unique_ptr<ObjectDetection>(new ObjectDetection());
  init_flag_ = pimpl_->objectdetection_->InitFromOption(option);
}

ObjectDetectionTask::ObjectDetectionTask(const std::string &config_file_path)
    : pimpl_(std::make_unique<impl>()) {
  init_flag_ = -1;
  pimpl_->objectdetection_ =
      std::unique_ptr<ObjectDetection>(new ObjectDetection());
  ObjectDetectionOption option;
  if (!configToOption(config_file_path, option)) {
    init_flag_ = pimpl_->objectdetection_->InitFromOption(option);
  }
}

int ObjectDetectionTask::getInitFlag() { return init_flag_; }

ObjectDetectionResult ObjectDetectionTask::Detect(const cv::Mat &img_raw) {
  return pimpl_->objectdetection_->Detect(img_raw);
}

ObjectDetectionResult ObjectDetectionTask::Detect(
    const std::vector<std::vector<float>> &input_tensors, const int img_height,
    const int img_width) {
  return pimpl_->objectdetection_->Detect(input_tensors, img_height, img_width);
}

std::vector<std::vector<float>> ObjectDetectionTask::Process(
    const cv::Mat &img_raw) {
  return pimpl_->objectdetection_->Process(img_raw);
}
