#include "task/vision/object_detection_task.h"

#include "include/utils/utils.h"
#include "src/task/vision/objectdetection/object_detection.h"
#include "src/utils/utils.h"

class objectDetectionTask::impl {
 public:
  std::unique_ptr<ObjectDetection> objectdetection_;
};

objectDetectionTask::objectDetectionTask(const std::string &filePath,
                                         const std::string &labelFilepath)
    : pimpl_(std::make_unique<impl>()) {
  init_flag_ = -1;
  pimpl_->objectdetection_ =
      std::unique_ptr<ObjectDetection>(new ObjectDetection());
  if (!checkLabelFileExtension(labelFilepath)) {
    std::cout << "[ ERROR ] The LabelFilepath is not set correctly and the "
                 "labels file should end with extension .txt"
              << std::endl;
  } else if (filePath.length() > 4) {
    std::string suffixStr = filePath.substr(filePath.length() - 4, 4);
    if (strcmp(suffixStr.c_str(), "onnx") == 0) {
      if (!checkModelFileExtension(filePath)) {
        std::cout << "[ ERROR ] The ModelFilepath is not correct. Make sure "
                     "you are setting the path to an onnx model file (.onnx)"
                  << std::endl;
      } else if (!exists_check(filePath) || !exists_check(labelFilepath)) {
        std::cout << "[ ERROR ] The File does not exist. Make sure you are "
                     "setting the correct path to the file"
                  << std::endl;
      } else {
        init_flag_ =
            pimpl_->objectdetection_->InitFromCommand(filePath, labelFilepath);
        if (init_flag_ != 0) {
          std::cout << "[Error] Init fail" << std::endl;
        }
      }
    } else if (strcmp(suffixStr.c_str(), "json") == 0) {
      if (!checkConfigFileExtension(filePath)) {
        std::cout << "[ ERROR ] The ConfigFilepath is not correct. Make sure "
                     "you are setting the path to an json file (.json)"
                  << std::endl;
      } else if (!exists_check(filePath)) {
        std::cout << "[ ERROR ] The File does not exist. Make sure you are "
                     "setting the correct path to the file"
                  << std::endl;
      } else {
        init_flag_ = pimpl_->objectdetection_->InitFromConfig(filePath);
        if (init_flag_ != 0) {
          std::cout << "[Error] Init fail" << std::endl;
        }
      }
    } else {
      std::cout << "[ ERROR ] Unsupport file" << std::endl;
    }
  } else {
    std::cout << "[ ERROR ] Unsupport filepath" << std::endl;
  }
}

objectDetectionTask::objectDetectionTask(const std::string &filePath)
    : pimpl_(std::make_unique<impl>()) {
  init_flag_ = -1;
  pimpl_->objectdetection_ =
      std::unique_ptr<ObjectDetection>(new ObjectDetection());
  if (filePath.length() > 4) {
    std::string suffixStr = filePath.substr(filePath.length() - 4, 4);
    if (strcmp(suffixStr.c_str(), "json") == 0) {
      if (!checkConfigFileExtension(filePath)) {
        std::cout << "[ ERROR ] The ConfigFilepath is not correct. Make sure "
                     "you are setting the path to an json file (.json)"
                  << std::endl;
      } else if (!exists_check(filePath)) {
        std::cout << "[ ERROR ] The File does not exist. Make sure you are "
                     "setting the correct path to the file"
                  << std::endl;
      } else {
        init_flag_ = pimpl_->objectdetection_->InitFromConfig(filePath);
        if (init_flag_ != 0) {
          std::cout << "[Error] Init fail" << std::endl;
        }
      }
    } else {
      std::cout << "[ ERROR ] Unsupport file" << std::endl;
    }
  } else {
    std::cout << "[ ERROR ] Unsupport filepath" << std::endl;
  }
}

int objectDetectionTask::getInitFlag() { return init_flag_; }

ObjectDetectionResult objectDetectionTask::Detect(const cv::Mat &raw_img) {
  return pimpl_->objectdetection_->Detect(raw_img);
}

ObjectDetectionResult objectDetectionTask::Detect(
    const std::vector<std::vector<float>> &input_tensors, const int img_height,
    const int img_width) {
  return pimpl_->objectdetection_->Detect(input_tensors, img_height, img_width);
}

std::vector<std::vector<float>> objectDetectionTask::Process(
    const cv::Mat &img_raw) {
  return pimpl_->objectdetection_->Process(img_raw);
}
