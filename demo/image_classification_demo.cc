#include <iostream>

#include "task/vision/image_classification_task.h"
#include "utils/time.h"
#include "utils/utils.h"

int main(int argc, char* argv[]) {
  ImageClassificationOption option;
  std::string config_file_path, image_file_path;
  std::unique_ptr<ImageClassificationTask> imageclassificationtask;
  if (argc == 3) {
    config_file_path = argv[1];
    image_file_path = argv[2];
    imageclassificationtask = std::unique_ptr<ImageClassificationTask>(
        new ImageClassificationTask(config_file_path));
  } else if (argc == 4) {
    option.model_path = argv[1];
    option.label_path = argv[2];
    image_file_path = argv[3];
    imageclassificationtask = std::unique_ptr<ImageClassificationTask>(
        new ImageClassificationTask(option));
  } else {
    std::cout << "Please run with " << argv[0]
              << " <model_file_path> <label_file_path> <image_file_path> or "
              << argv[0] << " <config_file_path> <image_file_path>"
              << std::endl;
    return -1;
  }
  if (imageclassificationtask->getInitFlag() != 0) {
    return -1;
  }
  cv::Mat img_raw;
#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
  {
#ifdef DEBUG
    TimeWatcher t("|-- Load input data");
#endif
    img_raw = cv::imread(image_file_path);
  }
  if (img_raw.empty()) {
    std::cout << "[ ERROR ] Read image failed" << std::endl;
    return -1;
  }
  ImageClassificationResult result = imageclassificationtask->Classify(img_raw);
  std::cout << "Classify result: " << result.label_text << std::endl;
  return 0;
}
