#include <iostream>

#include "task/vision/image_classification_task.h"
#include "utils/json_helper.hpp"
#ifdef DEBUG
#include "utils/time.h"
#endif

static void usage(const char* exe) {
  std::cout << "Usage: \n"
            << exe << " <model_path> <label_path> <image_path>\n"
            << exe << " <config_path> <image_path>\n";
}

int main(int argc, char* argv[]) {
  if (argc != 3 && argc != 4) {
    usage(argv[0]);
    return -1;
  }

  ImageClassificationOption option;
  std::string image_file_path;
  if (argc == 3) {
    if (configToOption(argv[1], option) != 0) {
      return -1;
    }
    image_file_path = argv[2];
  } else if (argc == 4) {
    option.model_path = argv[1];
    option.label_path = argv[2];
    image_file_path = argv[3];
  }

  std::unique_ptr<ImageClassificationTask> imageclassificationtask =
      std::unique_ptr<ImageClassificationTask>(
          new ImageClassificationTask(option));
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
