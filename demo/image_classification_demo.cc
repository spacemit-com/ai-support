#include <stdlib.h>
#include <unistd.h>  //for getopt

#include <iostream>

#include "task/vision/image_classification_task.h"
#include "utils/check_utils.h"
#include "utils/time.h"
#include "utils/utils.h"

int main(int argc, char* argv[]) {
  std::string filePath, labelFilepath, imageFilepath;
  bool disable_spacemit_ep{false};
  int intra_threads_num{1};
  if (argc == 4) {
    filePath = argv[1];
    labelFilepath = argv[2];
    imageFilepath = argv[3];
  } else if (argc > 4) {
    filePath = argv[1];
    labelFilepath = argv[2];
    imageFilepath = argv[3];
    int o;
    const char* optstring = "d:t:";
    while ((o = getopt(argc, argv, optstring)) != -1) {
      switch (o) {
        case 'd':
          disable_spacemit_ep = atoi(optarg);
          break;
        case 't':
          intra_threads_num = atoi(optarg);
          break;
        case '?':
          std::cout << "[ERROR] Unsupported usage" << std::endl;
          break;
      }
    }
    if (!checkImageFileExtension(imageFilepath)) {
      std::cout << "[ ERROR ] The ImageFilepath is not correct. Make sure you "
                   "are setting the path to an imgae file (.jpg/.jpeg/.png)"
                << std::endl;
      return -1;
    }
    if (!exists_check(imageFilepath)) {
      std::cout << "[ ERROR ] The Image File does not exist. Make sure you are "
                   "setting the correct path to the file"
                << std::endl;
      return -1;
    }
  } else {
    std::cout << "run with " << argv[0]
              << " <modelFilepath> <labelFilepath> <imageFilepath> option(-d "
                 "<disable_spacemit_ep>) option(-t <intra_threads_num>)"
              << std::endl;
    return -1;
  }
  cv::Mat imgRaw;
  std::unique_ptr<imageClassificationTask> imageclassification =
      std::unique_ptr<imageClassificationTask>(new imageClassificationTask(
          filePath, labelFilepath, disable_spacemit_ep, intra_threads_num));
#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
  {
#ifdef DEBUG
    TimeWatcher t("|-- Load input data");
#endif
    imgRaw = cv::imread(imageFilepath);
  }
  ImageClassificationResult result = imageclassification->Classify(imgRaw);
  std::cout << "classify result: " << result.label_text << std::endl;
  return 0;
}
