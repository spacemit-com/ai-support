#include <stdlib.h>
#ifdef _WIN32
#include "getopt.h"
#else
#include <unistd.h>  // for: getopt
#endif

#include <iomanip>  // for setprecision
#include <iostream>

#include "task/vision/object_detection_task.h"
#include "utils/box_utils.h"
#include "utils/check_utils.h"
#include "utils/time.h"
#include "utils/utils.h"

int main(int argc, char* argv[]) {
  std::vector<Boxi> resultBoxes;
  std::string filePath, modelFilepath, imageFilepath, saveImgpath,
      labelFilepath, configFilepath;
  bool disable_spacemit_ep{false};
  float score_threshold{0.4}, nms_threshold{0.5};
  int intra_threads_num{1};
  cv::Mat imgRaw;
#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
  if (argc == 4) {
    filePath = argv[1];
    imageFilepath = argv[2];
    saveImgpath = argv[3];
    if (!checkImageFileExtension(imageFilepath) ||
        !checkImageFileExtension(saveImgpath)) {
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
    {
#ifdef DEBUG
      TimeWatcher t("|-- Load input data");
#endif
      imgRaw = cv::imread(imageFilepath);
    }
    std::unique_ptr<objectDetectionTask> objectdetectiontask =
        std::unique_ptr<objectDetectionTask>(new objectDetectionTask(filePath));
    resultBoxes = objectdetectiontask->Detect(imgRaw).result_bboxes;
    {
#ifdef DEBUG
      TimeWatcher t("|-- Output result");
#endif
      for (int i = 0; i < static_cast<int>(resultBoxes.size()); i++) {
        std::cout << "bbox[" << std::setw(2) << i << "]"
                  << " "
                  << "x1y1x2y2: "
                  << "(" << std::setw(4) << resultBoxes[i].x1 << ","
                  << std::setw(4) << resultBoxes[i].y1 << "," << std::setw(4)
                  << resultBoxes[i].x2 << "," << std::setw(4)
                  << resultBoxes[i].y2 << ")"
                  << ", "
                  << "score: " << std::fixed << std::setprecision(3)
                  << std::setw(4) << resultBoxes[i].score << ", "
                  << "label_text: " << resultBoxes[i].label_text << std::endl;
      }
    }
    {
#ifdef DEBUG
      TimeWatcher t("|-- Box drawing");
#endif
      draw_boxes_inplace(imgRaw, resultBoxes);
    }

    cv::imwrite(saveImgpath, imgRaw);
    // cv::imshow("detected.jpg",imgRaw);
    // cv::waitKey(0);
  } else if (argc == 5) {
    filePath = argv[1];
    imageFilepath = argv[2];
    saveImgpath = argv[3];
    labelFilepath = argv[4];
    if (!checkImageFileExtension(imageFilepath) ||
        !checkImageFileExtension(saveImgpath)) {
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
    {
#ifdef DEBUG
      TimeWatcher t("|-- Load input data");
#endif
      imgRaw = cv::imread(imageFilepath);
    }
    std::unique_ptr<objectDetectionTask> objectdetectiontask =
        std::unique_ptr<objectDetectionTask>(
            new objectDetectionTask(filePath, labelFilepath));
    resultBoxes = objectdetectiontask->Detect(imgRaw).result_bboxes;
    {
#ifdef DEBUG
      TimeWatcher t("|-- Output result");
#endif
      for (int i = 0; i < static_cast<int>(resultBoxes.size()); i++) {
        std::cout << "bbox[" << std::setw(2) << i << "]"
                  << " "
                  << "x1y1x2y2: "
                  << "(" << std::setw(4) << resultBoxes[i].x1 << ","
                  << std::setw(4) << resultBoxes[i].y1 << "," << std::setw(4)
                  << resultBoxes[i].x2 << "," << std::setw(4)
                  << resultBoxes[i].y2 << ")"
                  << ", "
                  << "score: " << std::fixed << std::setprecision(3)
                  << std::setw(4) << resultBoxes[i].score << ", "
                  << "label_text: " << resultBoxes[i].label_text << std::endl;
      }
    }
    {
#ifdef DEBUG
      TimeWatcher t("|-- Box drawing");
#endif
      draw_boxes_inplace(imgRaw, resultBoxes);
    }

    cv::imwrite(saveImgpath, imgRaw);
    // cv::imshow("detected.jpg",imgRaw);
    // cv::waitKey(0);
  } else if (argc > 5) {
    filePath = argv[1];
    imageFilepath = argv[2];
    saveImgpath = argv[3];
    labelFilepath = argv[4];
    if (!checkImageFileExtension(imageFilepath) ||
        !checkImageFileExtension(saveImgpath)) {
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
    {
#ifdef DEBUG
      TimeWatcher t("|-- Load input data");
#endif
      imgRaw = cv::imread(imageFilepath);
    }
    int o;
    const char* optstring = "d:t:s:n:";
    while ((o = getopt(argc, argv, optstring)) != -1) {
      switch (o) {
        case 'd':
          disable_spacemit_ep = atoi(optarg);
          break;
        case 't':
          intra_threads_num = atoi(optarg);
          break;
        case 's':
          score_threshold = atof(optarg);
          break;
        case 'n':
          nms_threshold = atof(optarg);
          break;
        case '?':
          std::cout << "[ERROR] Unsupported usage" << std::endl;
          break;
      }
    }
    std::unique_ptr<objectDetectionTask> objectdetectiontask =
        std::unique_ptr<objectDetectionTask>(new objectDetectionTask(
            filePath, labelFilepath, disable_spacemit_ep, intra_threads_num,
            score_threshold, nms_threshold));
    resultBoxes = objectdetectiontask->Detect(imgRaw).result_bboxes;
    {
#ifdef DEBUG
      TimeWatcher t("|-- Output result");
#endif
      for (int i = 0; i < static_cast<int>(resultBoxes.size()); i++) {
        std::cout << "bbox[" << std::setw(2) << i << "]"
                  << " "
                  << "x1y1x2y2: "
                  << "(" << std::setw(4) << resultBoxes[i].x1 << ","
                  << std::setw(4) << resultBoxes[i].y1 << "," << std::setw(4)
                  << resultBoxes[i].x2 << "," << std::setw(4)
                  << resultBoxes[i].y2 << ")"
                  << ", "
                  << "score: " << std::fixed << std::setprecision(3)
                  << std::setw(4) << resultBoxes[i].score << ", "
                  << "label_text: " << resultBoxes[i].label_text << std::endl;
      }
    }
    {
#ifdef DEBUG
      TimeWatcher t("|-- Box drawing");
#endif
      draw_boxes_inplace(imgRaw, resultBoxes);
    }

    cv::imwrite(saveImgpath, imgRaw);
    // cv::imshow("detected.jpg",imgRaw);
    // cv::waitKey(0);
  } else {
    std::cout
        << "run with " << argv[0]
        << " <modelFilepath> <imageFilepath> <saveImgpath> <labelFilepath> "
           "option(-d <disable_spacemit_ep>) option(-t <intra_threads_num>) "
           "option(-s score_threshold) option(-n nms_threshold) or "
        << argv[0] << " <configFilepath> <imageFilepath> <saveImgpath>"
        << std::endl;
    return -1;
  }
  return 0;
}
