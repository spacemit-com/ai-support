#include <iomanip>  // for: setprecision
#include <iostream>

#include "object_detection.hpp"
#include "task/vision/object_detection_task.h"
#include "utils/check_utils.h"
#include "utils/time.h"
#include "utils/utils.h"

int main(int argc, char* argv[]) {
  std::vector<Boxi> resultBoxes;
  std::string filePath, modelFilepath, imageFilepath, saveImgpath,
      labelFilepath, configFilepath;
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
        if (resultBoxes[i].flag) {
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
    labelFilepath = argv[2];
    imageFilepath = argv[3];
    saveImgpath = argv[4];
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
    if (objectdetectiontask->getInitFlag() != 0) {
      return -1;
    }
    resultBoxes = objectdetectiontask->Detect(imgRaw).result_bboxes;
    {
#ifdef DEBUG
      TimeWatcher t("|-- Output result");
#endif
      for (int i = 0; i < static_cast<int>(resultBoxes.size()); i++) {
        if (resultBoxes[i].flag) {
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
        << " <modelFilepath> <labelFilepath> <imageFilepath> <saveImgpath> or "
        << argv[0] << " <configFilepath> <imageFilepath> <saveImgpath>"
        << std::endl;
    return -1;
  }
  return 0;
}
