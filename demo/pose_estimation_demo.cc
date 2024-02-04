#include <iomanip>  // for: setprecision
#include <iostream>

#include "task/vision/object_detection_task.h"
#include "task/vision/pose_estimation_task.h"
#include "utils/check_utils.h"
#include "utils/time.h"
#include "utils/utils.h"

int main(int argc, char* argv[]) {
  std::vector<std::pair<int, int>> coco_17_joint_links = {
      {0, 1},   {0, 2},   {1, 3},   {2, 4},  {5, 7},  {7, 9},
      {6, 8},   {8, 10},  {5, 6},   {5, 11}, {6, 12}, {11, 12},
      {11, 13}, {13, 15}, {12, 14}, {14, 16}};
  std::vector<PosePoint> resultPoints;
  std::vector<Boxi> resultBoxes;
  std::string detFilePath, poseFilePath, imageFilepath, saveImgpath;
  cv::Mat imgRaw, img;
#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
  if (argc == 5) {
    detFilePath = argv[1];
    poseFilePath = argv[2];
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
      resize_unscale(imgRaw, img, 320, 320);
    }
    std::unique_ptr<objectDetectionTask> objectdetectiontask =
        std::unique_ptr<objectDetectionTask>(
            new objectDetectionTask(detFilePath));
    if (objectdetectiontask->getInitFlag() != 0) {
      return -1;
    }
    resultBoxes = objectdetectiontask->Detect(img).result_bboxes;
    std::unique_ptr<poseEstimationTask> poseestimationtask =
        std::unique_ptr<poseEstimationTask>(
            new poseEstimationTask(poseFilePath));
    if (poseestimationtask->getInitFlag() != 0) {
      return -1;
    }
    Boxi box;
    for (int i = 0; i < static_cast<int>(resultBoxes.size()); i++) {
      box = resultBoxes[i];
      if (box.label != 0) {
        continue;
      }
      resultPoints = poseestimationtask->Estimate(img, box).result_points;
      if (resultPoints.size()) {
        int input_height = 320;
        int input_width = 320;
        int img_height = imgRaw.rows;
        int img_width = imgRaw.cols;
        float resize_ratio = std::min(
            static_cast<float>(input_height) / static_cast<float>(img_height),
            static_cast<float>(input_width) / static_cast<float>(img_width));
        float dw = (input_width - resize_ratio * img_width) / 2;
        float dh = (input_height - resize_ratio * img_height) / 2;
        for (int i = 0; i < static_cast<int>(resultPoints.size()); i++) {
          resultPoints[i].x = (resultPoints[i].x - dw) / resize_ratio;
          resultPoints[i].y = (resultPoints[i].y - dh) / resize_ratio;
        }
      }
      for (int i = 0; i < static_cast<int>(resultPoints.size()); ++i) {
        cv::circle(imgRaw, cv::Point(resultPoints[i].x, resultPoints[i].y), 2,
                   cv::Scalar{0, 0, 255}, 2, cv::LINE_AA);
      }

      for (int i = 0; i < static_cast<int>(coco_17_joint_links.size()); ++i) {
        std::pair<int, int> joint_links = coco_17_joint_links[i];
        cv::line(imgRaw,
                 cv::Point(resultPoints[joint_links.first].x,
                           resultPoints[joint_links.first].y),
                 cv::Point(resultPoints[joint_links.second].x,
                           resultPoints[joint_links.second].y),
                 cv::Scalar{0, 255, 0}, 2, cv::LINE_AA);
      }
    }
    cv::imwrite(saveImgpath, imgRaw);
  } else {
    std::cout << "run with " << argv[0]
              << " <detConfigFilepath> <poseConfigFilepath> <imageFilepath> "
                 "<saveImgpath> "
              << std::endl;
    return -1;
  }
  return 0;
}
