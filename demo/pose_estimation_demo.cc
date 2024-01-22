#include <stdlib.h>
#include <unistd.h>  //for getopt

#include <iomanip>  // for setprecision
#include <iostream>

#include "task/vision/object_detection_task.h"
#include "task/vision/pose_estimation_task.h"
#include "utils/box_utils.h"
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
  std::string detFilePath, poseFilePath, modelFilepath, imageFilepath,
      saveImgpath, labelFilepath, configFilepath;
  bool disable_spacemit_ep{false};
  float score_threshold{0.4}, nms_threshold{0.5};
  int intra_threads_num{1};
  cv::Mat imgRaw, img;
#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
  if (argc == 6) {
    detFilePath = argv[1];
    poseFilePath = argv[2];
    imageFilepath = argv[3];
    saveImgpath = argv[4];
    labelFilepath = argv[5];
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
            new objectDetectionTask(detFilePath, labelFilepath));
    resultBoxes = objectdetectiontask->Detect(img).result_bboxes;

    std::unique_ptr<poseEstimationTask> poseestimationtask =
        std::unique_ptr<poseEstimationTask>(new poseEstimationTask(
            poseFilePath, disable_spacemit_ep, intra_threads_num));
    Boxi box;
    for (int i = 0; i < resultBoxes.size(); i++) {
      box = resultBoxes[i];
      if (box.label != 0) {
        continue;
      }
      resultPoints = poseestimationtask->Estimate(img, box).result_points;
      for (int i = 0; i < resultPoints.size(); ++i) {
        cv::circle(img, cv::Point(resultPoints[i].x, resultPoints[i].y), 2,
                   cv::Scalar{0, 0, 255}, 2, cv::LINE_AA);
      }

      for (int i = 0; i < coco_17_joint_links.size(); ++i) {
        std::pair<int, int> joint_links = coco_17_joint_links[i];
        cv::line(img,
                 cv::Point(resultPoints[joint_links.first].x,
                           resultPoints[joint_links.first].y),
                 cv::Point(resultPoints[joint_links.second].x,
                           resultPoints[joint_links.second].y),
                 cv::Scalar{0, 255, 0}, 2, cv::LINE_AA);
      }
    }
    cv::imwrite(saveImgpath, img);
  } else if (argc > 6) {
    detFilePath = argv[1];
    poseFilePath = argv[2];
    imageFilepath = argv[3];
    saveImgpath = argv[4];
    labelFilepath = argv[5];
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
            detFilePath, labelFilepath, disable_spacemit_ep, intra_threads_num,
            score_threshold, nms_threshold));
    resultBoxes = objectdetectiontask->Detect(imgRaw).result_bboxes;

    std::unique_ptr<poseEstimationTask> poseestimationtask =
        std::unique_ptr<poseEstimationTask>(new poseEstimationTask(
            poseFilePath, disable_spacemit_ep, intra_threads_num));
    Boxi box;
    for (int i = 0; i < resultBoxes.size(); i++) {
      box = resultBoxes[i];
      if (box.label != 0) {
        continue;
      }
      resultPoints = poseestimationtask->Estimate(img, box).result_points;
      for (int i = 0; i < resultPoints.size(); ++i) {
        cv::circle(img, cv::Point(resultPoints[i].x, resultPoints[i].y), 2,
                   cv::Scalar{0, 0, 255}, 2, cv::LINE_AA);
      }

      for (int i = 0; i < coco_17_joint_links.size(); ++i) {
        std::pair<int, int> joint_links = coco_17_joint_links[i];
        cv::line(img,
                 cv::Point(resultPoints[joint_links.first].x,
                           resultPoints[joint_links.first].y),
                 cv::Point(resultPoints[joint_links.second].x,
                           resultPoints[joint_links.second].y),
                 cv::Scalar{0, 255, 0}, 2, cv::LINE_AA);
      }
    }
    cv::imwrite(saveImgpath, imgRaw);
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
