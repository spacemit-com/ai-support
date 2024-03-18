#include <iomanip>  // for: setprecision
#include <iostream>

#include "task/vision/object_detection_task.h"
#include "task/vision/pose_estimation_task.h"
#include "utils/time.h"
#include "utils/utils.h"

int main(int argc, char* argv[]) {
  std::vector<std::pair<int, int>> coco_17_joint_links = {
      {0, 1},   {0, 2},   {1, 3},   {2, 4},  {5, 7},  {7, 9},
      {6, 8},   {8, 10},  {5, 6},   {5, 11}, {6, 12}, {11, 12},
      {11, 13}, {13, 15}, {12, 14}, {14, 16}};
  std::vector<PosePoint> points;
  std::vector<Boxi> bboxes;
  std::string det_file_path, pose_file_path, image_file_path, save_img_path;
  PoseEstimationOption estimation_option;
  ObjectDetectionOption detection_option;
  std::unique_ptr<ObjectDetectionTask> objectdetectiontask;
  std::unique_ptr<PoseEstimationTask> poseestimationtask;
  cv::Mat img_raw, img;
#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
  if (argc == 5) {
    det_file_path = argv[1];
    pose_file_path = argv[2];
    image_file_path = argv[3];
    save_img_path = argv[4];
    objectdetectiontask = std::unique_ptr<ObjectDetectionTask>(
        new ObjectDetectionTask(det_file_path));
    poseestimationtask = std::unique_ptr<PoseEstimationTask>(
        new PoseEstimationTask(pose_file_path));
  } else if (argc == 6) {
    detection_option.model_path = argv[1];
    detection_option.label_path = argv[2];
    estimation_option.model_path = argv[3];
    image_file_path = argv[4];
    save_img_path = argv[5];
    objectdetectiontask = std::unique_ptr<ObjectDetectionTask>(
        new ObjectDetectionTask(detection_option));
    poseestimationtask = std::unique_ptr<PoseEstimationTask>(
        new PoseEstimationTask(estimation_option));
  } else {
    std::cout << "Please run with " << argv[0]
              << " <det_model_file_path> <det_label_file_path> "
                 "<pose_model_file_path> <image_file_path> "
                 "<save_img_path> or "
              << argv[0]
              << " <det_config_file_path> <pose_config_file_path> "
                 "<image_file_path> <save_img_path>"
              << std::endl;
    return -1;
  }
  {
#ifdef DEBUG
    TimeWatcher t("|-- Load input data");
#endif
    img_raw = cv::imread(image_file_path);
    if (img_raw.empty()) {
      std::cout << "[ ERROR ] Read image failed" << std::endl;
      return -1;
    }
    resizeUnscale(img_raw, img, 320, 320);
  }
  if (objectdetectiontask->getInitFlag() != 0) {
    return -1;
  }
  bboxes = objectdetectiontask->Detect(img).result_bboxes;
  if (poseestimationtask->getInitFlag() != 0) {
    return -1;
  }
  Boxi box;
  for (size_t i = 0; i < bboxes.size(); i++) {
    box = bboxes[i];
    if (box.label != 0) {
      continue;
    }
    points = poseestimationtask->Estimate(img, box).result_points;
    if (points.size()) {
      int input_height = 320;
      int input_width = 320;
      int img_height = img_raw.rows;
      int img_width = img_raw.cols;
      float resize_ratio = std::min(
          static_cast<float>(input_height) / static_cast<float>(img_height),
          static_cast<float>(input_width) / static_cast<float>(img_width));
      float dw = (input_width - resize_ratio * img_width) / 2;
      float dh = (input_height - resize_ratio * img_height) / 2;
      for (size_t i = 0; i < points.size(); i++) {
        points[i].x = (points[i].x - dw) / resize_ratio;
        points[i].y = (points[i].y - dh) / resize_ratio;
      }
    }
    for (size_t i = 0; i < points.size(); ++i) {
      cv::circle(img_raw, cv::Point(points[i].x, points[i].y), 2,
                 cv::Scalar{0, 0, 255}, 2, cv::LINE_AA);
    }

    for (size_t i = 0; i < coco_17_joint_links.size(); ++i) {
      std::pair<int, int> joint_links = coco_17_joint_links[i];
      cv::line(
          img_raw,
          cv::Point(points[joint_links.first].x, points[joint_links.first].y),
          cv::Point(points[joint_links.second].x, points[joint_links.second].y),
          cv::Scalar{0, 255, 0}, 2, cv::LINE_AA);
    }
  }
  try {
    cv::imwrite(save_img_path, img_raw);
  } catch (cv::Exception& e) {
    std::cout << "[ ERROR ] Write result image failed : " << e.what()
              << std::endl;
    return -1;
  }
  return 0;
}
