#ifndef SUPPORT_DEMO_UTILS_BOX_UTILS_H_
#define SUPPORT_DEMO_UTILS_BOX_UTILS_H_

#include <string>   // for std::string
#include <utility>  //for pair<>
#include <vector>

#include "opencv2/opencv.hpp"
#include "task/vision/object_detection_types.h"
#include "task/vision/pose_estimation_types.h"

static void draw_boxes_inplace(cv::Mat &mat_inplace,
                               const std::vector<Boxi> &boxes) {
  if (boxes.empty()) return;
  for (const auto &box : boxes) {
    if (box.flag) {
      cv::rectangle(mat_inplace, box.rect(), cv::Scalar(255, 255, 0), 2);
      if (box.label_text) {
        std::string label_text(box.label_text);
        label_text = label_text + ":" + std::to_string(box.score).substr(0, 4);
        cv::putText(mat_inplace, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                    .5f, cv::Scalar(0, 69, 255), 1);
      }
    }
  }
}
static void draw_points_inplace(cv::Mat &img,
                                const std::vector<PosePoint> &points) {
  std::vector<std::pair<int, int>> coco_17_joint_links = {
      {0, 1},   {0, 2},   {1, 3},   {2, 4},  {5, 7},  {7, 9},
      {6, 8},   {8, 10},  {5, 6},   {5, 11}, {6, 12}, {11, 12},
      {11, 13}, {13, 15}, {12, 14}, {14, 16}};
  for (int i = 0; i < points.size(); ++i) {
    cv::circle(img, cv::Point(points[i].x, points[i].y), 2,
               cv::Scalar{0, 0, 255}, 2, cv::LINE_AA);
  }
  for (int i = 0; i < coco_17_joint_links.size(); ++i) {
    std::pair<int, int> joint_links = coco_17_joint_links[i];
    cv::line(
        img,
        cv::Point(points[joint_links.first].x, points[joint_links.first].y),
        cv::Point(points[joint_links.second].x, points[joint_links.second].y),
        cv::Scalar{0, 255, 0}, 2, cv::LINE_AA);
  }
}
#endif  // SUPPORT_DEMO_UTILS_BOX_UTILS_H_
