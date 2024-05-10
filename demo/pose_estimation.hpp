#ifndef SUPPORT_DEMO_POSE_ESTIMATION_HPP_
#define SUPPORT_DEMO_POSE_ESTIMATION_HPP_

#include <string>   // for std::string
#include <utility>  // for std::pair
#include <vector>

#include "opencv2/opencv.hpp"
#include "task/vision/pose_estimation_types.h"

static std::vector<std::pair<int, int>> coco_17_joint_links = {
    {0, 1}, {0, 2},  {1, 3},  {2, 4},   {5, 7},   {7, 9},   {6, 8},   {8, 10},
    {5, 6}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}};

inline void draw_lines_inplace(cv::Mat &img,
                               const std::vector<PosePoint> &points) {
  for (size_t i = 0; i < coco_17_joint_links.size(); ++i) {
    std::pair<int, int> joint_links = coco_17_joint_links[i];
    cv::line(
        img,
        cv::Point(points[joint_links.first].x, points[joint_links.first].y),
        cv::Point(points[joint_links.second].x, points[joint_links.second].y),
        cv::Scalar{0, 255, 0}, 2, cv::LINE_AA);
  }
}

inline void draw_points_inplace(cv::Mat &img,
                                const std::vector<PosePoint> &points) {
  for (size_t i = 0; i < points.size(); ++i) {
    cv::circle(img, cv::Point(points[i].x, points[i].y), 2,
               cv::Scalar{0, 0, 255}, 2, cv::LINE_AA);
  }
  draw_lines_inplace(img, points);
}

#endif  // SUPPORT_DEMO_POSE_ESTIMATION_HPP_
