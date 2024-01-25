#ifndef SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TYPES_H_
#define SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TYPES_H_

#include <chrono>  //for chrono
#include <vector>  //for vector

struct PosePoint {
  int x;
  int y;
  float score;

  PosePoint() {
    x = 0;
    y = 0;
    score = 0.0;
  }
};

typedef PosePoint Vector2D;

struct PoseEstimationResult {
  std::vector<PosePoint> result_points;
  std::chrono::time_point< std::chrono::steady_clock > timestamp;
};

#endif  // SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TYPES_H_
