#ifndef SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TYPES_H_
#define SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TYPES_H_

#include <chrono>  // for chrono
#include <string>  // for string
#include <vector>  // for vector

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
  std::chrono::time_point<std::chrono::steady_clock> timestamp;
};

struct PoseEstimationOption {
  std::string model_path;
  int intra_threads_num = 2;
  int inter_threads_num = 2;
};

#endif  // SUPPORT_INCLUDE_TASK_VISION_POSE_ESTIMATION_TYPES_H_
