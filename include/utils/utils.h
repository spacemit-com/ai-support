#ifndef SUPPORT_INCLUDE_UTILS_UTILS_H_
#define SUPPORT_INCLUDE_UTILS_UTILS_H_

#include <algorithm>
#include <string>

#include "opencv2/opencv.hpp"
#include "task/vision/image_classification_types.h"
#include "task/vision/object_detection_types.h"
#include "task/vision/pose_estimation_types.h"

void resizeUnscale(const cv::Mat &mat, cv::Mat &mat_rs, int target_height,
                   int target_width);

int configToOption(const std::string &config_file_path,
                   PoseEstimationOption &option);

int configToOption(const std::string &config_file_path,
                   ObjectDetectionOption &option);

int configToOption(const std::string &config_file_path,
                   ImageClassificationOption &option);

#endif  // SUPPORT_INCLUDE_UTILS_UTILS_H_
