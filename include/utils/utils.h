#ifndef SUPPORT_INCLUDE_UTILS_UTILS_H_
#define SUPPORT_INCLUDE_UTILS_UTILS_H_

#include "opencv2/opencv.hpp"

void resizeUnscale(const cv::Mat &mat, cv::Mat &mat_rs, int target_height,
                   int target_width);

#endif  // SUPPORT_INCLUDE_UTILS_UTILS_H_
