#ifndef SUPPORT_INCLUDE_UTILS_UTILS_H_
#define SUPPORT_INCLUDE_UTILS_UTILS_H_

#include <sys/stat.h>

#include <algorithm>
#include <string>

#include "opencv2/opencv.hpp"

bool exists_check(const std::string &name);

void resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs, int target_height,
                    int target_width);

#endif  // SUPPORT_INCLUDE_UTILS_UTILS_H_
