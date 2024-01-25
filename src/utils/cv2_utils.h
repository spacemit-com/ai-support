#ifndef SUPPORT_SRC_UTILS_CV2_UTILS_H_
#define SUPPORT_SRC_UTILS_CV2_UTILS_H_

#include <vector>

#include "opencv2/opencv.hpp"

enum { CHW = 0, HWC = 1 };

cv::Mat normalize(const cv::Mat &mat, float mean, float scale);

cv::Mat normalize(const cv::Mat &mat, const float *mean, const float *scale);

void normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean, float scale);

void normalize_inplace(cv::Mat &mat_inplace, float mean, float scale);

void normalize_inplace(cv::Mat &mat_inplace, const float *mean,
                       const float *scale);

cv::Mat GetAffineTransform(float center_x, float center_y, float scale_width,
                           float scale_height, int output_image_width,
                           int output_image_height, bool inverse = false);

#endif  // SUPPORT_SRC_UTILS_CV2_UTILS_H_
