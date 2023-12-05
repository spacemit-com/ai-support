#ifndef _CV2_UTILS_H_
#define _CV2_UTILS_H_

#include <vector>
#include <stdexcept>
#include <iostream>

#include "opencv2/opencv.hpp"

enum
{
CHW = 0, HWC = 1
};

extern void resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width);

extern cv::Mat normalize(const cv::Mat &mat, float mean, float scale);

extern cv::Mat normalize(const cv::Mat &mat, const float mean[3], const float scale[3]);

extern void normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean, float scale);

extern void normalize_inplace(cv::Mat &mat_inplace, float mean, float scale);

extern void normalize_inplace(cv::Mat &mat_inplace, const float mean[3], const float scale[3]);

#endif