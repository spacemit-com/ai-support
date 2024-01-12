#ifndef SUPPORT_SRC_UTILS_CV2_UTILS_H_
#define SUPPORT_SRC_UTILS_CV2_UTILS_H_

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "opencv2/opencv.hpp"

enum { CHW = 0, HWC = 1 };

static cv::Mat normalize(const cv::Mat &mat, float mean, float scale) {
  cv::Mat matf;
  if (mat.type() != CV_32FC3)
    mat.convertTo(matf, CV_32FC3);
  else
    matf = mat;  // reference
  return (matf - mean) * scale;
}

static cv::Mat normalize(const cv::Mat &mat, const float *mean,
                         const float *scale) {
  cv::Mat mat_copy;
  if (mat.type() != CV_32FC3)
    mat.convertTo(mat_copy, CV_32FC3);
  else
    mat_copy = mat.clone();
  for (unsigned int i = 0; i < mat_copy.rows; ++i) {
    cv::Vec3f *p = mat_copy.ptr<cv::Vec3f>(i);
    for (unsigned int j = 0; j < mat_copy.cols; ++j) {
      p[j][0] = (p[j][0] - mean[0]) * scale[0];
      p[j][1] = (p[j][1] - mean[1]) * scale[1];
      p[j][2] = (p[j][2] - mean[2]) * scale[2];
    }
  }
  return mat_copy;
}

static void normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean,
                      float scale) {
  outmat = normalize(inmat, mean, scale);
}

static void normalize_inplace(cv::Mat &mat_inplace, float mean, float scale) {
  if (mat_inplace.type() != CV_32FC3)
    mat_inplace.convertTo(mat_inplace, CV_32FC3);
  normalize(mat_inplace, mat_inplace, mean, scale);
}

static void normalize_inplace(cv::Mat &mat_inplace, const float *mean,
                              const float *scale) {
  if (mat_inplace.type() != CV_32FC3)
    mat_inplace.convertTo(mat_inplace, CV_32FC3);
  for (unsigned int i = 0; i < mat_inplace.rows; ++i) {
    cv::Vec3f *p = mat_inplace.ptr<cv::Vec3f>(i);
    for (unsigned int j = 0; j < mat_inplace.cols; ++j) {
      p[j][0] = (p[j][0] - mean[0]) * scale[0];
      p[j][1] = (p[j][1] - mean[1]) * scale[1];
      p[j][2] = (p[j][2] - mean[2]) * scale[2];
    }
  }
}

#endif  // SUPPORT_SRC_UTILS_CV2_UTILS_H_
