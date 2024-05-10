#include "cv2_utils.h"

#include "opencv2/opencv.hpp"

cv::Mat normalize(const cv::Mat &mat, float mean, float scale) {
  cv::Mat matf;
  if (mat.type() != CV_32FC3) {
    mat.convertTo(matf, CV_32FC3);
  } else {
    matf = mat;  // reference
  }
  return (matf - mean) * scale;
}

cv::Mat normalize(const cv::Mat &mat, const float *mean, const float *scale) {
  cv::Mat mat_copy;
  if (mat.type() != CV_32FC3) {
    mat.convertTo(mat_copy, CV_32FC3);
  } else {
    mat_copy = mat.clone();
  }
  for (int i = 0; i < mat_copy.rows; ++i) {
    cv::Vec3f *p = mat_copy.ptr<cv::Vec3f>(i);
    for (int j = 0; j < mat_copy.cols; ++j) {
      p[j][0] = (p[j][0] - mean[0]) * scale[0];
      p[j][1] = (p[j][1] - mean[1]) * scale[1];
      p[j][2] = (p[j][2] - mean[2]) * scale[2];
    }
  }
  return mat_copy;
}

void normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean, float scale) {
  outmat = normalize(inmat, mean, scale);
}

void normalize_inplace(cv::Mat &mat_inplace, float mean, float scale) {
  if (mat_inplace.type() != CV_32FC3) {
    mat_inplace.convertTo(mat_inplace, CV_32FC3);
  }
  normalize(mat_inplace, mat_inplace, mean, scale);
}

void normalize_inplace(cv::Mat &mat_inplace, const float *mean,
                       const float *scale) {
  if (mat_inplace.type() != CV_32FC3) {
    mat_inplace.convertTo(mat_inplace, CV_32FC3);
  }
  for (int i = 0; i < mat_inplace.rows; ++i) {
    cv::Vec3f *p = mat_inplace.ptr<cv::Vec3f>(i);
    for (int j = 0; j < mat_inplace.cols; ++j) {
      p[j][0] = (p[j][0] - mean[0]) * scale[0];
      p[j][1] = (p[j][1] - mean[1]) * scale[1];
      p[j][2] = (p[j][2] - mean[2]) * scale[2];
    }
  }
}

cv::Mat GetAffineTransform(float center_x, float center_y, float scale_width,
                           float scale_height, int output_image_width,
                           int output_image_height, bool inverse) {
  // solve the affine transformation matrix

  // get the three points corresponding to the source picture and the target
  // picture
  cv::Point2f src_point_1;
  src_point_1.x = center_x;
  src_point_1.y = center_y;

  cv::Point2f src_point_2;
  src_point_2.x = center_x - scale_height * 0;
  src_point_2.y = center_y - scale_width * 0.5;

  cv::Point2f src_point_3;
  src_point_3.x = src_point_2.x - (src_point_1.y - src_point_2.y);
  src_point_3.y = src_point_2.y + (src_point_1.x - src_point_2.x);

  float alphapose_image_center_x = output_image_width / 2;
  float alphapose_image_center_y = output_image_height / 2;

  cv::Point2f dst_point_1;
  dst_point_1.x = alphapose_image_center_x;
  dst_point_1.y = alphapose_image_center_y;

  cv::Point2f dst_point_2;
  dst_point_2.x = alphapose_image_center_x;
  dst_point_2.y = alphapose_image_center_y - output_image_width * 0.5;

  cv::Point2f dst_point_3;
  dst_point_3.x = dst_point_2.x - (dst_point_1.y - dst_point_2.y);
  dst_point_3.y = dst_point_2.y + (dst_point_1.x - dst_point_2.x);

  cv::Point2f srcPoints[3];
  srcPoints[0] = src_point_1;
  srcPoints[1] = src_point_2;
  srcPoints[2] = src_point_3;

  cv::Point2f dstPoints[3];
  dstPoints[0] = dst_point_1;
  dstPoints[1] = dst_point_2;
  dstPoints[2] = dst_point_3;

  // get affine matrix
  cv::Mat affineTransform;
  if (inverse) {
    affineTransform = cv::getAffineTransform(dstPoints, srcPoints);
  } else {
    affineTransform = cv::getAffineTransform(srcPoints, dstPoints);
  }

  return affineTransform;
}
