#include "detection_preprocessor.h"

void DetectionPreprocessor::Preprocess(cv::Mat &mat, 
                                       std::vector<int64_t>& input_node_dims, 
                                       std::vector<float>& input_tensor_value,
                                       unsigned int data_format,
                                       int img_height,
                                       int img_width)
{
  if (mat.empty()) return;
  const int input_height = input_node_dims.at(1);
  const int input_width = input_node_dims.at(2);
  img_height = static_cast<int>(mat.rows);
  img_width = static_cast<int>(mat.cols);

  // resize & unscale
  cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
  resize_unscale(mat, resizedImageBGR, input_height, input_width);
  cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::COLOR_BGR2RGB);
  // resize without padding, (Done): add padding as the official Python implementation.
  // cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
  //                                  input_node_dims.at(2)));
  // (1,3,640,640) 1xCXHXW
  // step 5: Split the RGB channels from the image.   
  resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
  /*
  cv::Mat channels[3];
  cv::split(resizedImage, channels);

  //step 6: Normalize each channel.
  // Normalization per channel
  // Normalization parameters obtained from
  // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
  channels[0] = (channels[0] - mean_vals[0]) / scale_vals[0];
  channels[1] = (channels[1] - mean_vals[1]) / scale_vals[1];
  channels[2] = (channels[2] - mean_vals[2]) / scale_vals[2];
 
  cv::merge(channels, 3, resizedImage);
  */
  const unsigned int target_tensor_size = 3 * input_height * input_width;
  input_tensor_value.resize(target_tensor_size);
  std::memcpy(input_tensor_value.data(), resizedImage.data, target_tensor_size * sizeof(float));
}

void DetectionPreprocessor::resize_unscale(const cv::Mat& mat, 
                                           cv::Mat& mat_rs,
                                           int target_height, 
                                           int target_width)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);
  //std::cout<<img_height<<" "<<img_width<<std::endl;
  mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                   cv::Scalar(128, 128, 128));
  // scale ratio (new / old) new_shape(h,w)
  //std::cout<<target_width<<" "<<target_height<<std::endl;
  float w_r = (float) target_width / (float) img_width;
  float h_r = (float) target_height / (float) img_height;
  float r = std::min(w_r, h_r);
  // compute padding
  int new_unpad_w = static_cast<int>((float) img_width * r); // floor
  int new_unpad_h = static_cast<int>((float) img_height * r); // floor
  int pad_w = target_width - new_unpad_w; // >=0
  int pad_h = target_height - new_unpad_h; // >=0

  int dw = pad_w / 2;
  int dh = pad_h / 2;

  // resize with unscaling
  cv::Mat new_unpad_mat;
  // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
  cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
  //std::cout<<new_unpad_w<<new_unpad_h<<std::endl;
  //std::cout<<"new_unpad_mat.rows"<<new_unpad_mat.rows<<std::endl;
  //std::cout<<"new_unpad_mat.cols"<<new_unpad_mat.cols<<std::endl;
  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));
}