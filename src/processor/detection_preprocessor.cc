#include "detection_preprocessor.h"

// Function to validate the input image file extension.
bool DetectionPreprocessor::imageFileExtension(std::string str)
{
  // is empty throw error
  if (str.empty())
    throw std::runtime_error("[ ERROR ] The image File path is empty");

  size_t pos = str.rfind('.');
  if (pos == std::string::npos)
    return false;

  std::string ext = str.substr(pos+1);

  if (ext == "jpg" || ext == "jpeg" || ext == "gif" || ext == "png" || ext == "jfif" || 
        ext == "JPG" || ext == "JPEG" || ext == "GIF" || ext == "PNG" || ext == "JFIF") {
            return true;
  }

  return false;
}

void DetectionPreprocessor::Preprocess(const std::string imageFilepath, 
                                       std::vector<int64_t>& input_node_dims, 
                                       std::vector<float>& input_tensor_value,
                                       YoloXScaleParams& scale_params,
                                       unsigned int data_format,
                                       int img_height,
                                       int img_width)
{
  cv::Mat mat = cv::imread(imageFilepath);
  if (mat.empty()) return;
  const int input_height = input_node_dims.at(2);
  const int input_width = input_node_dims.at(3);
  img_height = static_cast<int>(mat.rows);
  img_width = static_cast<int>(mat.cols);

  // resize & unscale
  cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
  resize_unscale(mat, resizedImageBGR, input_height, input_width, scale_params);
  cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::COLOR_BGR2RGB);
  // resize without padding, (Done): add padding as the official Python implementation.
  // cv::resize(canva, canva, cv::Size(input_node_dims.at(3),
  //                                  input_node_dims.at(2)));
  // (1,3,640,640) 1xCXHXW
  // step 5: Split the RGB channels from the image.   
  resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
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
  const unsigned int target_tensor_size = 3 * input_height * input_width;

  cv::dnn::blobFromImage(resizedImage, preprocessedImage);
  input_tensor_value.resize(target_tensor_size);
  input_tensor_value.assign(preprocessedImage.begin<float>(),
                          preprocessedImage.end<float>());
}

void DetectionPreprocessor::resize_unscale(const cv::Mat& mat, 
                                           cv::Mat& mat_rs,
                                           int target_height, 
                                           int target_width,
                                           YoloXScaleParams& scale_params)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                   cv::Scalar(114, 114, 114));
  // scale ratio (new / old) new_shape(h,w)
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
  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

  // record scale params.
  scale_params.r = r;
  scale_params.dw = dw;
  scale_params.dh = dh;
  scale_params.new_unpad_w = new_unpad_w;
  scale_params.new_unpad_h = new_unpad_h;
  scale_params.flag = true;
}