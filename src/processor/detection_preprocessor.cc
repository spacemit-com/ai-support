#include "src/processor/detection_preprocessor.h"

void DetectionPreprocessor::Preprocess(cv::Mat &mat, 
                                       std::vector<int64_t>& input_node_dims, 
                                       std::vector<float>& input_tensor_value,
                                       unsigned int data_format)
{
  if (mat.empty()) return;
  if(data_format == 1)
  {
    const int input_height = input_node_dims.at(1);
    const int input_width = input_node_dims.at(2);

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
  else
  {
    const int input_height = input_node_dims.at(2);
    const int input_width = input_node_dims.at(3);
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();
    resize_unscale(mat, resizedImageBGR, input_height, input_width);
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    std::cout << "resize Latency: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end0 - begin0).count()
                << " ms" << std::endl;
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    // step 3: Convert the image to HWC RGB UINT8 format.
    cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::COLOR_BGR2RGB);
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    std::cout << "cv::cvtColor Latency: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count()
                << " ms" << std::endl;
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0);
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::cout << "to fp32 Latency: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count()
                << " ms" << std::endl;
    // step 5: Split the BGR channels from the image. 
    std::chrono::steady_clock::time_point begin3 = std::chrono::steady_clock::now();
    cv::Mat channels[3];
    cv::split(resizedImage, channels);

    //step 6: Normalize each channel.
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet

    const float mean_vals[3] = {116.28f, 116.28f, 116.28f};
    const float scale_vals[3] = {0.017429f, 0.017429f, 0.017429f};
    int channel = 3;
    for(int i=0;i<channel;i++)
    {
      channels[i] = (channels[i] - mean_vals[i]) * scale_vals[i];
    }
    //step 7: Merge the RGB channels back to the image.
    cv::merge(channels, 3, resizedImage);
    std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
    std::cout << "normalize Latency: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - begin3).count()
                << " ms" << std::endl;

    std::chrono::steady_clock::time_point begin4 = std::chrono::steady_clock::now();
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);
    std::chrono::steady_clock::time_point end4 = std::chrono::steady_clock::now();
    std::cout << "cv::dnn::blobFromImage Latency: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end4 - begin4).count()
                << " ms" << std::endl;  
    std::chrono::steady_clock::time_point begin5 = std::chrono::steady_clock::now();
    size_t inputTensorSize = vectorProduct(input_node_dims);
    input_tensor_value.resize(inputTensorSize);
    input_tensor_value.assign(preprocessedImage.begin<float>(),
                            preprocessedImage.end<float>());
    std::chrono::steady_clock::time_point end5 = std::chrono::steady_clock::now();
    std::cout << "cv::Mat to std::vector Latency: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end5 - begin5).count()
                << " ms" << std::endl;  
  }
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
                   cv::Scalar(0, 0, 0));
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