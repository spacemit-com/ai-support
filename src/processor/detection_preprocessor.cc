#include "src/processor/detection_preprocessor.h"
#include "utils/time.h"

void DetectionPreprocessor::PreprocessNanoDet(const cv::Mat &mat, 
                                              std::vector<int64_t>& input_node_dims, 
                                              std::vector<float>& input_tensor_value)
{
  const int input_height = input_node_dims.at(2);
  const int input_width = input_node_dims.at(3);

  cv::Mat resizedImageBGR,  resizedImage, preprocessedImage;
  {
#ifdef DEBUG
    TimeWatcher t("| |-- Resize unscale");
#endif
    resize_unscale(mat, resizedImage, input_height, input_width);
  }
  {
#ifdef DEBUG
    TimeWatcher t("| |-- Convert to fp32");
#endif
    resizedImage.convertTo(resizedImage, CV_32F, 1.0);
  }

  {
#ifdef DEBUG
    TimeWatcher t("| |-- Normalize");
#endif
    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    const float mean_vals[3] = {103.53, 116.28, 123.675};
    const float scale_vals[3] = {57.375, 57.12, 58.395};
    int channel = 3;
    for(int i=0;i<channel;i++)
    {
      channels[i] = (channels[i] - mean_vals[i]) / (scale_vals[i]);
      std::vector<float> data = std::vector<float>(channels[i].reshape(1, 1));
      input_tensor_value.insert(input_tensor_value.end(), data.begin(), data.end());
    }
  }
}

void DetectionPreprocessor::Preprocess(const cv::Mat &mat, 
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
    {
#ifdef DEBUG
    TimeWatcher t("| |-- Resize unscale");
#endif
      resize_unscale(mat, resizedImageBGR, input_height, input_width);
    }
          
    {
#ifdef DEBUG
    TimeWatcher t("| |-- Convert to RGB");
#endif
      // step 3: Convert the image to HWC RGB UINT8 format.
      cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::COLOR_BGR2RGB);
    }
    {
#ifdef DEBUG
    TimeWatcher t("| |-- Convert to fp32");
#endif
      resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0);
    }
    {
#ifdef DEBUG
    TimeWatcher t("| |-- Normalize");
#endif
      cv::Mat channels[3];
      cv::split(resizedImage, channels);
      const float mean_vals[3] = {116.28f, 116.28f, 116.28f};
      const float scale_vals[3] = {0.017429f, 0.017429f, 0.017429f};
      int channel = 3;
      for(int i=0;i<channel;i++)
      {
        channels[i] = (channels[i] - mean_vals[i]) * scale_vals[i];
        std::vector<float> data = std::vector<float>(channels[i].reshape(1, 1));
        input_tensor_value.insert(input_tensor_value.end(), data.begin(), data.end());
      }
    }
  }
}