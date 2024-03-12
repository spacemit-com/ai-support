#include "src/processor/detection_preprocessor.h"

#include "utils/time.h"
#include "utils/utils.h"

void DetectionPreprocessor::PreprocessNanoDetPlus(
    const cv::Mat& mat, std::vector<std::vector<int64_t>>& input_node_dims,
    std::vector<std::vector<float>>& input_tensor_values) {
  const int input_height = (int)input_node_dims[0][2];
  const int input_width = (int)input_node_dims[0][3];
  cv::Mat resizedImageBGR, resizedImage, preprocessedImage;
  {
#ifdef DEBUG
    TimeWatcher t("| |-- Resize unscale");
#endif
    if (input_height != mat.cols || input_width != mat.rows) {
      resizeUnscale(mat, resizedImage, input_height, input_width);
    } else {
      resizedImage = mat;
    }
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
    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float scale_vals[3] = {57.375f, 57.12f, 58.395f};
    int channel = 3;
    std::vector<float> input_tensor_value;
    for (int i = 0; i < channel; i++) {
      channels[i] = (channels[i] - mean_vals[i]) / (scale_vals[i]);
      std::vector<float> data = std::vector<float>(channels[i].reshape(1, 1));
      input_tensor_value.insert(input_tensor_value.end(), data.begin(),
                                data.end());
    }
    input_tensor_values.push_back(input_tensor_value);
  }
}

void DetectionPreprocessor::Preprocess(
    const cv::Mat& mat, std::vector<std::vector<int64_t>>& input_node_dims,
    std::vector<std::vector<float>>& input_tensor_values,
    unsigned int data_format) {
  if (mat.empty()) return;
  if (data_format == 1) {
    const int input_height = static_cast<int>(input_node_dims[0][1]);
    const int input_width = static_cast<int>(input_node_dims[0][2]);

    // resize & unscale
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    {
#ifdef DEBUG
      TimeWatcher t("| |-- Resize unscale");
#endif
      if (input_height != mat.cols || input_width != mat.rows) {
        resizeUnscale(mat, resizedImageBGR, input_height, input_width);
      } else {
        resizedImageBGR = mat;
      }
    }
    cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
    const unsigned int target_tensor_size = 3 * input_height * input_width;
    std::vector<float> input_tensor_value;
    input_tensor_value.resize(target_tensor_size);
    std::memcpy(input_tensor_value.data(), resizedImage.data,
                target_tensor_size * sizeof(float));
    input_tensor_values.push_back(input_tensor_value);
  } else {
    const int input_height = static_cast<int>(input_node_dims[0][2]);
    const int input_width = static_cast<int>(input_node_dims[0][3]);

    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    {
#ifdef DEBUG
      TimeWatcher t("| |-- Resize unscale");
#endif
      if (input_height != mat.cols || input_width != mat.rows) {
        resizeUnscale(mat, resizedImageBGR, input_height, input_width);
      } else {
        resizedImageBGR = mat;
      }
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
      resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
    }
    {
#ifdef DEBUG
      TimeWatcher t("| |-- Normalize");
#endif
      cv::Mat channels[3];
      cv::split(resizedImage, channels);
      // const float mean_vals[3] = {116.28f, 116.28f, 116.28f};
      // const float scale_vals[3] = {0.017429f, 0.017429f, 0.017429f};
      int channel = 3;
      std::vector<float> input_tensor_value;
      for (int i = 0; i < channel; i++) {
        // channels[i] = (channels[i] - mean_vals[i]) * scale_vals[i];
        std::vector<float> data = std::vector<float>(channels[i].reshape(1, 1));
        input_tensor_value.insert(input_tensor_value.end(), data.begin(),
                                  data.end());
      }
      input_tensor_values.push_back(input_tensor_value);
    }
  }
}
