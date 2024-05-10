#include "classification_preprocessor.h"

#include "utils/time.h"

void ClassificationPreprocessor::Preprocess(
    const cv::Mat& imageBGR, std::vector<std::vector<int64_t>> inputDims,
    std::vector<std::vector<float>>& input_tensor_values) {
  cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
  {
#ifdef DEBUG
    TimeWatcher t("| |-- Resize image");
#endif
    cv::resize(imageBGR, resizedImageBGR,
               cv::Size(static_cast<int>(inputDims[0][3]),
                        static_cast<int>(inputDims[0][2])),
               cv::InterpolationFlags::INTER_CUBIC);
  }

  // step 3: Convert the image to HWC RGB UINT8 format.
  cv::cvtColor(resizedImageBGR, resizedImageRGB,
               cv::ColorConversionCodes::COLOR_BGR2RGB);
  // step 4: Convert the image to HWC RGB float format by dividing each pixel by
  // 255.
  resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

  // step 5: Split the RGB channels from the image.
  cv::Mat channels[3];
  cv::split(resizedImage, channels);
  const float mean_vals[3] = {0.485f, 0.456f, 0.406f};
  const float scale_vals[3] = {0.229f, 0.224f, 0.225f};

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
