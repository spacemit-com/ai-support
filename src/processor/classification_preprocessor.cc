#include "utils/time.h"
#include "src/processor/classification_preprocessor.h"

void ClassificationPreprocessor::Preprocess(cv::Mat &imageBGR, std::vector<int64_t> inputDims, std::vector<float>& input_tensor_value)
{
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    {
#ifdef DEBUG
        TimeWatcher t("| |-- Resize image");
#endif
        cv::resize(imageBGR, resizedImageBGR,
                cv::Size(inputDims[3], inputDims[2]),
                cv::InterpolationFlags::INTER_CUBIC);
    }
    
    // step 3: Convert the image to HWC RGB UINT8 format.
    cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);
    // step 4: Convert the image to HWC RGB float format by dividing each pixel by 255.
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    // step 5: Split the RGB channels from the image.   
    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    const float mean_vals[3] = {0.485, 0.456, 0.406};
    const float scale_vals[3] = {0.229, 0.224, 0.225};

    int channel = 3;
    for(int i=0;i<channel;i++)
    {
      channels[i] = (channels[i] - mean_vals[i]) / (scale_vals[i]);
      std::vector<float> data = std::vector<float>(channels[i].reshape(1, 1));
      input_tensor_value.insert(input_tensor_value.end(), data.begin(), data.end());
    }
}