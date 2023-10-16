#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

#include "image_preprocessor.h"

#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


// Function to validate the input image file extension.
bool ImagePreprocessor::imageFileExtension(std::string str)
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

void ImagePreprocessor::Preprocess(const std::string imageFilepath, std::vector<int64_t>& inputDims, std::vector<float>& input_tensors)
{
    // Validate ImageFilePath
    imageFileExtension(imageFilepath);
    if(!imageFileExtension(imageFilepath)) {
        throw std::runtime_error("[ ERROR ] The imageFilepath doesn't have correct image extension. Choose from jpeg, jpg, gif, png, PNG, jfif");
    }
    std::ifstream f(imageFilepath.c_str());
    if(!f.good()) {
        throw std::runtime_error("[ ERROR ] The imageFilepath is not set correctly or doesn't exist");
    }
    //pre-processing the Image
    // step 1: Read an image in HWC BGR UINT8 format.
    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);

    // step 2: Resize the image.
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(imageBGR, resizedImageBGR,
            cv::Size(inputDims.at(3), inputDims.at(2)),
            cv::InterpolationFlags::INTER_CUBIC);

    // step 3: Convert the image to HWC RGB UINT8 format.
    cv::cvtColor(resizedImageBGR, resizedImageRGB,
                cv::ColorConversionCodes::COLOR_BGR2RGB);
    // step 4: Convert the image to HWC RGB float format by dividing each pixel by 255.
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);

    // step 5: Split the RGB channels from the image.   
    cv::Mat channels[3];
    cv::split(resizedImage, channels);

    //step 6: Normalize each channel.
    // Normalization per channel
    // Normalization parameters obtained from
    // https://github.com/onnx/models/tree/master/vision/classification/squeezenet
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;

    //step 7: Merge the RGB channels back to the image.
    cv::merge(channels, 3, resizedImage);

    // step 8: Convert the image to CHW RGB float format.
    // HWC to CHW
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);
  
    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                            preprocessedImage.end<float>());

    input_tensors = inputTensorValues;

}