#ifndef _IMAGE_CLASSIFY_H_
#define _IMAGE_CLASSIFY_H_

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <stdexcept> // To use runtime_error

#include "task/vision/image_classification_types.h"
#include "src/core/engine.h"
#include "src/utils/label_map_utils.h"
#include "src/task/vision/base_vision_task_api.h"
#include "src/processor/classification_postprocessor.h"
#include "src/processor/classification_preprocessor.h"

#include "opencv2/opencv.hpp"

class imageClassification : public BaseVisionTaskApi<ImageClassificationResult> 
{
    public:
    imageClassification():BaseVisionTaskApi<ImageClassificationResult>() {};
    ~imageClassification() {};
    int Init(const std::string modelFilepath, const std::string labelFilepath);
    void Preprocess(std::vector<float> &input_tensors,
              const cv::Mat& img_raw) override;
    ImageClassificationResult Classify(const cv::Mat &img_raw);

    protected:
    bool checkModelExtension(const std::string& filename);
    ImageClassificationResult Postprocess() override;
    private:
    void InitCheck();
    ClassificationPreprocessor preprocessor_;
    ClassificationPostprocessor postprocessor_;
    std::string instanceName_;
    std::string modelFilepath_;
    cv::Mat img_raw_;
    std::string labelFilepath_;
    std::vector<std::string> labels_;
    std::vector<Ort::Value> output_tensors_;
    std::vector<float> input_tensors_;
};

#endif