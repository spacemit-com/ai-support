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

#include "classification_postprocessor.h"
#include "engine.h"
#include "label_map_utils.h"
#include "base_vision_task_api.h"
#include "task_api_factory.h"

#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

class ImageClassify : public BaseVisionTaskApi
{
    public:
    explicit ImageClassify(std::unique_ptr<Engine> engine)
        : BaseVisionTaskApi{std::move(engine)} {}

    ~ImageClassify() {};
    void Classify(std::string instanceName, std::string modelFilepath, cv::Mat &img_raw, std::string labelFilepath);

    protected:
    bool checkModelExtension(const std::string& filename);
    void Postprocess() override;

    private:
    void Init(std::string instanceName, std::string modelFilepath, cv::Mat &img_raw, std::string labelFilepath);
    void InitCheck();
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