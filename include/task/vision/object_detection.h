#ifndef _OBJECT_DETECTION_H_
#define _OBJECT_DETECTION_H_

#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#include "opencv2/opencv.hpp"

#include "core/types.h"
#include "src/core/engine.h"
#include "src/task/vision/base_vision_task_api.h"
#include "src/processor/detection_preprocessor.h"
#include "src/processor/detection_postprocessor.h"

class ObjectDetection : public BaseVisionTaskApi<std::vector<Boxi>>
{
    public:
    explicit ObjectDetection(std::unique_ptr<Engine> engine)
        : BaseVisionTaskApi{std::move(engine)} {}
    ~ObjectDetection() {};
    std::vector<Boxi> Detect(std::string &instanceName, std::string &modelFilepath, cv::Mat &raw_img);

    protected:
    std::vector<Boxi> Postprocess() override;

    private:
    void Init(std::string &instanceName, std::string &modelFilepath, cv::Mat &raw_img);
    std::string instanceName_;
    std::string modelFilepath_;
    std::string labelFilepath_;
    std::vector<std::string> labels_;
    std::vector<float> input_tensors_;
    DetectionPreprocessor processor_;
    DetectionPostprocessor postprocessor_;
    std::vector<Boxi> result_boxes_;
    int img_height_;
    int img_width_;
};

#endif