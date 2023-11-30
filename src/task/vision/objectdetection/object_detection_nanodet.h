#ifndef _OBJECT_DETECTION_NANODET_H_
#define _OBJECT_DETECTION_NANODET_H_

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

class ObjectDetectionNanodet : public BaseVisionTaskApi<std::vector<Boxi>>
{
    public:
    ObjectDetectionNanodet(): BaseVisionTaskApi<std::vector<Boxi>>() {};
    ~ObjectDetectionNanodet() {};
    std::vector<Boxi> Detect(cv::Mat &raw_img);
    int Init(std::string &modelFilepath);
    void Preprocess(std::vector<float> &input_tensors,
            cv::Mat& img_raw) override;

    protected:
    std::vector<Boxi> Postprocess() override;

    private:
    std::string instanceName_;
    std::string modelFilepath_;
    std::string labelFilepath_;
    std::vector<int64_t> inputDims_;
    std::vector<float> input_tensors_;
    DetectionPreprocessor preprocessor_;
    DetectionPostprocessor postprocessor_;
    std::vector<Boxi> result_boxes_;
    int img_height_;
    int img_width_;
};

#endif