#ifndef _OBJECT_DETECTION_H_
#define _OBJECT_DETECTION_H_

#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#include "opencv2/opencv.hpp"

#include "src/core/engine.h"
#include "src/utils/label_map_utils.h"
#include "src/task/vision/base_vision_task_api.h"
#include "src/processor/detection_preprocessor.h"
#include "src/processor/detection_postprocessor.h"

#include "task/vision/object_detection_types.h"

class ObjectDetection : public BaseVisionTaskApi<ObjectDetectionResult>
{
    public:
    ObjectDetection(): BaseVisionTaskApi<ObjectDetectionResult>() {};
    ~ObjectDetection() {};
    ObjectDetectionResult Detect(const cv::Mat &raw_img);
    ObjectDetectionResult Detect_Yolov6(const cv::Mat &raw_img);
    ObjectDetectionResult Detect_NanoDet(const cv::Mat &raw_img);
    int Initfromcommand(std::string &modelFilepath, std::string &labelFilepath);
    int Initfromconfig(std::string &configFilepath);
    void Preprocess(std::vector<float> &input_tensors,
           const cv::Mat& img_raw) override;

    protected:
    ObjectDetectionResult Postprocess() override;

    private:
    std::string instanceName_;
    std::string modelFilepath_;
    std::string labelFilepath_;
    std::vector<std::string> labels_;
    std::vector<int64_t> inputDims_;
    std::vector<float> input_tensors_;
    DetectionPreprocessor processor_;
    DetectionPostprocessor postprocessor_;
    std::vector<Boxi> result_boxes_;
    ObjectDetectionResult result_;
    int img_height_;
    int img_width_;
};

#endif