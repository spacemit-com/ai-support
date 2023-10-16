#ifndef _OBJECT_DETECTION_H_
#define _OBJECT_DETECTION_H_

#include <string>
#include <vector>
#include <iostream>

#include "yolox_config.h"
#include "base_vision_task_api.h"
#include "engine.h"
#include "label_map_utils.h"
#include "detection_preprocessor.h"
#include "detection_postprocessor.h"

class ObjectDetection : public BaseVisionTaskApi
{
    public:
    explicit ObjectDetection(std::unique_ptr<Engine> engine)
        : BaseVisionTaskApi{std::move(engine)} {}
    ~ObjectDetection() {};
    void Detect();

    protected:
    void Postprocess() override;

    private:
    void Init();
    YoloXScaleParams scaleparams_;
    std::string instanceName_;
    std::string modelFilepath_;
    std::string imageFilepath_;
    std::string labelFilepath_;
    std::vector<std::string> labels_;
    std::vector<float> input_tensors_;
    DetectionPreprocessor processor_;
    DetectionPostprocessor postprocessor_;
    std::vector<Boxf> detected_boxes_;
    int img_height_;
    int img_width_;
};

#endif