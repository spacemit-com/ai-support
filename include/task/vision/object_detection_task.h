#ifndef _OBJECT_DETECTION_TASK_H_
#define _OBJECT_DETECTION_TASK_H_

#include <string>
#include "core/types.h"
#include "utils/json.hpp"
#include "opencv2/opencv.hpp"

using json=nlohmann::json;

class objectDetectionTask
{
    public:
    objectDetectionTask();
    ~objectDetectionTask() = default;
    int Init(std::string &modelFilepath);
    int Init(json config);
    std::vector<Boxi> Detect(cv::Mat &raw_img);
    std::vector<Boxi> Detect_Yolov6(cv::Mat &raw_img);
    std::vector<Boxi> Detect_NanoDet(cv::Mat &raw_img);
    std::vector<Boxi> Inference(cv::Mat &raw_img);

    private:
	class impl;
	std::shared_ptr<impl> pimpl_;
};

#endif