#ifndef _OBJECT_DETECTION_TASK_H_
#define _OBJECT_DETECTION_TASK_H_

#include <string>
#include "task/vision/object_detection_types.h"
#include "opencv2/opencv.hpp"

class objectDetectionTask
{
    public:
    objectDetectionTask();
    ~objectDetectionTask() = default;
    int Init(std::string &modelFilepath, std::string &labelFilepath);
    ObjectDetectionResult Detect(const cv::Mat &raw_img);
    ObjectDetectionResult Detect_Yolov6(const cv::Mat &raw_img);
    ObjectDetectionResult Detect_NanoDet(const cv::Mat &raw_img);

    private:
	class impl;
	std::shared_ptr<impl> pimpl_;
};

#endif