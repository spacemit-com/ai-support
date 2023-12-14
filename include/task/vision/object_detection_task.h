#ifndef _OBJECT_DETECTION_TASK_H_
#define _OBJECT_DETECTION_TASK_H_

#include "task/vision/object_detection_types.h"

#include <memory>   // for: shared_ptr
#include <string>

#include "opencv2/opencv.hpp"

class objectDetectionTask
{
    public:
    objectDetectionTask(const std::string &filePath, const std::string& labelFilepath);
    objectDetectionTask(const std::string &filePath);
    ~objectDetectionTask() = default;
    ObjectDetectionResult Detect(const cv::Mat &raw_img);

    private:
	class impl;
	std::shared_ptr<impl> pimpl_;
};

#endif
