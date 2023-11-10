#ifndef _OBJECT_DETECTION_TASK_H_
#define _OBJECT_DETECTION_TASK_H_

#include <string>
#include "core/types.h"
#include "opencv2/opencv.hpp"

class objectDetectionTask
{
    public:
    objectDetectionTask() {};
    ~objectDetectionTask() {};
    std::vector<Boxi> Detect(std::string &instanceName, 
                             std::string &modelFilepath, 
                             cv::Mat &raw_img);
};

#endif