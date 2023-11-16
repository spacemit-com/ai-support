#ifndef _OBJECT_DETECTION_STREAM_TASK_H_
#define _OBJECT_DETECTION_STREAM_TASK_H_

#include <string>
#include "core/types.h"
#include "utils/utils.h"
#include "opencv2/opencv.hpp"

class objectDetectionStreamTask
{
    public:
    objectDetectionStreamTask() {};
    ~objectDetectionStreamTask() {};
    int DetectVideo(std::string &instanceName, 
                    std::string &modelFilepath, 
                    int video);
};

#endif