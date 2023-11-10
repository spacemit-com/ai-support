#ifndef _IMAGE_CLASSIFICATION_TASK_H_
#define _IMAGE_CLASSIFICATION_TASK_H_

#include <string>
#include "opencv2/opencv.hpp"

class imageClassificationTask
{
    public:
    imageClassificationTask() {};
    ~imageClassificationTask() {};
    std::string Classify(std::string instanceName, 
                         std::string modelFilepath, 
                         cv::Mat &img_raw, 
                         std::string labelFilepath);
};

#endif