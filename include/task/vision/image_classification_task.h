#ifndef _IMAGE_CLASSIFICATION_TASK_H_
#define _IMAGE_CLASSIFICATION_TASK_H_

#include <string>
#include "core/types.h"
#include "opencv2/opencv.hpp"

class imageClassificationTask
{
    public:
    imageClassificationTask();
    ~imageClassificationTask() = default;
    int Init(std::string modelFilepath,   
             std::string labelFilepath);
    ImageClassificationResult Classify(cv::Mat &img_raw);
    private:
	class impl;
	std::shared_ptr<impl> pimpl_;
};

#endif