#ifndef _IMAGE_CLASSIFICATION_TASK_H_
#define _IMAGE_CLASSIFICATION_TASK_H_

#include "task/vision/image_classification_types.h"

#include <memory>   // for: shared_ptr
#include <string>

#include "opencv2/opencv.hpp"

class imageClassificationTask
{
    public:
    imageClassificationTask(const std::string& filePath, const std::string& labelFilepath);
    ~imageClassificationTask() = default;
    ImageClassificationResult Classify(const cv::Mat &img_raw);
    private:
	class impl;
	std::shared_ptr<impl> pimpl_;
};

#endif
