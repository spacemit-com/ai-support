#include "task/vision/image_classification_task.h"
#include "src/task/vision/imageclassification/image_classification.h"

class imageClassificationTask::impl {
    public:
    std::unique_ptr<imageClassification> imageclassification_;
};

imageClassificationTask::imageClassificationTask(std::string& filePath, std::string& labelFilepath) : pimpl_(std::make_unique<impl>())
{
    pimpl_->imageclassification_ = std::unique_ptr<imageClassification>(new imageClassification());
    int flag = pimpl_->imageclassification_->Init(filePath, labelFilepath);
}

ImageClassificationResult imageClassificationTask::Classify(const cv::Mat &img_raw)
{    
    return pimpl_->imageclassification_->Classify(img_raw);
}