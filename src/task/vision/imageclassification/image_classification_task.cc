#include "task/vision/image_classification_task.h"
#include "src/task/vision/imageclassification/image_classification.h"

class imageClassificationTask::impl {
    public:
    std::unique_ptr<imageClassification> imageclassification_;
};

imageClassificationTask::imageClassificationTask() : pimpl_(std::make_unique<impl>()) {}

int imageClassificationTask::Init(std::string modelFilepath,
                                  std::string labelFilepath)
{
    pimpl_->imageclassification_ = std::unique_ptr<imageClassification>(new imageClassification());
    return pimpl_->imageclassification_->Init(modelFilepath, labelFilepath);
}

ImageClassificationResult imageClassificationTask::Classify(cv::Mat &img_raw)
{    
    return pimpl_->imageclassification_->Classify(img_raw);
}