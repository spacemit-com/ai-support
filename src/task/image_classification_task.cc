#include "task/vision/image_classification_task.h"
#include "src/task/vision/image_classification.h"

std::string imageClassificationTask::Classify(std::string instanceName, 
                                                std::string modelFilepath, 
                                                cv::Mat &img_raw, 
                                                std::string labelFilepath)
{
    std::unique_ptr<imageClassification> imageclassification = std::unique_ptr<imageClassification>(new imageClassification());
    return imageclassification->Classify(instanceName, modelFilepath, img_raw, labelFilepath);
}