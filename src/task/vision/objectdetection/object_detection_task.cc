#include "task/vision/object_detection_task.h"
#include "src/task/vision/objectdetection/object_detection.h"

class objectDetectionTask::impl {
    public:
    std::unique_ptr<ObjectDetection> objectdetection_;
};

objectDetectionTask::objectDetectionTask(const std::string &filePath, const std::string& labelFilepath) : pimpl_(std::make_unique<impl>()) 
{
    if(filePath.length()>4)
    {
        pimpl_->objectdetection_ = std::unique_ptr<ObjectDetection>(new ObjectDetection());
        std::string suffixStr = filePath.substr(filePath.length()-4,4);
        if(strcmp(suffixStr.c_str(), "onnx") == 0)
        {
            int flag = pimpl_->objectdetection_->InitFromCommand(filePath, labelFilepath);
        }
        else if(strcmp(suffixStr.c_str(), "json") == 0)
        {
            int flag = pimpl_->objectdetection_->InitFromConfig(filePath);
        }
    }
    else
    {
        std::cout<<"Unsupport filepath"<<std::endl;
    }
}

objectDetectionTask::objectDetectionTask(const std::string &filePath) : pimpl_(std::make_unique<impl>()) 
{
    if(filePath.length()>4)
    {
        pimpl_->objectdetection_ = std::unique_ptr<ObjectDetection>(new ObjectDetection());
        std::string suffixStr = filePath.substr(filePath.length()-4,4);
        if(strcmp(suffixStr.c_str(), "json") == 0)
        {
        int flag = pimpl_->objectdetection_->InitFromConfig(filePath);
        }
        else
        {
            std::cout<<"Unsupport file"<<std::endl;
        }
    }
    else
    {
        std::cout<<"Unsupport filepath"<<std::endl;
    }
}

ObjectDetectionResult objectDetectionTask::Detect(const cv::Mat &raw_img)
{
    return pimpl_->objectdetection_->Detect(raw_img);
}