#include "task/vision/object_detection_task.h"
#include "src/task/vision/objectdetection/object_detection.h"

class objectDetectionTask::impl {
    public:
    std::unique_ptr<ObjectDetection> objectdetection_;
};

objectDetectionTask::objectDetectionTask(std::string &filePath, std::string& labelFilepath) : pimpl_(std::make_unique<impl>()) 
{
    pimpl_->objectdetection_ = std::unique_ptr<ObjectDetection>(new ObjectDetection());
    std::string suffixStr = filePath.substr(filePath.find_last_of('.') + 1);
    if(strcmp(suffixStr.c_str(), "onnx") == 0)
    {
        int flag = pimpl_->objectdetection_->Initfromcommand(filePath, labelFilepath);
    }
    else if(strcmp(suffixStr.c_str(), "json") == 0)
    {
        int flag = pimpl_->objectdetection_->Initfromconfig(filePath);
    }
}

ObjectDetectionResult objectDetectionTask::Detect_Yolov6(const cv::Mat &raw_img)
{
    return pimpl_->objectdetection_->Detect_Yolov6(raw_img);
}

ObjectDetectionResult objectDetectionTask::Detect(const cv::Mat &raw_img)
{
    return pimpl_->objectdetection_->Detect(raw_img);
}

ObjectDetectionResult objectDetectionTask::Detect_NanoDet(const cv::Mat &raw_img)
{
    return pimpl_->objectdetection_->Detect_NanoDet(raw_img);
}