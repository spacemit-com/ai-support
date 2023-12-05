#include "task/vision/object_detection_task.h"
#include "src/task/vision/objectdetection/object_detection.h"

class objectDetectionTask::impl {
    public:
    std::unique_ptr<ObjectDetection> objectdetection_;
};

objectDetectionTask::objectDetectionTask() : pimpl_(std::make_unique<impl>()) {}

int objectDetectionTask::Init(std::string &modelFilepath, std::string &labelFilepath)
{
    pimpl_->objectdetection_ = std::unique_ptr<ObjectDetection>(new ObjectDetection());
    int flag = pimpl_->objectdetection_->Init(modelFilepath, labelFilepath);
    return flag;
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