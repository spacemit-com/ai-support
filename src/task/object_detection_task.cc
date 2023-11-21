#include "task/vision/object_detection_task.h"
#include "src/task/vision/object_detection.h"

class objectDetectionTask::impl {
    public:
    std::unique_ptr<ObjectDetection> objectdetection_;
};

objectDetectionTask::objectDetectionTask() : pimpl_(std::make_unique<impl>()) {}

int objectDetectionTask::Init(std::string &modelFilepath)
{
    pimpl_->objectdetection_ = std::unique_ptr<ObjectDetection>(new ObjectDetection());
    int flag = pimpl_->objectdetection_->Init(modelFilepath);
    return flag;
}

std::vector<Boxi> objectDetectionTask::Detect_Yolov6(cv::Mat &raw_img)
{
    return pimpl_->objectdetection_->Detect_Yolov6(raw_img);
}

std::vector<Boxi> objectDetectionTask::Detect(cv::Mat &raw_img)
{
    return pimpl_->objectdetection_->Detect(raw_img);
}
