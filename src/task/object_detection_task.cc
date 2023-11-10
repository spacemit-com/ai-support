#include "task/vision/object_detection_task.h"
#include "src/task/vision/object_detection.h"

std::vector<Boxi> objectDetectionTask::Detect(std::string &instanceName, 
                                            std::string &modelFilepath, 
                                            cv::Mat &raw_img)
{
    std::unique_ptr<ObjectDetection> objectdetection = std::unique_ptr<ObjectDetection>(new ObjectDetection());
    return objectdetection->Detect(instanceName, modelFilepath, raw_img);
}
