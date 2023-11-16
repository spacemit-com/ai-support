#include "task/vision/object_detection_stream_task.h"

int main()
{
    std::string instanceName="object-detection-inference";
    std::string modelFilepath="/home/gexy5/Documents/bianbu-support/data/models/yolov4.onnx";
    objectDetectionStreamTask objectdetectionstreamtask;
    int flag = objectdetectionstreamtask.DetectVideo(instanceName, modelFilepath, 0);
    return 0;
}