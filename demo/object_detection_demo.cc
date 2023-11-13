#include "task/vision/object_detection_task.h"
#include "utils/utils.h"

int main()
{
    std::string instanceName="object-detection-inference";
    std::string modelFilepath="/home/gexy5/Documents/bianbu-support/data/models/yolov4.onnx";
    std::string imageFilepath="/home/gexy5/Documents/bianbu-support/data/imgs/4.jpg";
    std::string saveImgpath="/home/gexy5/Documents/bianbu-support/data/imgs/4_result.jpg";
    cv::Mat imgRaw = cv::imread(imageFilepath);
    std::vector<Boxi> resultBoxes;
    objectDetectionTask objectdetectiontask;
    resultBoxes = objectdetectiontask.Detect(instanceName, modelFilepath, imgRaw);
    draw_boxes_inplace(imgRaw , resultBoxes);
    cv::imwrite(saveImgpath, imgRaw);
    cv::imshow("detected.jpg",imgRaw);
    cv::waitKey(0);
    return 0;
}