#include "task/vision/object_detection_task.h"
#include "utils/cv2_utils.h"

int main()
{
    std::string instanceName="object-detection-inference";
    std::string modelFilepath="/home/gexy5/Documents/bianbu-support/data/models/yolov4.onnx";
    std::string imageFilepath="/home/gexy5/Documents/bianbu-support/data/imgs/4.jpg";
    std::string save_img_path="/home/gexy5/Documents/bianbu-support/data/imgs/4_result.jpg";
    cv::Mat img_raw = cv::imread(imageFilepath);
    std::vector<Boxi> result_boxes;
    objectDetectionTask objectdetectiontask;
    result_boxes = objectdetectiontask.Detect(instanceName, modelFilepath, img_raw);
    draw_boxes_inplace(img_raw , result_boxes);
    cv::imwrite(save_img_path, img_raw);
    cv::imshow("detected.jpg",img_raw);
    cv::waitKey(0);
    return 0;
}