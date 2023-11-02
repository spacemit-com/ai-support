#include "cv2_utils.h"
#include "task_api_factory.h"
#include "object_detection.h"

using namespace std;
int main()
{
    std::string instanceName="object-detection-inference";
    std::string modelFilepath="/home/geduer/Documents/BianbuAI/data/models/yolov4.onnx";
    std::string imageFilepath="/home/geduer/Documents/BianbuAI/data/imgs/4.jpg";
    std::string save_img_path="/home/geduer/Documents/BianbuAI/data/imgs/4_result.jpg";
    cv::Mat img_raw = cv::imread(imageFilepath);
    std::vector<Boxf> detected_boxes;
    std::unique_ptr<ObjectDetection> objectdetection = TaskAPIFactory::Create<ObjectDetection>();
    detected_boxes = objectdetection->Detect(instanceName, modelFilepath, img_raw);
    draw_boxes_inplace(img_raw , detected_boxes);
    cv::imwrite(save_img_path, img_raw);
    cv::imshow("detected.jpg",img_raw);
    cv::waitKey(0);
    return 0;
}