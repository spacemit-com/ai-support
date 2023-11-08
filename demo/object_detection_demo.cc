#include "utils/cv2_utils.h"
#include "task/task_api_factory.h"
#include "task/vision/object_detection.h"

using namespace std;
int main()
{
    std::string instanceName="object-detection-inference";
    std::string modelFilepath="${HOME_DIR}/support/data/models/yolov4.onnx";
    std::string imageFilepath="${HOME_DIR}/support/data/imgs/4.jpg";
    std::string save_img_path="${HOME_DIR}/support/data/imgs/4_result.jpg";
    cv::Mat img_raw = cv::imread(imageFilepath);
    std::vector<Boxi> result_boxes;
    std::unique_ptr<ObjectDetection> objectdetection = TaskAPIFactory::Create<ObjectDetection>();
    result_boxes = objectdetection->Detect(instanceName, modelFilepath, img_raw);
    draw_boxes_inplace(img_raw , result_boxes);
    cv::imwrite(save_img_path, img_raw);
    cv::imshow("detected.jpg",img_raw);
    cv::waitKey(0);
    return 0;
}