#include "task/vision/object_detection_task.h"
#include "utils/utils.h"

int main()
{
    std::string modelFilepath="/home/gexy5/Documents/bianbu-support/data/models/yolov4.onnx";
    std::string imageFilepath="/home/gexy5/Documents/bianbu-support/data/imgs/4.jpg";
    std::string saveImgpath="/home/gexy5/Documents/bianbu-support/data/imgs/person_yolov6_result.jpg";
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    cv::Mat imgRaw = cv::imread(imageFilepath);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Image reading Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
    std::vector<Boxi> resultBoxes;
    objectDetectionTask objectdetectiontask;
    if(objectdetectiontask.Init(modelFilepath))
    {
        resultBoxes = objectdetectiontask.Detect(imgRaw);
    }
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    draw_boxes_inplace(imgRaw , resultBoxes);
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    std::cout << "Box drawing Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count()
              << " ms" << std::endl;
    cv::imwrite(saveImgpath, imgRaw);
    //cv::imshow("detected.jpg",imgRaw);
    //cv::waitKey(0);
    return 0;
}