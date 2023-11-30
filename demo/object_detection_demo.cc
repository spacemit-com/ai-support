#include "task/vision/object_detection_task.h"
#include "utils/utils.h"

int main()
{
    std::string modelFilepath="/home/gexy5/Documents/bianbu-support/data/models/nanodet-plus-m_320.v13.sim.qop.pc.onnx";
    std::string imageFilepath="/home/gexy5/Documents/bianbu-support/data/imgs/3.jpg";
    std::string saveImgpath="/home/gexy5/Documents/bianbu-support/data/imgs/person_nanodet_result.jpg";
#ifdef DEBUG
    std::cout<<"."<<std::endl;
#endif

#ifdef DEBUG
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    cv::Mat imgRaw = cv::imread(imageFilepath);
#ifdef DEBUG
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "|-- Image reading Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " ms" << std::endl;
#endif
    std::vector<Boxi> resultBoxes;
    objectDetectionTask objectdetectiontask;
    if(objectdetectiontask.Init(modelFilepath))
    {
        resultBoxes = objectdetectiontask.Detect_NanoDet(imgRaw);
    }
#ifdef DEBUG
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
#endif
    draw_boxes_inplace(imgRaw , resultBoxes);


#ifdef DEBUG
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    std::cout << "|-- Box drawing Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count()
              << " ms" << std::endl;
#endif
    cv::imwrite(saveImgpath, imgRaw);
    //cv::imshow("detected.jpg",imgRaw);
    //cv::waitKey(0);
    return 0;
}