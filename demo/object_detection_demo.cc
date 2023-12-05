#include "task/vision/object_detection_task.h"
#include "utils/utils.h"

int main(int argc, char* argv[])
{
    std::vector<Boxi> resultBoxes;
    std::string filePath;
    std::string modelFilepath, imageFilepath, saveImgpath, labelFilepath, configFilepath;
#ifdef DEBUG
    std::cout<<"."<<std::endl;
#endif
    if(argc == 5)
    {
        filePath = argv[1];
        //"/home/gexy5/Documents/bianbu-support/data/config/nanodet.json";
        imageFilepath = argv[2]; 
        //"/home/gexy5/Documents/bianbu-support/data/imgs/3.jpg";
        saveImgpath =  argv[3];
        //"/home/gexy5/Documents/bianbu-support/data/imgs/3_result.jpg";
        labelFilepath = argv[4];
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
        std::unique_ptr<objectDetectionTask> objectdetectiontask = std::unique_ptr<objectDetectionTask>(new objectDetectionTask(filePath, labelFilepath));
        resultBoxes = objectdetectiontask->Detect_NanoDet(imgRaw).result_bboxes;
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
    }
    else
    {
        std::cout<<"run with ./detection_demo <modelFilepath> <imageFilepath> <saveImgpath> <labelFilepath> or ./detection_demo <configFilepath> <imageFilepath> <saveImgpath>"<<std::endl;
    }
    return 0;
}