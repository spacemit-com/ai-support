#include "task/vision/object_detection_task.h"
#include "utils/utils.h"
#include "utils/time.h"

int main(int argc, char* argv[])
{
    std::vector<Boxi> resultBoxes;
    std::string filePath;
    std::string modelFilepath, imageFilepath, saveImgpath, labelFilepath, configFilepath;
    cv::Mat imgRaw;
#ifdef DEBUG
    std::cout<<"."<<std::endl;
#endif
    if(argc == 4)
    {
        filePath = argv[1];
        imageFilepath = argv[2]; 
        saveImgpath =  argv[3];
        {
#ifdef DEBUG
          TimeWatcher t("|-- Load input data");
#endif
          imgRaw = cv::imread(imageFilepath);
        }
        std::unique_ptr<objectDetectionTask> objectdetectiontask = std::unique_ptr<objectDetectionTask>(new objectDetectionTask(filePath));
        resultBoxes = objectdetectiontask->Detect(imgRaw).result_bboxes;
        {
#ifdef DEBUG
          TimeWatcher t("|-- Box drawing");
#endif
          draw_boxes_inplace(imgRaw , resultBoxes);
        }

        cv::imwrite(saveImgpath, imgRaw);
        //cv::imshow("detected.jpg",imgRaw);
        //cv::waitKey(0);
    }
    else if(argc == 5)
    {
        filePath = argv[1];
        imageFilepath = argv[2]; 
        saveImgpath =  argv[3];
        labelFilepath = argv[4];
        {
#ifdef DEBUG
          TimeWatcher t("|-- Load input data");
#endif
          imgRaw = cv::imread(imageFilepath);
        }
        std::unique_ptr<objectDetectionTask> objectdetectiontask = std::unique_ptr<objectDetectionTask>(new objectDetectionTask(filePath, labelFilepath));
        resultBoxes = objectdetectiontask->Detect(imgRaw).result_bboxes;
        {
#ifdef DEBUG
          TimeWatcher t("|-- Box drawing");
#endif
          draw_boxes_inplace(imgRaw , resultBoxes);
        }

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