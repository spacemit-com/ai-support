#include "task/vision/object_detection_task.h"
#include "utils/utils.h"

int DetectVideo(std::string &modelFilepath, std::string &labelFilepath, std::string videoPath)
{
    std::unique_ptr<objectDetectionTask> objectdetectiontask = std::unique_ptr<objectDetectionTask>(new objectDetectionTask(modelFilepath, labelFilepath));
    cv::VideoCapture capture(videoPath);
    cv::Mat frame;
	while (true)
	{
		capture >> frame;
        if(frame.empty())
        {
            break;
        }
        std::vector<Boxi> resultBoxes = objectdetectiontask->Detect(frame).result_bboxes;
        draw_boxes_inplace(frame , resultBoxes);
        cv::imshow("Detection", frame);
    };
    return 0;
}

int main(int argc, char* argv[])
{
    std::string modelFilepath, labelFilepath, videoFilepath;
#ifdef DEBUG
    std::cout<<"."<<std::endl;
#endif
    if(argc == 4)
    {
        modelFilepath = argv[1];
        labelFilepath = argv[2];
        videoFilepath = argv[3];
        DetectVideo(modelFilepath, labelFilepath, videoFilepath);
    }
    else{
        std::cout<<"run with "<<argv[0]<<" <modelFilepath>  <labelFilepath> <videoFilepath>"<<std::endl;
    }
    return 0;
}