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
        {
#ifdef DEBUG
          TimeWatcher t("|-- Output result");
#endif
          for(int i=0;i<resultBoxes.size();i++)
          {
            std::cout<<"bbox["<<std::setw(2)<<i<<"]"<<" "\
            <<"x1y1x2y2: "\
            <<"("<<std::setw(4)<<resultBoxes[i].x1<<","\
            <<std::setw(4)<<resultBoxes[i].y1<<","\
            <<std::setw(4)<<resultBoxes[i].x2<<","\
            <<std::setw(4)<<resultBoxes[i].y2<<")"<<", "\
            <<"score: "<<std::fixed<<std::setprecision(3)<<std::setw(4)<<resultBoxes[i].score<<", "\
            <<"label_text: "<<std::setw(4)<<resultBoxes[i].label_text<<std::endl;
          }
        }
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