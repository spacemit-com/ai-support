#include "task/vision/object_detection_task.h"
#include "utils/utils.h"

#include <thread>
#include <mutex>

int DetectVideo(std::string &modelFilepath, std::string &labelFilepath, std::string videoPath)
{
    std::string filePath = "/home/gexy5/Documents/bianbu-support/data/config/nanodet.json";
    std::unique_ptr<objectDetectionTask> objectdetectiontask = std::unique_ptr<objectDetectionTask>(new objectDetectionTask(filePath, labelFilepath));
    std::string save_name = "/home/gexy5/Documents/bianbu-support/data/imgs/saved.avi";
    cv::VideoCapture capture(videoPath);
    int width = 1920;
    int height = 1080;
    int fps = 30;
    //auto out = cv::VideoWriter(save_name, cv::VideoWriter::fourcc('M','j','P','G'), fps);
	while (true)
	{
		cv::Mat frame;
		capture >> frame;
        std::vector<Boxi> resultBoxes = objectdetectiontask->Detect(frame).result_bboxes;
        draw_boxes_inplace(frame , resultBoxes);
        //out.write(frame);
		//imshow("Detection", frame);
		//cv::waitKey(30);
    };
    //out.release();
    return 0;
}


