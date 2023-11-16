#include "task/vision/object_detection_stream_task.h"
#include "src/task/vision/object_detection.h"  

int objectDetectionStreamTask::DetectVideo(std::string &instanceName, 
                                     std::string &modelFilepath, 
                                     int video)
{
    std::unique_ptr<ObjectDetection> objectdetection = std::unique_ptr<ObjectDetection>(new ObjectDetection());
    int flag = objectdetection->Init(instanceName, modelFilepath);
    cv::VideoCapture capture(video);
	while (true)
	{
		cv::Mat frame;
		capture >> frame;
        std::vector<Boxi> resultBoxes = objectdetection->DetectFrame(instanceName, modelFilepath, frame);
        draw_boxes_inplace(frame , resultBoxes);
		imshow("Detection", frame);
		cv::waitKey(30);
    }
    return 0;
}