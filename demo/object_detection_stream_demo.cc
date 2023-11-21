#include "task/vision/object_detection_task.h"
#include "utils/utils.h"

int DetectStream(std::string &modelFilepath, int stream)
{
    objectDetectionTask objectdetectiontask;
    int flag = objectdetectiontask.Init(modelFilepath);
    cv::VideoCapture capture(stream);
	while (true)
	{
		cv::Mat frame;
		capture >> frame;
        std::vector<Boxi> resultBoxes = objectdetectiontask.Detect(frame);
        draw_boxes_inplace(frame , resultBoxes);
		imshow("Detection", frame);
		cv::waitKey(30);
    }
    return 0;
}

int DetectVideo(std::string &modelFilepath, std::string videoPath)
{
    objectDetectionTask objectdetectiontask;
    int flag = objectdetectiontask.Init(modelFilepath);
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
        std::vector<Boxi> resultBoxes = objectdetectiontask.Detect(frame);
        draw_boxes_inplace(frame , resultBoxes);
        //out.write(frame);
		//imshow("Detection", frame);
		//cv::waitKey(30);
    };
    //out.release();
    return 0;
}

int main()
{
    std::string modelFilepath="/home/gexy5/Documents/bianbu-support/data/models/yolov4.onnx";
    std::string videoPath="/home/gexy5/Documents/bianbu-support/data/imgs/test.mp4";
    int flag = DetectVideo(modelFilepath, videoPath);
    return 0;
}
