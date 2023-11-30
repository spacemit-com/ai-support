#include "task/vision/object_detection_task.h"
#include "utils/utils.h"

#include <thread>
#include <mutex>

cv::Mat frame;
std::mutex data_mutex;

void DetectFrame()
{
    std::string modelFilepath="/home/gexy5/Documents/bianbu-support/data/models/nanodet-plus-m_320.onnx";
    objectDetectionTask objectdetectiontask;
    int flag = objectdetectiontask.Init(modelFilepath);
    cv::Mat frame_detected;
    while(true)
    {
        data_mutex.lock();
        if(frame.empty()) continue;
        frame_detected = frame;
        data_mutex.unlock();
        std::vector<Boxi> resultBoxes = objectdetectiontask.Detect(frame_detected);
        data_mutex.lock();
        frame_detected = frame;
        data_mutex.unlock();
        draw_boxes_inplace(frame_detected , resultBoxes);
        imshow("Detection", frame_detected);
        cv::waitKey(30);
    }
}

int Stream(int stream)
{
    cv::VideoCapture capture(stream);
    //int width = 1280;
    //int height = 720;
    //capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    //capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	while (true)
	{
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        data_mutex.lock();
		capture >> frame;
        data_mutex.unlock();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Image reading Latency: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                << " ms" << std::endl;
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
/*
int main()
{
    std::string modelFilepath="/home/gexy5/Documents/bianbu-support/data/models/nanodet-plus-m_320.v13.sim.qop.pc.onnx";
    std::string videoPath="/home/gexy5/Documents/bianbu-support/data/imgs/test.mp4";
    int flag = DetectVideo(modelFilepath, videoPath);
    return 0;
}
*/
int main()
{
    std::thread t1(Stream,0);
    std::thread t2(DetectFrame);
    t1.join();
    t2.join();
    return 0;
}


