#include <iostream>

#include "object_detection.hpp"
#include "task/vision/object_detection_task.h"
#ifdef DEBUG
#include "utils/time.h"
#endif

#include "utils/utils.h"
int DetectVideo(const std::string &filepath, const std::string &videoPath,
                const std::string &srcPath) {
  std::unique_ptr<objectDetectionTask> objectdetectiontask =
      std::unique_ptr<objectDetectionTask>(new objectDetectionTask(filepath));
  if (objectdetectiontask->getInitFlag() != 0) {
    return -1;
  }
  cv::VideoCapture capture(videoPath);
  if (!capture.isOpened()) {
    std::cout << "Open video capture failed" << std::endl;
    return -1;
  }
  cv::Mat frame;
  if (!capture.read(frame)) {
    std::cout << "Read frame failed" << std::endl;
    return -1;
  }
  double rate = capture.get(cv::CAP_PROP_FPS);
  int delay = 1000 / rate;
  int fps = rate;
  int frameWidth = frame.rows;
  int frameHeight = frame.cols;
  cv::VideoWriter writer(srcPath, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'),
                         fps, cv::Size(frameHeight, frameWidth), 1);
  while (true) {
    capture >> frame;
    if (frame.empty()) {
      break;
    }
    std::vector<Boxi> resultBoxes =
        objectdetectiontask->Detect(frame).result_bboxes;
    {
#ifdef DEBUG
      TimeWatcher t("|-- Output result");
#endif
      for (int i = 0; i < static_cast<int>(resultBoxes.size()); i++) {
        if (resultBoxes[i].flag) {
          std::cout << "bbox[" << std::setw(2) << i << "]"
                    << " "
                    << "x1y1x2y2: "
                    << "(" << std::setw(4) << resultBoxes[i].x1 << ","
                    << std::setw(4) << resultBoxes[i].y1 << "," << std::setw(4)
                    << resultBoxes[i].x2 << "," << std::setw(4)
                    << resultBoxes[i].y2 << ")"
                    << ", "
                    << "score: " << std::fixed << std::setprecision(3)
                    << std::setw(4) << resultBoxes[i].score << ", "
                    << "label_text: " << resultBoxes[i].label_text << std::endl;
        }
      }
    }
    draw_boxes_inplace(frame, resultBoxes);
    writer.write(frame);
    cv::waitKey(
        delay);  // 因为图像处理需要消耗一定时间,所以图片展示速度比保存视频要慢
                 // cv::imshow("Detection", frame);
  }
  capture.release();
  writer.release();
  return 0;
}

int main(int argc, char *argv[]) {
  std::string filepath, videoFilepath, dstFilepath;
#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
  if (argc == 4) {
    filepath = argv[1];
    videoFilepath = argv[2];
    dstFilepath = argv[3];
    int flag = DetectVideo(filepath, videoFilepath, dstFilepath);
    if (flag != 0) {
      std::cout << "[Error] Detect fail" << std::endl;
    }
  } else {
    std::cout << "run with " << argv[0]
              << " <configFilepath> <videoFilepath> "
                 "<dstFilepath> (end with .avi)"
              << std::endl;
    return -1;
  }
  return 0;
}
