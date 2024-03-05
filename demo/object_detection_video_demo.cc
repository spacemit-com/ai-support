#include <iostream>

#include "object_detection.hpp"
#include "task/vision/object_detection_task.h"
#ifdef DEBUG
#include "utils/time.h"
#endif

#include "utils/utils.h"

int main(int argc, char *argv[]) {
  std::unique_ptr<ObjectDetectionTask> objectdetectiontask;
  std::string config_file_path, video_file_path, dst_file_path;
  ObjectDetectionOption option;
#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
  if (argc == 4) {
    config_file_path = argv[1];
    video_file_path = argv[2];
    dst_file_path = argv[3];
    objectdetectiontask = std::unique_ptr<ObjectDetectionTask>(
        new ObjectDetectionTask(config_file_path));
  }
  if (argc == 5) {
    option.model_path = argv[1];
    option.label_path = argv[2];
    video_file_path = argv[3];
    dst_file_path = argv[4];
    objectdetectiontask =
        std::unique_ptr<ObjectDetectionTask>(new ObjectDetectionTask(option));
  } else {
    std::cout << "Please run with " << argv[0]
              << " <model_file_path> <label_file_path> <video_file_path> "
                 "<dst_file_path> (end with .avi) or "
              << argv[0]
              << " <config_file_path> <video_file_path> "
                 "<dst_file_path> (end with .avi)"
              << std::endl;
    return -1;
  }
  if (objectdetectiontask->getInitFlag() != 0) {
    return -1;
  }
  cv::VideoCapture capture(video_file_path);
  if (!capture.isOpened()) {
    std::cout << "[ ERROR ] Open video capture failed" << std::endl;
    return -1;
  }
  cv::Mat frame;
  if (!capture.read(frame)) {
    std::cout << "[ ERROR ] Read frame failed" << std::endl;
    return -1;
  }
  double rate = capture.get(cv::CAP_PROP_FPS);
  int delay = 1000 / rate;
  int fps = rate;
  int frameWidth = frame.rows;
  int frameHeight = frame.cols;
  cv::VideoWriter writer(dst_file_path,
                         cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), fps,
                         cv::Size(frameHeight, frameWidth), 1);
  while (true) {
    capture >> frame;
    if (frame.empty()) {
      break;
    }
    std::vector<Boxi> bboxes = objectdetectiontask->Detect(frame).result_bboxes;
    {
#ifdef DEBUG
      TimeWatcher t("|-- Output result");
#endif
      for (size_t i = 0; i < bboxes.size(); i++) {
        std::cout << "bbox[" << std::setw(2) << i << "]"
                  << " "
                  << "x1y1x2y2: "
                  << "(" << std::setw(4) << bboxes[i].x1 << "," << std::setw(4)
                  << bboxes[i].y1 << "," << std::setw(4) << bboxes[i].x2 << ","
                  << std::setw(4) << bboxes[i].y2 << ")"
                  << ", "
                  << "score: " << std::fixed << std::setprecision(3)
                  << std::setw(4) << bboxes[i].score << ", "
                  << "label_text: " << bboxes[i].label_text << std::endl;
      }
    }
    draw_boxes_inplace(frame, bboxes);
    writer.write(frame);
    cv::waitKey(
        delay);  // 因为图像处理需要消耗一定时间,所以图片展示速度比保存视频要慢
                 // cv::imshow("Detection", frame);
  }
  capture.release();
  writer.release();
  return 0;
}
