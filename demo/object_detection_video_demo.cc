#include <iostream>

#include "object_detection.hpp"
#include "task/vision/object_detection_task.h"
#include "utils/json_helper.hpp"
#ifdef DEBUG
#include "utils/time.h"
#endif

static void usage(const char *exe) {
  std::cout << "Usage: \n"
            << exe
            << " <model_path> <label_path> <video_path> <save_path>(*.avi)\n"
            << exe << " <config_path> <video_path> <save_path>(*.avi)\n";
}

int main(int argc, char *argv[]) {
  if (argc != 4 && argc != 5) {
    usage(argv[0]);
    return -1;
  }

  ObjectDetectionOption option;
  std::string video_file_path, dst_file_path;
  if (argc == 4) {
    if (configToOption(argv[1], option) != 0) {
      return -1;
    }
    video_file_path = argv[2];
    dst_file_path = argv[3];
  } else if (argc == 5) {
    option.model_path = argv[1];
    option.label_path = argv[2];
    video_file_path = argv[3];
    dst_file_path = argv[4];
  }

  std::unique_ptr<ObjectDetectionTask> objectdetectiontask =
      std::unique_ptr<ObjectDetectionTask>(new ObjectDetectionTask(option));
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
  // int delay = 1000 / rate;
  int fps = rate;
  int frameWidth = frame.rows;
  int frameHeight = frame.cols;
  cv::VideoWriter writer(dst_file_path,
                         cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), fps,
                         cv::Size(frameHeight, frameWidth), 1);

#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
  while (true) {
    capture >> frame;
    if (frame.empty()) {
      break;
    }

    std::vector<Boxi> bboxes = objectdetectiontask->Detect(frame).result_bboxes;
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

    draw_boxes_inplace(frame, bboxes);
    writer.write(frame);
    // cv::waitKey(delay);
    // cv::imshow("Detection", frame);
  }

  capture.release();
  writer.release();
  return 0;
}
