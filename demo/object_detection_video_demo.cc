#include "task/vision/object_detection_task.h"
#include "utils/box_utils.h"
#include "utils/check_utils.h"
#include "utils/utils.h"

int DetectVideo(const std::string &modelFilepath,
                const std::string &labelFilepath, const std::string &videoPath,
                const std::string &srcPath) {
  std::unique_ptr<objectDetectionTask> objectdetectiontask =
      std::unique_ptr<objectDetectionTask>(
          new objectDetectionTask(modelFilepath, labelFilepath));
  cv::VideoCapture capture(videoPath);
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
      for (int i = 0; i < resultBoxes.size(); i++) {
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
  std::string modelFilepath, labelFilepath, videoFilepath, dstFilepath;
#ifdef DEBUG
  std::cout << "." << std::endl;
#endif
  if (argc == 5) {
    modelFilepath = argv[1];
    labelFilepath = argv[2];
    videoFilepath = argv[3];
    dstFilepath = argv[4];
    DetectVideo(modelFilepath, labelFilepath, videoFilepath, dstFilepath);
  } else {
    std::cout << "run with " << argv[0]
              << " <modelFilepath>  <labelFilepath> <videoFilepath> "
                 "<dstFilepath> (end with .avi)"
              << std::endl;
  }
  return 0;
}
