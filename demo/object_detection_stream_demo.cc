#include <stdlib.h>
#include <unistd.h>  // for: getopt

#include <algorithm>  // for: swap
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "dataloader.hpp"
#include "object_detection.hpp"
#include "opencv2/opencv.hpp"
#include "task/vision/object_detection_task.h"
#include "utils/cv_helper.hpp"
#ifdef DEBUG
#include "utils/time.h"
#endif

#include "utils/utils.h"

class Detector {
 public:
  explicit Detector(const std::string& config_file_path) {
    config_file_path_ = config_file_path;
  }
  explicit Detector(ObjectDetectionOption& option) { option_ = option; }
  ~Detector() {}
  // 初始化/反初始化
  int init() {
    if (!config_file_path_.empty()) {
      objectdetectiontask_ = std::unique_ptr<ObjectDetectionTask>(
          new ObjectDetectionTask(config_file_path_));
    } else {
      objectdetectiontask_ = std::unique_ptr<ObjectDetectionTask>(
          new ObjectDetectionTask(option_));
    }
    return getInitFlag();
  }

  int uninit() { return 0; }

  // 推理
  int infer(cv::Mat frame) {
    if (frame.empty()) {
      return -1;
    }
    ObjectDetectionResult objs_temp = objectdetectiontask_->Detect(frame);
    objs_mutex_.lock();
    objs_array_.push(objs_temp);  // 直接替换掉当前的 objs_array_
    objs_mutex_.unlock();
    return objs_array_.size();
  }

  // 查询检测结果
  int detected() { return objs_array_.size(); }
  int getInitFlag() { return objectdetectiontask_->getInitFlag(); }
  // 移走检测结果
  ObjectDetectionResult getObject() {
    ObjectDetectionResult objs_moved;
    objs_mutex_.lock();
    objs_moved = objs_array_.back();
    std::queue<struct ObjectDetectionResult> empty;
    std::swap(empty, objs_array_);
    objs_mutex_.unlock();
    return objs_moved;
  }

 private:
  std::mutex objs_mutex_;
  std::queue<struct ObjectDetectionResult> objs_array_;
  std::unique_ptr<ObjectDetectionTask> objectdetectiontask_;
  std::string config_file_path_;
  ObjectDetectionOption option_;
};

// 检测线程
void Detection(DataLoader& dataloader, Detector& detector) {
  setThreadName("DetectionThread");
  if (detector.init() != 0) {
    std::cout << "[ ERROR ] Detector init error" << std::endl;
    dataloader.setDisable();
  }
  cv::Mat frame;
  int total_dur = 0;
  int count = 0;
  auto start = std::chrono::steady_clock::now();
  while (dataloader.ifEnable()) {
    frame = dataloader.peekFrame();  // 取(拷贝)一帧数据
    if (frame.empty()) {
      continue;
    }
    int flag = detector.infer(frame);  // 推理并保存检测结果
    auto end = std::chrono::steady_clock::now();
    auto detection_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    total_dur = detection_duration.count();
    count++;
    if (total_dur > 1000) {
      dataloader.setDetectionFps(count);
      start = std::chrono::steady_clock::now();
      count = 0;
      total_dur = 0;
    }
    if (flag == -1) {
      std::cout << "[ ERROR ] Infer frame failed" << std::endl;
      break;  // 摄像头结束拍摄或者故障
    }
  }
  std::cout << "Detection thread quit" << std::endl;
}

// 预览线程
void Preview(DataLoader& dataloader, Detector& detector) {
  cv::Mat frame;
  ObjectDetectionResult objs;
  auto now = std::chrono::steady_clock::now();
  objs.timestamp = now;
  int count = 0;
  int dur = 0;
  int enable_show = 1;
  const char* showfps = getenv("SUPPORT_SHOWFPS");
  const char* show = getenv("SUPPORT_SHOW");
  if (show && strcmp(show, "-1") == 0) {
    enable_show = -1;
  }
  while (dataloader.ifEnable()) {
    auto start = std::chrono::steady_clock::now();
    frame = dataloader.fetchFrame();  // 取(搬走)一帧数据
    if ((frame).empty()) {
      dataloader.setDisable();
      break;
    }
    if (detector.detected())  // 判断原因: detector.detected 不用锁,
                              // detector.get_object 需要锁;
    {
      // 是否有检测结果
      objs = detector.getObject();  // 取(搬走)检测结果(移动赋值)
      if (objs.result_bboxes.size()) {
        int input_height = dataloader.getResizeHeight();
        int input_width = dataloader.getResizeWidth();
        int img_height = frame.rows;
        int img_width = frame.cols;
        float resize_ratio = std::min(
            static_cast<float>(input_height) / static_cast<float>(img_height),
            static_cast<float>(input_width) / static_cast<float>(img_width));
        float dw = (input_width - resize_ratio * img_width) / 2;
        float dh = (input_height - resize_ratio * img_height) / 2;
        for (size_t i = 0; i < objs.result_bboxes.size(); i++) {
          objs.result_bboxes[i].x1 =
              (objs.result_bboxes[i].x1 - dw) / resize_ratio;
          objs.result_bboxes[i].x2 =
              (objs.result_bboxes[i].x2 - dw) / resize_ratio;
          objs.result_bboxes[i].y1 =
              (objs.result_bboxes[i].y1 - dh) / resize_ratio;
          objs.result_bboxes[i].y2 =
              (objs.result_bboxes[i].y2 - dh) / resize_ratio;
        }
      }
      {
#ifdef DEBUG
        TimeWatcher t("|-- Output result");
#endif
        for (size_t i = 0; i < objs.result_bboxes.size(); i++) {
          std::cout << "bbox[" << std::setw(2) << i << "]"
                    << " "
                    << "x1y1x2y2: "
                    << "(" << std::setw(4) << objs.result_bboxes[i].x1 << ","
                    << std::setw(4) << objs.result_bboxes[i].y1 << ","
                    << std::setw(4) << objs.result_bboxes[i].x2 << ","
                    << std::setw(4) << objs.result_bboxes[i].y2 << ")"
                    << ", "
                    << "score: " << std::fixed << std::setprecision(3)
                    << std::setw(4) << objs.result_bboxes[i].score << ", "
                    << "label_text: " << objs.result_bboxes[i].label_text
                    << std::endl;
        }
      }
    }
    // 调用 detector.detected 和 detector.get_object 期间,
    // 检测结果依然可能被刷新
    now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - objs.timestamp);
    if (duration.count() < 1000) {
      draw_boxes_inplace((frame), objs.result_bboxes);  // 画框
    }
    int preview_fps = dataloader.getPreviewFps();
    int detection_fps = dataloader.getDetectionFps();
    if (showfps != nullptr) {
      cv::putText(frame, "preview fps: " + std::to_string(preview_fps),
                  cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5f,
                  cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
      cv::putText(frame, "detection fps: " + std::to_string(detection_fps),
                  cv::Point(frame.cols - 140, 15), cv::FONT_HERSHEY_SIMPLEX,
                  0.5f, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    if (enable_show != -1) {
      cv::imshow("Detection", (frame));
      cv::waitKey(10);
    }
    auto end = std::chrono::steady_clock::now();
    auto preview_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    count++;
    dur = dur + preview_duration.count();
    if (dur >= 1000) {
      dataloader.setPreviewFps(count);
      dur = 0;
      count = 0;
    }
    if (enable_show != -1) {
      if (cv::getWindowProperty("Detection", cv::WND_PROP_VISIBLE) < 1) {
        dataloader.setDisable();
        break;
      }
    }
  }
  std::cout << "Preview thread quit" << std::endl;
  if (enable_show != -1) {
    cv::destroyAllWindows();
  }
}

int main(int argc, char* argv[]) {
  cvConfig();

  std::string config_file_path, input, input_type;
  ObjectDetectionOption option;
  int resize_height{320}, resize_width{320};
  std::unique_ptr<Detector> detector;
  int o;
  const char* optstring = "w:h:";
  while ((o = getopt(argc, argv, optstring)) != -1) {
    switch (o) {
      case 'w':
        resize_width = atoi(optarg);
        break;
      case 'h':
        resize_height = atoi(optarg);
        break;
      case '?':
        std::cout << "[ ERROR ] Unsupported usage" << std::endl;
        break;
    }
  }
  if (argc - optind == 3) {
    config_file_path = argv[optind];
    input = argv[optind + 1];
    input_type = argv[optind + 2];
    detector = std::unique_ptr<Detector>(new Detector(config_file_path));
  } else if (argc - optind == 4) {
    option.model_path = argv[optind];
    option.label_path = argv[optind + 1];
    input = argv[optind + 2];
    input_type = argv[optind + 3];
    detector = std::unique_ptr<Detector>(new Detector(option));
  } else {
    std::cout << "Please run with " << argv[0]
              << " <model_file_path> <label_file_path> <input> <input_type> "
                 "(video or camera_id) option(-h <resize_height>) option(-w "
                 "<resize_width>) or "
              << argv[0]
              << " <config_file_path> <input> <input_type> (video "
                 "or camera_id) option(-h <resize_height>) option(-w "
                 "<resize_width>)"
              << std::endl;
    return -1;
  }
  SharedDataLoader dataloader{resize_height, resize_width};
  if (dataloader.init(input) != 0) {
    std::cout << "[ ERROR ] Dataloader init error" << std::endl;
    return -1;
  }

  std::thread t(Detection, std::ref(dataloader), std::ref(*detector));
  setThreadName("PreviewThread");
  Preview(dataloader, *detector);
  t.join();
  return 0;
}
