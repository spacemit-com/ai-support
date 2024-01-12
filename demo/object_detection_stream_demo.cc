#include <stdlib.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "opencv2/opencv.hpp"
#include "task/vision/object_detection_task.h"
#include "utils/box_utils.h"
#include "utils/check_utils.h"
#ifdef DEBUG
#include "utils/time.h"
#endif

#include "utils/utils.h"

class Detector {
 public:
  Detector(const std::string& filePath, const std::string& labelFilepath,
           const bool disableSpacemitEp, const int intraThreadsNum,
           const float scoreThreshold, const float nmsThreshold) {
    filePath_ = filePath;
    labelFilepath_ = labelFilepath;
    disableSpacemitEp_ = disableSpacemitEp;
    intraThreadsNum_ = intraThreadsNum;
    scoreThreshold_ = scoreThreshold;
    nmsThreshold_ = nmsThreshold;
  }
  ~Detector() {}
  // 初始化/反初始化
  int init() {
    objectdetectiontask_ =
        std::unique_ptr<objectDetectionTask>(new objectDetectionTask(
            filePath_, labelFilepath_, disableSpacemitEp_, intraThreadsNum_,
            scoreThreshold_, nmsThreshold_));
    return 0;
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

  // 移走检测结果
  ObjectDetectionResult get_object() {
    ObjectDetectionResult objs_moved;
    objs_mutex_.lock();
    objs_moved = objs_array_.front();
    objs_array_.pop();  // 移走后 objs_array_ 为空数组
    objs_mutex_.unlock();
    return objs_moved;
  }

 private:
  std::mutex objs_mutex_;
  std::queue<ObjectDetectionResult> objs_array_;
  std::unique_ptr<objectDetectionTask> objectdetectiontask_;
  std::string filePath_;
  std::string labelFilepath_;
  bool disableSpacemitEp_;
  int intraThreadsNum_;
  float scoreThreshold_;
  float nmsThreshold_;
};

class DataLoader {
 public:
  DataLoader() { enable = true; }
  ~DataLoader() {}
  bool ifenable() { return enable; }
  virtual cv::Mat fetch_frame() = 0;
  virtual cv::Mat peek_frame() = 0;
  virtual std::shared_ptr<cv::Mat> fetch_frame_v2() = 0;
  virtual std::shared_ptr<cv::Mat> clone_frame() = 0;

 private:
  bool enable;
};

// 独占式
class ExclusiveDataLoader : public DataLoader {
 public:
  ExclusiveDataLoader() {}
  ~ExclusiveDataLoader() {}
  int init(const std::string& path) {
    capture_.open(path);
    if (capture_.isOpened()) {
      return 0;
    } else {
      std::cout << "Open video capture failed" << std::endl;
      return -1;
    }
  }
  int init(const int cameraId) {
    capture_.open(cameraId);
    if (capture_.isOpened()) {
      return 0;
    } else {
      std::cout << "Open camera capture failed" << std::endl;
      return -1;
    }
  }
  cv::Mat fetch_frame() {
    cv::Mat frame;
    capture_.read(frame);
    return frame;
  }
  cv::Mat peek_frame() { return fetch_frame(); }

 private:
  cv::VideoCapture capture_;
};

// 共享式
class SharedDataLoader : public DataLoader {
 public:
  SharedDataLoader() {}
  ~SharedDataLoader() {}
  int init(const std::string& path) {
    capture_.open(path);
    if (capture_.isOpened()) {
      int width = 1280;
      int height = 720;
      capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
      capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
      return 0;
    } else {
      std::cout << "Open video capture failed" << std::endl;
      return -1;
    }
  }
  int init(int cameraId) {
    capture_.open(cameraId);
    if (capture_.isOpened()) {
      int width = 640;
      int height = 480;
      capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
      capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
      return 0;
    } else {
      std::cout << "Open camera capture failed" << std::endl;
      return -1;
    }
  }

  std::shared_ptr<cv::Mat> fetch_frame_v2() {
    cv::Mat _frame, temp;
    capture_.read(_frame);
    resize_unscale(_frame, temp, 320, 320);
    frame_mutex_.lock();
    frame = std::make_shared<cv::Mat>(temp);
    frame_mutex_.unlock();
    return frame;
  }
  std::shared_ptr<cv::Mat> clone_frame() {
    frame_mutex_.lock();
    std::shared_ptr<cv::Mat> ptr = std::make_shared<cv::Mat>(frame->clone());
    frame_mutex_.unlock();
    return ptr;
  }
  cv::Mat fetch_frame() {
    cv::Mat frame, temp;
    capture_.read(frame);
    resize_unscale(frame, temp, 320, 320);
    if (!frame.empty()) {
      frame_mutex_.lock();
      frame_ = temp.clone();
      frame_mutex_.unlock();
    }
    return frame;
  }
  cv::Mat peek_frame() {
    cv::Mat frame;
    frame_mutex_.lock();
    if (!frame_.empty()) {
      frame = frame_.clone();  // 深拷贝
    }
    frame_mutex_.unlock();
    return frame;
  }

 private:
  std::shared_ptr<cv::Mat> frame;
  cv::Mat frame_;
  std::mutex frame_mutex_;
  cv::VideoCapture capture_;
  std::queue<cv::Mat> frame_queue_;
};

// 检测线程
void Detection(DataLoader& dataloader, Detector& detector) {
  if (detector.init() != 0) {
    std::cout << "[ERROR] detector init error" << std::endl;
    return;
  }
  cv::Mat frame;
  while (dataloader.ifenable()) {
    frame = dataloader.peek_frame();  // 取(拷贝)一帧数据
    // std::shared_ptr<cv::Mat> frame = dataloader.clone_frame();
    if ((frame).empty()) {
      continue;
    }
    int flag = detector.infer(frame);  // 推理并保存检测结果
    if (flag == -1) {
      std::cout << "detection thread quit" << std::endl;
      break;  // 摄像头结束拍摄或者故障
    }
  }
}

// 预览线程
void Preview(DataLoader& dataloader, Detector& detector) {
  cv::Mat frame;
  ObjectDetectionResult objs;
  auto now = std::chrono::high_resolution_clock::now();
  objs.timestamp = now;
  while (dataloader.ifenable()) {
    frame = dataloader.fetch_frame();  // 取(搬走)一帧数据
    // std::shared_ptr<cv::Mat> frame = dataloader.fetch_frame_v2();
    if ((frame).empty()) {
      break;
    }
    if (detector.detected())  // 判断原因: detector.detected 不用锁,
                              // detector.get_object 需要锁;
    {
      // 是否有检测结果
      objs = detector.get_object();  // 取(搬走)检测结果(移动赋值)
      if (objs.result_bboxes.size()) {
        int input_height = 320;
        int input_width = 320;
        int img_height = frame.rows;
        int img_width = frame.cols;
        float resize_ratio = std::min(
            static_cast<float>(input_height) / static_cast<float>(img_height),
            static_cast<float>(input_width) / static_cast<float>(img_width));
        float dw = (input_width - resize_ratio * img_width) / 2;
        float dh = (input_height - resize_ratio * img_height) / 2;
        for (int i = 0; i < objs.result_bboxes.size(); i++) {
          objs.result_bboxes[i].x1 =
              (objs.result_bboxes[i].x1 - dw) / resize_ratio;
          objs.result_bboxes[i].x2 =
              (objs.result_bboxes[i].x2 - dw) / resize_ratio;
          objs.result_bboxes[i].y1 =
              (objs.result_bboxes[i].y1 - dh) / resize_ratio;
          objs.result_bboxes[i].y2 =
              (objs.result_bboxes[i].y2 - dh) / resize_ratio;
        }
        {
#ifdef DEBUG
          TimeWatcher t("|-- Output result");
#endif
          for (int i = 0; i < objs.result_bboxes.size(); i++) {
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
    }  // 调用 detector.detected 和 detector.get_object 期间,
       // 检测结果依然可能被刷新
    now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        now - objs.timestamp);
    if (duration.count() < 1000000) {
      draw_boxes_inplace((frame), objs.result_bboxes);  // 画框
    }
    cv::imshow("Detection", (frame));
    cv::waitKey(25);
  }
}

int main(int argc, char* argv[]) {
  std::string filePath, labelFilepath, input, inputType;
  bool disable_spacemit_ep{false};
  float score_threshold{0.4}, nms_threshold{0.5};
  int intra_threads_num{1};
  if (argc == 5) {
    filePath = argv[1];
    labelFilepath = argv[2];
    input = argv[3];
    inputType = argv[4];
  } else if (argc > 5) {
    filePath = argv[1];
    labelFilepath = argv[2];
    input = argv[3];
    inputType = argv[4];
    std::string str_d, str_t, str_s, str_n;
    int o;
    const char* optstring = "d:t:s:n:";
    while ((o = getopt(argc, argv, optstring)) != -1) {
      switch (o) {
        case 'd':
          str_d = optarg;
          disable_spacemit_ep = std::stoi(str_d);
          break;
        case 't':
          str_t = optarg;
          intra_threads_num = std::stoi(str_t);
          break;
        case 's':
          str_s = optarg;
          score_threshold = std::stof(str_s);
          break;
        case 'n':
          str_n = optarg;
          nms_threshold = std::stof(str_n);
          break;
        case '?':
          std::cout << "[Errot] Unsupported usage" << std::endl;
          break;
      }
    }
  } else {
    std::cout << "run with " << argv[0]
              << " <modelFilepath> <labelFilepath> <input> <inputType> (video "
                 "or cameraId)  option(-d <disable_spacemit_ep>) option(-t "
                 "<intra_threads_num>) "
                 "option(-s score_threshold) option(-n nms_threshold) "
              << std::endl;
    return -1;
  }
  Detector detector{filePath,          labelFilepath,   disable_spacemit_ep,
                    intra_threads_num, score_threshold, nms_threshold};
  SharedDataLoader dataloader;
  if (inputType == "video") {
    if (dataloader.init(input) != 0) {
      std::cout << "[ERROR] dataloader init error" << std::endl;
      return -1;
    }
  } else if (inputType == "cameraId" && isNumber(input) == 1) {
    int cameraId = std::stoi(input);
    if (dataloader.init(cameraId) != 0) {
      std::cout << "[ERROR] dataloader init error" << std::endl;
      return -1;
    }
  } else {
    std::cout << "[ERROR] unsupported input type" << std::endl;
    return -1;
  }
  std::thread t1(Preview, std::ref(dataloader), std::ref(detector));
  // std::this_thread::sleep_for(std::chrono::seconds(5));
  std::thread t2(Detection, std::ref(dataloader), std::ref(detector));
  t1.join();
  t2.join();
  return 0;
}
