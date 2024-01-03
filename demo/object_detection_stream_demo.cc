#include <stdlib.h>
#include <unistd.h>

#include <chrono>
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
    cv::Mat _frame;
    capture_.read(_frame);
    frame_mutex_.lock();
    frame = std::make_shared<cv::Mat>(_frame);
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
    cv::Mat temp;
    capture_.read(temp);
    if (!temp.empty()) {
      frame_mutex_.lock();
      frame_ = temp.clone();
      frame_mutex_.unlock();
    }
    return temp;
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
  // cv::Mat frame;
  while (dataloader.ifenable()) {
    // frame = dataloader.peek_frame();   // 取(拷贝)一帧数据
    std::shared_ptr<cv::Mat> frame = dataloader.clone_frame();
    if ((*frame).empty()) {
      continue;
    }
    int flag = detector.infer(*frame);  // 推理并保存检测结果
    if (flag == -1) {
      std::cout << "detection thread quit" << std::endl;
      break;  // 摄像头结束拍摄或者故障
    }
  }
}

// 预览线程
void Preview(DataLoader& dataloader, Detector& detector) {
  // cv::Mat frame;
  ObjectDetectionResult objs;
  auto now = std::chrono::high_resolution_clock::now();
  objs.timestamp = now;
  while (dataloader.ifenable()) {
    // frame = dataloader.fetch_frame();  // 取(搬走)一帧数据
    std::shared_ptr<cv::Mat> frame = dataloader.fetch_frame_v2();
    if ((*frame).empty()) {
      break;
    }
    if (detector.detected())  // 判断原因: detector.detected 不用锁,
                              // detector.get_object 需要锁;
    {
      // 是否有检测结果
      objs = detector.get_object();  // 取(搬走)检测结果(移动赋值)
      if (objs.result_bboxes.size()) {
        std::vector<Boxi> resultBoxes = objs.result_bboxes;
        {
#ifdef DEBUG
          TimeWatcher t("|-- Output result");
#endif
          for (int i = 0; i < resultBoxes.size(); i++) {
            std::cout << "bbox[" << std::setw(2) << i << "]"
                      << " "
                      << "x1y1x2y2: "
                      << "(" << std::setw(4) << resultBoxes[i].x1 << ","
                      << std::setw(4) << resultBoxes[i].y1 << ","
                      << std::setw(4) << resultBoxes[i].x2 << ","
                      << std::setw(4) << resultBoxes[i].y2 << ")"
                      << ", "
                      << "score: " << std::fixed << std::setprecision(3)
                      << std::setw(4) << resultBoxes[i].score << ", "
                      << "label_text: " << resultBoxes[i].label_text
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
      draw_boxes_inplace((*frame), objs.result_bboxes);  // 画框
    }
    cv::imshow("Detection", (*frame));
    cv::waitKey(30);
  }
}

int main(int argc, char* argv[]) {
  std::string filePath, labelFilepath, input, inputType;
  bool disable_spacemit_ep;
  float score_threshold, nms_threshold;
  int intra_threads_num;
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
    std::string argv5, argv6, argv7, argv8;
    int o;
    const char* optstring =
        "d:t:s:n:";  // 有三个选项-abc，其中c选项后有两个冒号，表示后面可选参数
    while ((o = getopt(argc, argv, optstring)) != -1) {
      switch (o) {
        case 'd':
          argv5 = optarg;
          disable_spacemit_ep = std::stoi(argv5);
          break;
        case 't':
          argv6 = optarg;
          std::cout << argv6 << std::endl;
          intra_threads_num = std::stoi(argv6);
          break;
        case 's':
          argv7 = optarg;
          score_threshold = std::stof(argv7);
          break;
        case 'n':
          argv8 = optarg;
          nms_threshold = std::stof(argv8);
          break;
        case '?':
          std::cout << "[Errot] Unsupported usage" << std::endl;
          break;
      }
    }
  } else {
    std::cout << "run with " << argv[0]
              << " <modelFilepath> <labelFilepath> <input> <inputType> (video "
                 "or cameraId)"
              << std::endl;
    return -1;
  }
  if (intra_threads_num == 0) {
    intra_threads_num = 1;
  }
  if (score_threshold == 0.0) {
    score_threshold = 0.5;
  }
  if (nms_threshold == 0.0) {
    nms_threshold = 0.5;
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
