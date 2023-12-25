#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "opencv2/opencv.hpp"
#include "task/vision/object_detection_task.h"
#include "utils/box_utils.h"
#include "utils/check_utils.h"
#include "utils/utils.h"

class Detector {
 public:
  Detector() {}
  ~Detector() {}
  // 初始化/反初始化
  int init(std::string filePath, std::string labelFilepath) {
    objectdetectiontask_ = std::unique_ptr<objectDetectionTask>(
        new objectDetectionTask(filePath, labelFilepath));
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
};

class DataLoader {
 public:
  DataLoader() { enable = true; }
  ~DataLoader() {}
  bool ifenable() { return enable; }
  virtual cv::Mat fetch_frame() = 0;
  virtual cv::Mat peek_frame() = 0;

 private:
  bool enable;
};

// 独占式
class ExclusiveDataLoader : public DataLoader {
 public:
  ExclusiveDataLoader() {}
  ~ExclusiveDataLoader() {}
  int init(std::string& path) {
    capture.open(path);
    return 0;
  }
  int init(int cameraId) {
    capture.open(cameraId);
    return 0;
  }
  cv::Mat fetch_frame() {
    cv::Mat frame;
    capture.read(frame);
    return frame;
  }
  cv::Mat peek_frame() { return fetch_frame(); }

 private:
  cv::VideoCapture capture;
};

// 共享式
class SharedDataLoader : public DataLoader {
 public:
  SharedDataLoader() {}
  ~SharedDataLoader() {}
  int init(const std::string& path) {
    capture_.open(path);
    int width = 1280;
    int height = 720;
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    return 0;
  }
  int init(int cameraId) {
    capture_.open(cameraId);
    int width = 1280;
    int height = 720;
    capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    return 0;
  }
  cv::Mat fetch_frame() {
    frame_mutex_.lock();
    if (frame_queue_.size()) {
      frame_ = frame_queue_.front();
      frame_queue_.pop();  // 弹出一帧
    }                      // 有缓存
    else {
      capture_.read(frame_);  // 无缓存: 新取一帧
    }
    frame_mutex_.unlock();
    return frame_;
  }
  cv::Mat peek_frame() {
    frame_mutex_.lock();
    capture_.read(frame_);      // 取最新一帧数据
    frame_queue_.push(frame_);  // 缓存(预览)
    frame_mutex_.unlock();
    return frame_;
  }

 private:
  cv::Mat frame_;
  std::mutex frame_mutex_;
  cv::VideoCapture capture_;
  std::queue<cv::Mat> frame_queue_;
};

// 检测线程
void Detection(DataLoader& dataloader, Detector& detector) {
  cv::Mat frame;
  while (dataloader.ifenable()) {
    frame = dataloader.peek_frame();   // 取(拷贝)一帧数据
    int flag = detector.infer(frame);  // 推理并保存检测结果
    if (flag == -1) {
      break;  // 摄像头结束拍摄或者故障
    }
  }
}

// 预览线程
void Preview(DataLoader& dataloader, Detector& detector) {
  cv::Mat frame;
  while (dataloader.ifenable()) {
    frame = dataloader.fetch_frame();  // 取(搬走)一帧数据
    if (frame.empty()) {
      break;
    }
    if (detector.detected())  // 判断原因: detector.detected 不用锁,
                              // detector.get_object 需要锁;
    {
      // 是否有检测结果
      ObjectDetectionResult objs =
          detector.get_object();  // 取(搬走)检测结果(移动赋值)
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
        draw_boxes_inplace(frame, objs.result_bboxes);  // 画框
      }
      // cv::imshow("Detection", frame);
      cv::waitKey(30);
    }  // 调用 detector.detected 和 detector.get_object 期间,
       // 检测结果依然可能被刷新
  }
}

int main(int argc, char* argv[]) {
  Detector detector;
  std::string filePath, labelFilepath, input, inputType;
  if (argc == 5) {
    filePath = argv[1];
    labelFilepath = argv[2];
    input = argv[3];
    inputType = argv[4];
  } else {
    std::cout << "run with " << argv[0]
              << " <modelFilepath> <labelFilepath> <input> <inputType> (video "
                 "or cameraId)"
              << std::endl;
    return 0;
  }

  if (detector.init(filePath, labelFilepath) != 0) {
    std::cout << "[ERROR] detector init error" << std::endl;
  }
  SharedDataLoader dataloader;
  if (inputType == "video") {
    if (dataloader.init(input) != 0) {
      std::cout << "[ERROR] dataloader init error" << std::endl;
    }
  } else if (inputType == "cameraId" && isNumber(input) == 1) {
    int cameraId = std::stoi(input);
    if (dataloader.init(cameraId) != 0) {
      std::cout << "[ERROR] dataloader init error" << std::endl;
    }
  } else {
    std::cout << "[ERROR] unsupported input type" << std::endl;
  }
  std::thread t1(Detection, std::ref(dataloader), std::ref(detector));
  std::thread t2(Preview, std::ref(dataloader), std::ref(detector));
  t1.join();
  t2.join();
  return 0;
}
