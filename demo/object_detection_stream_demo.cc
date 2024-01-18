#include <stdlib.h>
#include <unistd.h>  //for getopt

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
  DataLoader(const int& resize_height, const int& resize_width) {
    enable = true;
    resize_height_ = resize_height;
    resize_width_ = resize_width;
    preview_fps_ = 0;
    detection_fps_ = 0;
  }
  ~DataLoader() {}
  bool ifenable() { return enable; }
  void set_disable() { enable = false; }
  void set_preview_fps(int preview_fps) { preview_fps_ = preview_fps; }
  void set_detection_fps(int detection_fps) { detection_fps_ = detection_fps; }
  int get_preview_fps() { return preview_fps_; }
  int get_detection_fps() { return detection_fps_; }
  int get_resize_height() { return resize_height_; }
  int get_resize_width() { return resize_width_; }
  virtual cv::Mat fetch_frame() = 0;
  virtual cv::Mat peek_frame() = 0;

 private:
  bool enable;
  int resize_height_;
  int resize_width_;
  int preview_fps_;
  int detection_fps_;
};

// 独占式
class ExclusiveDataLoader : public DataLoader {
 public:
  ExclusiveDataLoader(const int& resize_height, const int& resize_width)
      : DataLoader(resize_height, resize_width) {}
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
  SharedDataLoader(const int& resize_height, const int& resize_width)
      : DataLoader(resize_height, resize_width) {}
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
    if (!capture_.isOpened()) {
      std::cout
          << "Open camera capture failed, try to figure out right cameraId"
          << std::endl;
      std::string path = "/dev/video";
      for (int i = 0; i <= 100; ++i) {
        std::string device_path = path + std::to_string(i);
        if (is_valid_camera(device_path)) {
          cameraId = i;
          break;
        }
      }
    }
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

  cv::Mat fetch_frame() {
    cv::Mat frame, temp;
    capture_.read(frame);
    resize_unscale(frame, temp, get_resize_height(), get_resize_width());
    frame_mutex_.lock();
    frame_ = temp.clone();
    frame_mutex_.unlock();
    return frame;
  }
  cv::Mat peek_frame() {
    cv::Mat frame;
    frame_mutex_.lock();
    frame = frame_.clone();  // 深拷贝
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
  int flag;
  while (dataloader.ifenable()) {
    auto start = std::chrono::high_resolution_clock::now();
    frame = dataloader.peek_frame();  // 取(拷贝)一帧数据
    if ((frame).empty()) {
      continue;
    }
    int flag = detector.infer(frame);  // 推理并保存检测结果
    auto end = std::chrono::high_resolution_clock::now();
    auto detection_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    dataloader.set_detection_fps(1000 / (detection_duration.count()));
    if (flag == -1) {
      std::cout << "[Error] infer frame failed" << std::endl;
      break;  // 摄像头结束拍摄或者故障
    }
  }
  std::cout << "detection thread quit" << std::endl;
}

// 预览线程
void Preview(DataLoader& dataloader, Detector& detector) {
  cv::Mat frame;
  ObjectDetectionResult objs;
  auto now = std::chrono::high_resolution_clock::now();
  objs.timestamp = now;
  while (dataloader.ifenable()) {
    // auto start = std::chrono::high_resolution_clock::now();
    frame = dataloader.fetch_frame();  // 取(搬走)一帧数据
    if ((frame).empty()) {
      break;
    }
    if (detector.detected())  // 判断原因: detector.detected 不用锁,
                              // detector.get_object 需要锁;
    {
      // 是否有检测结果
      objs = detector.get_object();  // 取(搬走)检测结果(移动赋值)
      if (objs.result_bboxes.size()) {
        int input_height = dataloader.get_resize_height();
        int input_width = dataloader.get_resize_width();
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
    // 调用 detector.detected 和 detector.get_object 期间,
    // 检测结果依然可能被刷新
    now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - objs.timestamp);
    if (duration.count() < 1000) {
      draw_boxes_inplace((frame), objs.result_bboxes);  // 画框
    }
    int preview_fps = dataloader.get_preview_fps();
    int detection_fps = dataloader.get_detection_fps();
    /*
    cv::putText(frame, "preview fps: " + std::to_string(preview_fps),
                cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5f,
                cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
                */
    cv::putText(frame, "detection fps: " + std::to_string(detection_fps),
                cv::Point(500, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5f,
                cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    cv::imshow("Detection", (frame));
    cv::waitKey(10);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto preview_duration =
    // std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // dataloader.set_preview_fps(1000 / (preview_duration.count()));
    if (cv::getWindowProperty("Detection", cv::WND_PROP_VISIBLE) < 1) {
      dataloader.set_disable();
      break;
    }
  }
  std::cout << "preview thread quit" << std::endl;
  cv::destroyAllWindows();
}

int main(int argc, char* argv[]) {
  std::string filePath, labelFilepath, input, inputType;
  bool disable_spacemit_ep{false};
  float score_threshold{0.4}, nms_threshold{0.5};
  int intra_threads_num{1}, resize_height{320}, resize_width{320};
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
    int o;
    const char* optstring = "d:t:s:n:w:h:";
    while ((o = getopt(argc, argv, optstring)) != -1) {
      switch (o) {
        case 'd':
          disable_spacemit_ep = atoi(optarg);
          break;
        case 't':
          intra_threads_num = atoi(optarg);
          break;
        case 's':
          score_threshold = atof(optarg);
          break;
        case 'n':
          nms_threshold = atof(optarg);
          break;
        case 'w':
          resize_width = atoi(optarg);
          break;
        case 'h':
          resize_height = atoi(optarg);
          break;
        case '?':
          std::cout << "[ERROR] Unsupported usage" << std::endl;
          break;
      }
    }
  } else {
    std::cout << "run with " << argv[0]
              << " <modelFilepath> <labelFilepath> <input> <inputType> (video "
                 "or cameraId)  option(-d <disable_spacemit_ep>) option(-t "
                 "<intra_threads_num>) "
                 "option(-s score_threshold) option(-n nms_threshold) "
                 "option(-h resize_height) option(-w resize_width)"
              << std::endl;
    return -1;
  }
  Detector detector{filePath,          labelFilepath,   disable_spacemit_ep,
                    intra_threads_num, score_threshold, nms_threshold};
  SharedDataLoader dataloader{resize_height, resize_width};
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
