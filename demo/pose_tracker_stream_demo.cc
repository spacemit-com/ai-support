#include <pthread.h>
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
#include "opencv2/opencv.hpp"
#include "pose_estimation.hpp"
#include "task/vision/object_detection_task.h"
#include "task/vision/pose_estimation_task.h"
#ifdef DEBUG
#include "utils/time.h"
#endif

#include "utils/utils.h"

class Tracker {
 public:
  Tracker(const std::string& detFilePath, const std::string& poseFilePath,
          const std::string& labelFilepath, const bool disableSpacemitEp,
          const int intraThreadsNum, const float scoreThreshold,
          const float nmsThreshold) {
    detFilePath_ = detFilePath;
    poseFilePath_ = poseFilePath;
    labelFilepath_ = labelFilepath;
    disableSpacemitEp_ = disableSpacemitEp;
    intraThreadsNum_ = intraThreadsNum;
    scoreThreshold_ = scoreThreshold;
    nmsThreshold_ = nmsThreshold;
  }
  ~Tracker() {}
  // 初始化/反初始化
  int init() {
    objectdetectiontask_ =
        std::unique_ptr<objectDetectionTask>(new objectDetectionTask(
            detFilePath_, labelFilepath_, disableSpacemitEp_, intraThreadsNum_,
            scoreThreshold_, nmsThreshold_));
    poseestimationtask_ =
        std::unique_ptr<poseEstimationTask>(new poseEstimationTask(
            poseFilePath_, disableSpacemitEp_, intraThreadsNum_));
    return 0;
  }

  int uninit() { return 0; }

  // 推理
  int infer(cv::Mat frame) {
    if (frame.empty()) {
      return -1;
    }
    ObjectDetectionResult objs_temp = objectdetectiontask_->Detect(frame);
    int count{0}, flag{0};
    for (count = 0; count < static_cast<int>(objs_temp.result_bboxes.size());
         count++) {
      if (objs_temp.result_bboxes[count].label == 0) {
        flag = 1;
        break;
      }
    }
    if (flag) {
      PoseEstimationResult poses_temp =
          poseestimationtask_->Estimate(frame, objs_temp.result_bboxes[count]);
      poses_mutex_.lock();
      poses_array_.push(poses_temp);  // 直接替换掉当前的 objs_array_
      poses_mutex_.unlock();
      return poses_array_.size();
    } else {
      return 0;
    }
  }

  // 查询检测结果
  int estimated() { return poses_array_.size(); }

  // 移走检测结果
  struct PoseEstimationResult get_pose() {
    struct PoseEstimationResult poses_moved;
    poses_mutex_.lock();
    poses_moved = poses_array_.back();
    std::queue<struct PoseEstimationResult> empty;
    std::swap(empty, poses_array_);
    poses_mutex_.unlock();
    return poses_moved;
  }

 private:
  std::mutex poses_mutex_;
  std::queue<struct PoseEstimationResult> poses_array_;
  std::unique_ptr<objectDetectionTask> objectdetectiontask_;
  std::unique_ptr<poseEstimationTask> poseestimationtask_;
  std::string poseFilePath_;
  std::string detFilePath_;
  std::string labelFilepath_;
  bool disableSpacemitEp_;
  int intraThreadsNum_;
  float scoreThreshold_;
  float nmsThreshold_;
};

// 检测线程
void Track(DataLoader& dataloader, Tracker& tracker) {
  if (tracker.init() != 0) {
    std::cout << "[ERROR] tracker init error" << std::endl;
    return;
  }
  cv::Mat frame;
  while (dataloader.ifenable()) {
    auto start = std::chrono::steady_clock::now();
    frame = dataloader.peek_frame();  // 取(拷贝)一帧数据
    if ((frame).empty()) {
      continue;
    }
    int flag = tracker.infer(frame);  // 推理并保存检测结果
    auto end = std::chrono::steady_clock::now();
    auto detection_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    dataloader.set_detection_fps(1000 / (detection_duration.count()));
    if (flag == 0) {
      std::cout << "[Warning] unable to catch person" << std::endl;  // 无人
    }
    if (flag == -1) {
      std::cout << "[Error] infer frame failed" << std::endl;
      break;  // 摄像头结束拍摄或者故障
    }
  }
  std::cout << "track thread quit" << std::endl;
}

// 预览线程
void Preview(DataLoader& dataloader, Tracker& tracker) {
  cv::Mat frame;
  PoseEstimationResult poses;
  auto now = std::chrono::steady_clock::now();
  poses.timestamp = now;
  int count = 0;
  int dur = 0;
  while (dataloader.ifenable()) {
    auto start = std::chrono::steady_clock::now();
    frame = dataloader.fetch_frame();  // 取(搬走)一帧数据
    if ((frame).empty()) {
      break;
    }
    if (tracker.estimated())  // 判断原因: detector.detected 不用锁,
                              // detector.get_object 需要锁;
    {
      // 是否有检测结果
      poses = tracker.get_pose();  // 取(搬走)检测结果(移动赋值)
      if (poses.result_points.size()) {
        int input_height = dataloader.get_resize_height();
        int input_width = dataloader.get_resize_width();
        int img_height = frame.rows;
        int img_width = frame.cols;
        float resize_ratio = std::min(
            static_cast<float>(input_height) / static_cast<float>(img_height),
            static_cast<float>(input_width) / static_cast<float>(img_width));
        float dw = (input_width - resize_ratio * img_width) / 2;
        float dh = (input_height - resize_ratio * img_height) / 2;
        for (int i = 0; i < static_cast<int>(poses.result_points.size()); i++) {
          poses.result_points[i].x =
              (poses.result_points[i].x - dw) / resize_ratio;
          poses.result_points[i].y =
              (poses.result_points[i].y - dh) / resize_ratio;
        }
      }
    }
    // 调用 detector.detected 和 detector.get_object 期间,
    // 检测结果依然可能被刷新
    now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - poses.timestamp);
    if (duration.count() < 1000 && poses.result_points.size()) {
      draw_points_inplace((frame), poses.result_points);  // 画框
    }
    int preview_fps = dataloader.get_preview_fps();
    int detection_fps = dataloader.get_detection_fps();
    const char* showfps = getenv("SHOWFPS");
    if (showfps != nullptr) {
      cv::putText(frame, "preview fps: " + std::to_string(preview_fps),
                  cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5f,
                  cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
      cv::putText(frame, "detection fps: " + std::to_string(detection_fps),
                  cv::Point(500, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5f,
                  cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    cv::imshow("Track", (frame));
    cv::waitKey(10);
    auto end = std::chrono::steady_clock::now();
    auto preview_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    count++;
    dur = dur + preview_duration.count();
    if (dur >= 1000) {
      dataloader.set_preview_fps(count);
      dur = 0;
      count = 0;
    }
    if (cv::getWindowProperty("Track", cv::WND_PROP_VISIBLE) < 1) {
      dataloader.set_disable();
      break;
    }
  }
  std::cout << "preview thread quit" << std::endl;
  cv::destroyAllWindows();
}

#ifndef _WIN32
void setThreadName(std::thread& thread, const char* name) {
  pthread_setname_np(thread.native_handle(), name);
}
#endif

int main(int argc, char* argv[]) {
  std::string detFilePath, poseFilePath, labelFilepath, input, inputType;
  bool disable_spacemit_ep{false};
  float score_threshold{0.4}, nms_threshold{0.5};
  int intra_threads_num{1}, resize_height{320}, resize_width{320};
  if (argc == 6) {
    detFilePath = argv[1];
    poseFilePath = argv[2];
    labelFilepath = argv[3];
    input = argv[4];
    inputType = argv[5];
  } else if (argc > 6) {
    detFilePath = argv[1];
    poseFilePath = argv[2];
    labelFilepath = argv[3];
    input = argv[4];
    inputType = argv[5];
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
              << " <detModelFilepath> <poseModelFilepath> <labelFilepath> "
                 "<input> <inputType> (video "
                 "or cameraId)  option(-d <disable_spacemit_ep>) option(-t "
                 "<intra_threads_num>) "
                 "option(-s <score_threshold>) option(-n <nms_threshold>) "
                 "option(-h <resize_height>) option(-w <resize_width>)"
              << std::endl;
    return -1;
  }
  Tracker tracker{detFilePath,         poseFilePath,      labelFilepath,
                  disable_spacemit_ep, intra_threads_num, score_threshold,
                  nms_threshold};
  SharedDataLoader dataloader{resize_height, resize_width};
  if (dataloader.init(input) != 0) {
    std::cout << "[ERROR] dataloader init error" << std::endl;
    return -1;
  }

  std::thread t1(Preview, std::ref(dataloader), std::ref(tracker));
  // std::this_thread::sleep_for(std::chrono::seconds(5));
  std::thread t2(Track, std::ref(dataloader), std::ref(tracker));
#ifndef _WIN32
  setThreadName(t1, "PreviewThread");
  setThreadName(t2, "TrackerThread");
#endif
  t1.join();
  t2.join();
  return 0;
}
