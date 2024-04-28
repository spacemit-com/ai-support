#ifndef SUPPORT_DEMO_DATALOADER_HPP_
#define SUPPORT_DEMO_DATALOADER_HPP_

#include <cctype>  // for: std::isdigit
#include <memory>
#include <mutex>
#include <queue>
#include <string>
/* opencv header files */
#include "opencv2/opencv.hpp"
/* bianbu-ai-support header files */
#include "utils/utils.h"

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
  bool ifEnable() { return enable; }
  void setDisable() { enable = false; }
  void setPreviewFps(int preview_fps) { preview_fps_ = preview_fps; }
  void setDetectionFps(int detection_fps) { detection_fps_ = detection_fps; }
  int getPreviewFps() { return preview_fps_; }
  int getDetectionFps() { return detection_fps_; }
  int getResizeHeight() { return resize_height_; }
  int getResizeWidth() { return resize_width_; }
  virtual cv::Mat fetchFrame() = 0;
  virtual cv::Mat peekFrame() = 0;

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
  int init(const int camera_id) {
    capture_.open(camera_id);
    if (capture_.isOpened()) {
      return 0;
    } else {
      std::cout << "Open camera capture failed" << std::endl;
      return -1;
    }
  }
  cv::Mat fetchFrame() {
    cv::Mat frame;
    capture_.read(frame);
    return frame;
  }
  cv::Mat peekFrame() { return fetchFrame(); }

 private:
  cv::VideoCapture capture_;
};

#ifndef _WIN32

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>  //for close

static bool isValidCamera(const std::string& path) {
  int fd = open(path.c_str(), O_RDWR);
  if (fd == -1) {
    return false;
  }
  struct v4l2_capability cap;
  if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
    close(fd);
    return false;
  }
  close(fd);
  return (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) != 0;
}

#endif

inline bool isNumber(const std::string& str) {
  for (char const& c : str) {
    if (std::isdigit(c) == 0) {
      return false;
    }
  }
  return true;
}

// 共享式
class SharedDataLoader : public DataLoader {
 public:
  SharedDataLoader(const int& resize_height, const int& resize_width)
      : DataLoader(resize_height, resize_width) {}
  ~SharedDataLoader() {}

  int init(const std::string& path) {
    if (isNumber(path)) {
      return init(std::stoi(path));
    }
    capture_.open(path);
    return setCapture();
  }

  int init(int camera_id) {
#ifndef _WIN32
    capture_.open(camera_id, cv::CAP_V4L2);
    if (!capture_.isOpened()) {
      std::cout
          << "Open camera capture failed, try to figure out right camera id"
          << std::endl;
      std::string path = "/dev/video";
      for (int i = 0; i <= 100; ++i) {
        std::string device_path = path + std::to_string(i);
        if (isValidCamera(device_path)) {
          capture_.open(i, cv::CAP_V4L2);
          if (capture_.isOpened()) {
            break;
          }
        }
      }
    }
#else
    capture_.open(camera_id);
#endif
    return setCapture();
  }

  cv::Mat fetchFrame() {
    cv::Mat frame;
    capture_.read(frame);
    if (!frame.empty()) {
      frame_mutex_.lock();
      resizeUnscale(frame, frame_, getResizeHeight(), getResizeWidth());
      frame_mutex_.unlock();
    }
    return frame;
  }

  cv::Mat peekFrame() {
    cv::Mat frame;
    frame_mutex_.lock();
    std::swap(frame, frame_);
    frame_mutex_.unlock();
    return frame;
  }

  int setCapture() {
    if (capture_.isOpened()) {
      int width = 640;  // Because of k1x performance reasons, the resolution is
                        // set to 640*480
      if (capture_.get(cv::CAP_PROP_FRAME_WIDTH) > 640) {
        if (capture_.set(cv::CAP_PROP_FRAME_WIDTH, width) &&
            capture_.get(cv::CAP_PROP_FRAME_WIDTH) == width) {
          std::cout << "The video capture width is set to " << width
                    << " successfully" << std::endl;
        } else {
          std::cout << "[ WARNING ] Video capture width set to " << width
                    << " failed, the resolution is "
                    << capture_.get(cv::CAP_PROP_FRAME_WIDTH) << "*"
                    << capture_.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
        }
      }
      if (capture_.set(cv::CAP_PROP_FOURCC,
                       cv::VideoWriter::fourcc('M', 'J', 'P', 'G')) &&
          capture_.get(cv::CAP_PROP_FOURCC) ==
              cv::VideoWriter::fourcc('M', 'J', 'P', 'G')) {
        std::cout << "Video capture format has been set to MJPG successfully"
                  << std::endl;
      } else {
        std::cout << "[ WARNING ] Video capture format set to MJPG failed, "
                     "using default format"
                  << std::endl;
      }
      return 0;
    } else {
      std::cout << "Open camera capture failed" << std::endl;
      return -1;
    }
  }

 private:
  std::shared_ptr<cv::Mat> frame;
  cv::Mat frame_;
  std::mutex frame_mutex_;
  cv::VideoCapture capture_;
  std::queue<cv::Mat> frame_queue_;
};

#endif  // SUPPORT_DEMO_DATALOADER_HPP_
