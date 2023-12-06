#ifndef _INLCUDE_UTILS_TIME_H_
#define _INLCUDE_UTILS_TIME_H_

#include <iostream>
#include <chrono>
#include <iomanip>

// 计时器定义
class TimeWatcher {
public:
  TimeWatcher(const std::string& name) : msg_(name) {
    start_ = std::chrono::steady_clock::now();
  }
  float DurationWithMicroSecond() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
  }
  ~TimeWatcher() {
    end_ = std::chrono::steady_clock::now();
    std::cout << msg_ << " consumes "  <<  std::fixed << std::setprecision(0) << DurationWithMicroSecond() << " ms" << std::endl;
  }
private:
  std::string msg_;
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point end_;
};

#endif