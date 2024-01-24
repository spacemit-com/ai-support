#ifndef SUPPORT_INCLUDE_UTILS_TIME_H_
#define SUPPORT_INCLUDE_UTILS_TIME_H_

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

class TimeWatcher {
 public:
  TimeWatcher(const std::string& name) : msg_(name) {
    start_ = std::chrono::steady_clock::now();
  }
  int64_t DurationWithMicroSecond() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_)
        .count();
  }
  ~TimeWatcher() {
    end_ = std::chrono::steady_clock::now();
    std::cout << msg_ << " consumes " << std::fixed << std::setprecision(0)
              << DurationWithMicroSecond() << " ms" << std::endl;
  }

 private:
  std::string msg_;
  std::chrono::steady_clock::time_point start_;
  std::chrono::steady_clock::time_point end_;
};

#endif  // SUPPORT_INCLUDE_UTILS_TIME_H_
