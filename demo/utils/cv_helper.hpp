#include <stdlib.h>     // for: atoi
#ifndef _WIN32
#include <sys/prctl.h>  // for: prctl
#endif
#include <iostream>
#include <sstream>

#include "opencv2/opencv.hpp"

static void setThreadName(const char* name) {
#ifndef _WIN32
  prctl(PR_SET_NAME, name);
#endif /* !_WIN32 */
}

void cvConfig() {
  setThreadName("OpenCV");
#ifdef _DEBUG
  std::string buildInfo = cv::getBuildInformation();
  std::cout << "OpenCV Build Information:\n" << buildInfo << std::endl;
#endif /* _DEBUG */

#if CV_VERSION_MAJOR >= 4
  const char* cv_thread_num = getenv("SUPPORT_OPENCV_THREAD_NUM");
  if (cv_thread_num) {
    int num = atoi(cv_thread_num);
    cv::setNumThreads(num);
  }
  // smoke test to init opencv internal threads
  cv::parallel_for_(cv::Range(0, 16), [&](const cv::Range range) {
#if 0
    std::ostringstream out;
    out << "Thread " << cv::getThreadNum() << "(opencv=" << cv::utils::getThreadID() << "): range " << range.start << "-" << range.end << std::endl;
    std::cout << out.str() << std::flush;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
#endif
  });
#endif /* CV_VERSION_MAJOR */
  std::cout << "[INFO] OpenCV thread num: " << cv::getNumThreads() << std::endl;
}

