#ifndef SUPPORT_DEMO_OBJECT_DETECTION_HPP_
#define SUPPORT_DEMO_OBJECT_DETECTION_HPP_

#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "task/vision/object_detection_types.h"

inline void draw_boxes_inplace(cv::Mat &mat_inplace,
                               const std::vector<Boxi> &boxes) {
  if (boxes.empty()) {
    return;
  }
  for (const auto &box : boxes) {
    if (box.flag) {
      cv::rectangle(mat_inplace, box.rect(), cv::Scalar(255, 255, 0), 2);
      if (box.label_text) {
        std::string label_text(box.label_text);
        label_text = label_text + ":" + std::to_string(box.score).substr(0, 4);
        cv::putText(mat_inplace, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                    .5f, cv::Scalar(0, 69, 255), 1);
      }
    }
  }
}

#endif  // SUPPORT_DEMO_OBJECT_DETECTION_HPP_
