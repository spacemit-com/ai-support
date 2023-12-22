#ifndef _DEMO_UTILS_BOX_UTILS_H_
#define _DEMO_UTILS_BOX_UTILS_H_

#include "task/vision/object_detection_types.h"

#include "opencv2/opencv.hpp"

static void draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<Boxi> &boxes)
{
  if (boxes.empty()) return;
  for (const auto &box: boxes)
  {
    if (box.flag)
    {
      cv::rectangle(mat_inplace, box.rect(), cv::Scalar(255, 255, 0), 2);
      if (box.label_text)
      {
        std::string label_text(box.label_text);
        label_text = label_text + ":" + std::to_string(box.score).substr(0, 4);
        cv::putText(mat_inplace, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                    .6f, cv::Scalar(0, 0, 255), 1);

      }
    }
  }
}

#endif