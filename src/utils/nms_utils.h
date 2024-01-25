#ifndef SUPPORT_SRC_UTILS_NMS_UTILS_H_
#define SUPPORT_SRC_UTILS_NMS_UTILS_H_

#include <vector>

#include "task/vision/object_detection_types.h"

enum { BLEND = 0, OFFSET = 1, HARD = 2 };

const unsigned int max_nms = 30000;

void hard_nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
              float iou_threshold, unsigned int topk);

void blending_nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
                  float iou_threshold, unsigned int topk);

// reference: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
void offset_nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
                float iou_threshold, unsigned int topk);

#endif  // SUPPORT_SRC_UTILS_NMS_UTILS_H_
