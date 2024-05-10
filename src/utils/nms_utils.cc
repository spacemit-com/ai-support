#include "nms_utils.h"

#include <algorithm>
#include <cmath>  // for: std::exp
#include <vector>

#include "task/vision/object_detection_types.h"

void hard_nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
              float iou_threshold, unsigned int topk) {
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const Boxf &a, const Boxf &b) { return a.score > b.score; });
  const size_t box_num = input.size();
  std::vector<int> merged(box_num, 0);

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i) {
    if (merged[i]) continue;
    std::vector<Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j) {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));
      if (iou > iou_threshold) {
        merged[j] = 1;
        buf.push_back(input[j]);
      }
    }
    output.push_back(buf[0]);

    // keep top k
    count += 1;
    if (count >= topk) break;
  }
}

void blending_nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
                  float iou_threshold, unsigned int topk) {
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const Boxf &a, const Boxf &b) { return a.score > b.score; });
  const size_t box_num = input.size();
  std::vector<int> merged(box_num, 0);

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i) {
    if (merged[i]) continue;
    std::vector<Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j) {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));
      if (iou > iou_threshold) {
        merged[j] = 1;
        buf.push_back(input[j]);
      }
    }

    float total = 0.f;
    for (unsigned int k = 0; k < buf.size(); ++k) {
      total += std::exp(buf[k].score);
    }
    Boxf rects;
    for (unsigned int l = 0; l < buf.size(); ++l) {
      float rate = std::exp(buf[l].score) / total;
      rects.x1 += buf[l].x1 * rate;
      rects.y1 += buf[l].y1 * rate;
      rects.x2 += buf[l].x2 * rate;
      rects.y2 += buf[l].y2 * rate;
      rects.score += buf[l].score * rate;
    }
    output.push_back(rects);

    // keep top k
    count += 1;
    if (count >= topk) break;
  }
}

// reference: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
void offset_nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
                float iou_threshold, unsigned int topk) {
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const Boxf &a, const Boxf &b) { return a.score > b.score; });
  const size_t box_num = input.size();
  std::vector<int> merged(box_num, 0);

  const float offset = 4096.f;
  /** Add offset according to classes.
   * That is, separate the boxes into categories, and each category performs its
   * own NMS operation. The same offset will be used for those predicted to be
   * of the same category. Therefore, the relative positions of boxes of the
   * same category will remain unchanged. Box of different classes will be
   * farther away after offset, because offsets are different. In this way, some
   * overlapping but different categories of entities are not filtered out by
   * the NMS. Very clever!
   */
  for (unsigned int i = 0; i < box_num; ++i) {
    input[i].x1 += static_cast<float>(input[i].label) * offset;
    input[i].y1 += static_cast<float>(input[i].label) * offset;
    input[i].x2 += static_cast<float>(input[i].label) * offset;
    input[i].y2 += static_cast<float>(input[i].label) * offset;
  }

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i) {
    if (merged[i]) continue;
    std::vector<Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j) {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));
      if (iou > iou_threshold) {
        merged[j] = 1;
        buf.push_back(input[j]);
      }
    }
    output.push_back(buf[0]);

    // keep top k
    count += 1;
    if (count >= topk) break;
  }

  /** Substract offset.*/
  if (!output.empty()) {
    for (unsigned int i = 0; i < output.size(); ++i) {
      output[i].x1 -= static_cast<float>(output[i].label) * offset;
      output[i].y1 -= static_cast<float>(output[i].label) * offset;
      output[i].x2 -= static_cast<float>(output[i].label) * offset;
      output[i].y2 -= static_cast<float>(output[i].label) * offset;
    }
  }
}
