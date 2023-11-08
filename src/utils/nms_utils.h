#ifndef _NMS_UTILS_H_
#define _NMS_UTILS_H_

#include <vector>

#include "core/types.h"

enum
{
    BLEND=0, OFFSET=1, HARD=2
};

static constexpr const unsigned int max_nms = 30000;

extern void hard_nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
                           float iou_threshold, unsigned int topk);

extern void blending_nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
                               float iou_threshold, unsigned int topk);

extern void offset_nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
                             float iou_threshold, unsigned int topk);

#endif