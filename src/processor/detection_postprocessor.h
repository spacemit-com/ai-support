#ifndef _DETECTION_POSTPROCESSOR_H_
#define _DETECTION_POSTPROCESSOR_H_

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

#include "task/vision/object_detection_types.h"
#include "src/utils/nms_utils.h"
#include "src/utils/utils.h"
#include "src/processor/processor.h"

#include "onnxruntime_cxx_api.h"

class DetectionPostprocessor : public Postprocessor{
    public:
    DetectionPostprocessor() {};

    void Postprocess(std::vector<Ort::Value> output_tensors,
                    std::vector<Boxi> &result_boxes,
                    std::vector<int64_t>& input_dims,
                    int img_height,
                    int img_width,
                    std::vector<std::string> labels,
                    float score_threshold = 0.25f, 
                    float iou_threshold = 0.45f, 
                    unsigned int topk = 100, 
                    unsigned int nms_type = OFFSET);

    void Postprocess_Yolov6(std::vector<Ort::Value> output_tensors,
                std::vector<Boxi> &result_boxes,
                std::vector<int64_t>& input_dims,
                int img_height,
                int img_width,
                std::vector<std::string> labels);

    void Postprocess_NanoDet(std::vector<Ort::Value> output_tensors,
            std::vector<Boxi> &result_boxes,
            std::vector<int64_t>& input_dims,
            int img_height,
            int img_width,
            std::vector<std::string> labels);

    private: 
    void nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
                float iou_threshold, unsigned int topk, unsigned int nms_type);
};

#endif