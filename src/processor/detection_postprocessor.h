#ifndef SUPPORT_SRC_PROCESSOR_DETECTION_POSTPROCESSOR_H_
#define SUPPORT_SRC_PROCESSOR_DETECTION_POSTPROCESSOR_H_

#include <vector>

#include "onnxruntime_cxx_api.h"
#include "processor.h"
#include "src/utils/nms_utils.h"
#include "task/vision/object_detection_types.h"

class DetectionPostprocessor : public Postprocessor {
 public:
  DetectionPostprocessor() {}

  void Postprocess(std::vector<Ort::Value> output_tensors,
                   std::vector<Boxi> &result_boxes,
                   std::vector<std::vector<int64_t>> &input_dims,
                   int img_height, int img_width,
                   std::vector<std::string> &labels, float score_threshold,
                   float iou_threshold, unsigned int topk = 100,
                   unsigned int nms_type = OFFSET);

  void PostprocessYolov6(std::vector<Ort::Value> output_tensors,
                         std::vector<Boxi> &result_boxes,
                         std::vector<std::vector<int64_t>> &input_dims,
                         int img_height, int img_width,
                         std::vector<std::string> &labels,
                         float &score_threshold);

  void PostprocessNanoDetPlus(std::vector<Ort::Value> output_tensors,
                              std::vector<Boxi> &result_boxes,
                              std::vector<std::vector<int64_t>> &input_dims,
                              int img_height, int img_width,
                              std::vector<std::string> &labels,
                              float &score_threshold, float &nms_threshold);

  void PostprocessRtmDet(std::vector<Ort::Value> output_tensors,
                         std::vector<Boxi> &result_boxes,
                         std::vector<std::vector<int64_t>> &input_dims,
                         int img_height, int img_width,
                         std::vector<std::string> &labels,
                         float &score_threshold, float &nms_threshold);

 private:
  void nms(std::vector<Boxf> &input, std::vector<Boxf> &output,
           float iou_threshold, unsigned int topk, unsigned int nms_type);
};
#endif  // SUPPORT_SRC_PROCESSOR_DETECTION_POSTPROCESSOR_H_
