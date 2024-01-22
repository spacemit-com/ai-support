#ifndef SUPPORT_SRC_PROCESSOR_ESTIMATION_POSTPROCESSOR_H_
#define SUPPORT_SRC_PROCESSOR_ESTIMATION_POSTPROCESSOR_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "src/processor/processor.h"
#include "src/utils/nms_utils.h"
#include "src/utils/utils.h"
#include "task/vision/pose_estimation_types.h"

class EstimationPostprocessor : public Postprocessor {
 public:
  EstimationPostprocessor() {}
  void Postprocess(std::vector<Ort::Value> output_tensors,
                   std::pair<cv::Mat, cv::Mat> crop_result_pair,
                   std::vector<PosePoint> &result_points);
};

#endif  // SUPPORT_SRC_PROCESSOR_ESTIMATION_POSTPROCESSOR_H_
