#include "classification_postprocessor.h"

#include <cmath>      // for: std::exp
#include <limits>     // for: numeric_limits<>
#include <stdexcept>  // for: runtime_error

#include "utils/time.h"

float ClassificationPostprocessor::division(float num, float den) {
  if (den == 0) {
    throw std::runtime_error(
        "[ ERROR ] Math error: Attempted to divide by Zero\n");
  }
  return (num / den);
}

ImageClassificationResult ClassificationPostprocessor::Postprocess(
    std::vector<Ort::Value> output_tensors, std::vector<std::string> &labels) {
#ifdef DEBUG
  TimeWatcher t("|-- Postprocess");
#endif
  int predId = 0;
  float activation = 0;
  float maxActivation = std::numeric_limits<float>::lowest();
  float expSum = 0;
  /* The inference result could be found in the buffer for the output tensors,
  which are usually the buffer from std::vector instances. */
  Ort::Value &pred = output_tensors.at(0);
  const float *output_pred_ptr = pred.GetTensorData<float>();
  for (size_t i = 0; i < labels.size(); i++) {
    activation = output_pred_ptr[i];
    expSum += std::exp(activation);
    if (activation > maxActivation) {
      predId = i;
      maxActivation = activation;
    }
  }
  ImageClassificationResult result;
  result.label = predId;
  result.label_text = labels.at(predId);
  result.score = std::exp(maxActivation) / expSum;
  return result;
}
