#ifndef SUPPORT_SRC_PROCESSOR_CLASSIFICATION_POSTPROCESSOR_H_
#define SUPPORT_SRC_PROCESSOR_CLASSIFICATION_POSTPROCESSOR_H_

#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "processor.h"
#include "task/vision/image_classification_types.h"

class ClassificationPostprocessor : public Postprocessor {
 public:
  ClassificationPostprocessor() {}
  ~ClassificationPostprocessor() {}
  // Function to validate the input image file extension.
  ImageClassificationResult Postprocess(std::vector<Ort::Value> output_tensors,
                                        std::vector<std::string> &labels);

  // Handling divide by zero
  float division(float num, float den);
};

#endif  // SUPPORT_SRC_PROCESSOR_CLASSIFICATION_POSTPROCESSOR_H_
