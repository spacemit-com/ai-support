#ifndef _CLASSIFICATION_POSTPROCESSOR_H_
#define _CLASSIFICATION_POSTPROCESSOR_H_

#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <cmath>
#include <stdexcept> // To use runtime_error

#include "processor.h"
#include "onnxruntime_cxx_api.h"

class ClassificationPostprocessor : public Postprocessor{
    public:
    ClassificationPostprocessor() {};
    ~ClassificationPostprocessor() {};
    // Function to validate the input image file extension.
    void Postprocess(std::vector<Ort::Value> output_tensors, std::vector<std::string> labels);

    //Handling divide by zero
    float division(float num, float den);
};

#endif
