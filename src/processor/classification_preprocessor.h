#ifndef _CLASSIFICATION_PREPROCESSOR_H_
#define _CLASSIFICATION_PREPROCESSOR_H_

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "src/processor/processor.h"

#include "opencv2/opencv.hpp"

class ClassificationPreprocessor : public Preprocessor{
    public:
    ClassificationPreprocessor() {};
    ~ClassificationPreprocessor() {};
    void Preprocess(cv::Mat &imageBGR, std::vector<std::vector<int64_t>> inputDims, std::vector<std::vector<float>>& input_tensor_values);
};

#endif
