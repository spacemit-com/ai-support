#ifndef _IMAGE_PREPROCESSOR_H_
#define _IMAGE_PREPROCESSOR_H_

#include <string>
#include <numeric>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "src/processor/processor.h"

#include "opencv2/opencv.hpp"

class ImagePreprocessor : public Preprocessor{
    public:
    ImagePreprocessor() {};
    ~ImagePreprocessor() {};
    void Preprocess(cv::Mat &imageBGR, std::vector<int64_t> inputDims, std::vector<float>& input_tensors);
    template <typename T>
    T vectorProduct(const std::vector<T>& v)
    {
        return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }
};

#endif
