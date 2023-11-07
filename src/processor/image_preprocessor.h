#ifndef _IMAGE_PREPROCESSOR_H_
#define _IMAGE_PREPROCESSOR_H_

#include <vector>
#include <string>
#include <numeric>

#include "processor.h"

#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

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
