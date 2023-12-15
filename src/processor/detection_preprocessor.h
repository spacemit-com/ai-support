#ifndef _DETECTION_PREPROCESSOR_H_
#define _DETECTION_PREPROCESSOR_H_

#include <vector>
#include <string>
#include <chrono>

#include "src/utils/cv2_utils.h"
#include "src/utils/nms_utils.h"
#include "src/processor/processor.h"

#include "opencv2/opencv.hpp"

class DetectionPreprocessor : public Preprocessor{
    public:
    DetectionPreprocessor() {};
    ~DetectionPreprocessor() {};
    // Function to validate the input image file extension.
    void Preprocess(const cv::Mat &mat, 
                    std::vector<int64_t>& input_node_dims, 
                    std::vector<float>& input_tensor_value,
                    unsigned int data_format);
    void PreprocessNanoDet(const cv::Mat &mat, 
                    std::vector<int64_t>& input_node_dims, 
                    std::vector<float>& input_tensor_value);
};

#endif