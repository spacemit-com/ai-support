#ifndef _DETECTION_PREPROCESSOR_H_
#define _DETECTION_PREPROCESSOR_H_

#include <vector>
#include <string>

#include "processor.h"
#include "cv2_utils.h"
#include "nms_utils.h"
#include "yolox_config.h"

#include <chrono>

#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

class DetectionPreprocessor : public Preprocessor{
    public:
    DetectionPreprocessor() {};
    ~DetectionPreprocessor() {};
    // Function to validate the input image file extension.
    void Preprocess(cv::Mat &mat, 
                    std::vector<int64_t>& input_node_dims, 
                    std::vector<float>& input_tensor_value,
                    unsigned int data_format,
                    int img_height,
                    int img_width);
    
    private:
    void resize_unscale(const cv::Mat& mat, 
                        cv::Mat& mat_rs,
                        int target_height, 
                        int target_width);
    const float mean_vals[3] = {0.485, 0.456, 0.406};
    const float scale_vals[3] = {0.229, 0.224, 0.225};

};

#endif