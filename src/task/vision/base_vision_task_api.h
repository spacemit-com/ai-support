#ifndef _BASE_VISION_TASK_API_H_
#define _BASE_VISION_TASK_API_H_

#include <string>
#include <vector>

#include "src/task/core/base_task_api.h"

#include "opencv2/opencv.hpp"

template <class OutputType>
class BaseVisionTaskApi : public BaseTaskApi<OutputType, const cv::Mat&>{
    public:
        BaseVisionTaskApi():BaseTaskApi<OutputType, const cv::Mat&>() {};
        ~BaseVisionTaskApi() {};
        // BaseVisionTaskApi is neither copyable nor movable.
        BaseVisionTaskApi(const BaseVisionTaskApi&) = delete;
        BaseVisionTaskApi& operator=(const BaseVisionTaskApi&) = delete;

    protected:
        virtual void Preprocess(std::vector<float> &input_tensors,
                       const cv::Mat& img_raw) override{}
    
};

#endif