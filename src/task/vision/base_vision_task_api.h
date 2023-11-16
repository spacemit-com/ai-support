#ifndef _BASE_VISION_TASK_API_H_
#define _BASE_VISION_TASK_API_H_

#include <string>
#include <vector>

#include "src/task/core/base_task_api.h"
#include "src/processor/image_preprocessor.h"

#include "opencv2/opencv.hpp"

template <class OutputType>
class BaseVisionTaskApi : public BaseTaskApi<OutputType, cv::Mat&>{
    public:
        BaseVisionTaskApi():BaseTaskApi<OutputType, cv::Mat&>() {};
        ~BaseVisionTaskApi() {};
        // BaseVisionTaskApi is neither copyable nor movable.
        BaseVisionTaskApi(const BaseVisionTaskApi&) = delete;
        BaseVisionTaskApi& operator=(const BaseVisionTaskApi&) = delete;

    protected:
        void Preprocess(std::vector<float> &input_tensors,
                        cv::Mat& img_raw) override{
            auto inputDims = this->GetInputShape();
            preprocessor_->Preprocess(img_raw, inputDims, input_tensors);
        }
    
    private:
        std::unique_ptr<ImagePreprocessor> preprocessor_ = nullptr;
};

#endif