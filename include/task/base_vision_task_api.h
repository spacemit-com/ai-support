#ifndef _BASE_VISION_TASK_API_H_
#define _BASE_VISION_TASK_API_H_

#include <string>
#include <vector>

#include "base_task_api.h"
#include "image_preprocessor.h"

#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

class BaseVisionTaskApi : public BaseTaskApi{
    public:
        explicit BaseVisionTaskApi(std::unique_ptr<Engine> engine)
        : BaseTaskApi(std::move(engine)) {}
        ~BaseVisionTaskApi() {};
        // BaseVisionTaskApi is neither copyable nor movable.
        BaseVisionTaskApi(const BaseVisionTaskApi&) = delete;
        BaseVisionTaskApi& operator=(const BaseVisionTaskApi&) = delete;

    protected:
        void Preprocess(std::string& imageFilepath, std::vector<float> &input_tensors) override{
            std::vector<int64_t> inputDims = GetEngine()->GetInputDims();
            preprocessor_->Preprocess(imageFilepath, inputDims, input_tensors);
        }
        std::vector<Ort::Value> Infer(std::vector<float>& input_tensors) override{
            return GetEngine()->Interpreter(input_tensors);
        }
    
    private:
        std::unique_ptr<ImagePreprocessor> preprocessor_ = nullptr;
};

#endif