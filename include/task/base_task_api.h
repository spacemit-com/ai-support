#ifndef _BASE_TASK_API_H_
#define _BASE_TASK_API_H_

#include <memory>
#include <vector>

#include "engine.h"

#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "onnxruntime_cxx_api.h"


class BaseUntypedTaskApi {
    public:
        explicit BaseUntypedTaskApi(std::unique_ptr<Engine> engine)
            : engine_{std::move(engine)} {}

        virtual ~BaseUntypedTaskApi() = default;

    protected:
        // TODO(b/200258103): It's a short term solution. In the future we will forbid
        // Tasks exposing the underlying TfLiteEngine. Please try not rely on this
        // function.
        //
        // Returns a raw pointer to the underlying TfLiteEngine.
        Engine* GetEngine() { return engine_.get(); }

    private:
        std::unique_ptr<Engine> engine_;
};

class BaseTaskApi : public BaseUntypedTaskApi{
    public:
        explicit BaseTaskApi(std::unique_ptr<Engine> engine)
        : BaseUntypedTaskApi(std::move(engine)) {}
        // BaseTaskApi is neither copyable nor movable.
        BaseTaskApi(const BaseTaskApi&) = delete;
        BaseTaskApi& operator=(const BaseTaskApi&) = delete;

        void Cancel() {};

    protected:
        // Subclasses need to populate input_tensors from api_inputs.
        virtual void Preprocess(std::string& imageFilepath, std::vector<float> &input_tensors) = 0;  
        // Subclasses need to construct OutputType object from output_tensors.
        // Original inputs are also provided as they may be needed.
        virtual void Postprocess() = 0; 
        virtual std::vector<Ort::Value> Infer(std::vector<float>& input_tensors) = 0; 
};

#endif