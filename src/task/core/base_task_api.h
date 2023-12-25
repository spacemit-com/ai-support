#ifndef SUPPORT_SRC_TASK_CORE_BASE_TASK_API_H_
#define SUPPORT_SRC_TASK_CORE_BASE_TASK_API_H_

#include <memory>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include "src/core/engine.h"

class BaseUntypedTaskApi {
 public:
  BaseUntypedTaskApi() { engine_ = std::unique_ptr<Engine>(new Engine()); }
  virtual ~BaseUntypedTaskApi() = default;

 protected:
  // TODO(b/200258103): It's a short term solution. In the future we will forbid
  // Tasks exposing the underlying TfLiteEngine. Please try not rely on this
  // function.
  //
  // Returns a raw pointer to the underlying TfLiteEngine.
  Engine* GetEngine() { return engine_.get(); }
  std::unique_ptr<Engine> engine_;
};

template <class OutputType, class... InputTypes>
class BaseTaskApi : public BaseUntypedTaskApi {
 public:
  BaseTaskApi() : BaseUntypedTaskApi() {}
  ~BaseTaskApi() {}
  // BaseTaskApi is neither copyable nor movable.
  BaseTaskApi(const BaseTaskApi&) = delete;
  BaseTaskApi& operator=(const BaseTaskApi&) = delete;
  std::vector<std::vector<int64_t>> GetInputShape() {
    return GetEngine()->GetInputDims();
  }
  void Cancel() {}

 protected:
  // Subclasses need to populate input_tensors from api_inputs.
  virtual void Preprocess(std::vector<std::vector<float>>& input_tensors,
                          InputTypes... api_inputs) = 0;
  // Subclasses need to construct OutputType object from output_tensors.
  // Original inputs are also provided as they may be needed.
  virtual OutputType Postprocess() = 0;
  std::vector<Ort::Value> Infer(
      std::vector<std::vector<float>>& input_tensors) {
    return GetEngine()->Interpreter(input_tensors);
  }
};

#endif  // SUPPORT_SRC_TASK_CORE_BASE_TASK_API_H_
