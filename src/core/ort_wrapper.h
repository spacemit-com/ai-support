#ifndef SUPPORT_SRC_CORE_ORT_WRAPPER_H_
#define SUPPORT_SRC_CORE_ORT_WRAPPER_H_

#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include "src/utils/json.hpp"
using json = nlohmann::json;

class OrtWrapper {
 public:
  OrtWrapper() {}
  ~OrtWrapper() {}
  int Init(std::string instanceName,
           std::basic_string<ORTCHAR_T> modelFilepath);
  int Init(json config);
  size_t GetInputCount() { return session_->GetInputCount(); }
  size_t GetOutputCount() { return session_->GetOutputCount(); }
  std::vector<std::vector<int64_t>> GetInputDims();
  std::vector<std::vector<int64_t>> GetOutputDims();

  std::vector<Ort::Value> Invoke(
      std::vector<std::vector<float>>& input_tensor_values);

 protected:
 private:
  std::unique_ptr<Ort::Env> env_;
  Ort::SessionOptions sessionOptions_;
  std::unique_ptr<Ort::Session> session_;
};
#endif  // SUPPORT_SRC_CORE_ORT_WRAPPER_H_
