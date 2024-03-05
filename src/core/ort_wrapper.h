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

class OrtWrapper {
 public:
  OrtWrapper() {}
  ~OrtWrapper() {}
  int Init(const std::string& instance_name,
           const std::basic_string<ORTCHAR_T>& model_file_path,
           const int intra_threads_num, const int inter_threads_num);
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
