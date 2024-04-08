#include "src/core/ort_wrapper.h"

#include <stdlib.h>  // for: getenv atoi

#include <memory>
#include <utility>  // for move

#ifdef _WIN32
#include "src/utils/utils.h"
#endif /* _WIN32 */
#include "utils/time.h"
#ifdef HAS_SPACEMIT_EP
#include "spacemit_ort_env.h"
#endif

int OrtWrapper::Init(const std::string &instance_name,
                     const std::basic_string<ORTCHAR_T> &model_file_path,
                     const int intra_threads_num, const int inter_threads_num) {
  std::unique_ptr<Ort::Env> env(new Ort::Env(
      OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instance_name.c_str()));
  // Creation: The Ort::Session is created here
  env_ = std::move(env);
  sessionOptions_.SetIntraOpNumThreads(intra_threads_num);
  sessionOptions_.SetInterOpNumThreads(inter_threads_num);
#ifdef HAS_SPACEMIT_EP
  const char *disable_spacemit_ep = getenv("SUPPORT_DISABLE_SPACEMIT_EP");
  if (disable_spacemit_ep != nullptr && strcmp(disable_spacemit_ep, "1") == 0) {
    std::cout << "Disable spacemit ep now" << std::endl;
  } else {
    SessionOptionsSpaceMITEnvInit(sessionOptions_);
    // auto providers = Ort::GetAvailableProviders();
    std::cout << "Enable spacemit ep now" << std::endl;
  }
#else
  std::cout << "Unsupport spacemit ep now" << std::endl;
#endif

  const char *opt_model_path = getenv("SUPPORT_OPT_MODEL_PATH");
  if (opt_model_path != nullptr) {
#ifdef _WIN32
    std::wstring wstr = to_wstring(opt_model_path);
    sessionOptions_.SetOptimizedModelFilePath(wstr.c_str());
#else
    sessionOptions_.SetOptimizedModelFilePath(opt_model_path);
#endif /* _WIN32 */
  }
  const char *graph_optimization_level =
      getenv("SUPPORT_GRAPH_OPTIMIZATION_LEVEL");
  if (graph_optimization_level != nullptr) {
    if (strcmp(graph_optimization_level, "ort_disable_all") == 0) {
      sessionOptions_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_DISABLE_ALL);
    } else if (strcmp(graph_optimization_level, "ort_enable_basic") == 0) {
      sessionOptions_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_BASIC);
    } else if (strcmp(graph_optimization_level, "ort_enable_extended") == 0) {
      sessionOptions_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    } else {
      sessionOptions_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_ALL);
    }
  }
  const char *profiling_projects = getenv("SUPPORT_PROFILING_PROJECTS");
  if (profiling_projects != nullptr) {
#ifdef _WIN32
    std::wstring wstr = to_wstring(profiling_projects);
    sessionOptions_.EnableProfiling(wstr.c_str());
#else
    sessionOptions_.EnableProfiling(profiling_projects);
#endif /* _WIN32 */
  }

  const char *log_level_str = getenv("SUPPORT_LOG_LEVEL");
  if (log_level_str != nullptr) {
    int log_level = atoi(log_level_str);
    if (log_level >= 0 && log_level <= 4) {
      sessionOptions_.SetLogSeverityLevel(log_level);
    }
  }
  std::unique_ptr<Ort::Session> session;
  try {
    session = std::make_unique<Ort::Session>(*env_, model_file_path.c_str(),
                                             sessionOptions_);
  } catch (Ort::Exception &e) {
    std::cout << "[ ERROR ] Init failed, onnxruntime error code = "
              << e.GetOrtErrorCode() << ", error message: " << e.what()
              << std::endl;
    return -1;
  }
  session_ = std::move(session);
  return 0;
}

std::vector<std::vector<int64_t>> OrtWrapper::GetInputDims() {
  size_t num_inputs = session_->GetInputCount();
  std::vector<std::vector<int64_t>> input_node_dims;
  input_node_dims.resize(num_inputs);
  for (size_t i = 0; i < num_inputs; ++i) {
    Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    auto input_dims = input_tensor_info.GetShape();
    input_dims[0] = abs(input_dims[0]);
    input_dims[1] = abs(input_dims[1]);
    input_node_dims[i] = input_dims;
  }
  return input_node_dims;
}

std::vector<std::vector<int64_t>> OrtWrapper::GetOutputDims() {
  size_t num_outputs = session_->GetOutputCount();
  std::vector<std::vector<int64_t>> output_node_dims;
  output_node_dims.resize(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    output_node_dims[i] = output_dims;
  }
  return output_node_dims;
}

std::vector<Ort::Value> OrtWrapper::Invoke(
    std::vector<std::vector<float>> &input_tensor_values) {
#ifdef DEBUG
  TimeWatcher t("|-- Infer tensor");
#endif
  // init onnxruntime allocator.
  Ort::AllocatorWithDefaultOptions allocator;

  // input names initial and build
  std::vector<const char *> input_node_names;
  std::vector<std::string> input_names;
  size_t num_inputs = session_->GetInputCount();
  input_node_names.resize(num_inputs);
  for (size_t i = 0; i < num_inputs; i++) {
    input_names.push_back(std::string(""));
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    auto input_name = session_->GetInputNameAllocated(i, allocator);
    input_names[i].append(input_name.get());
    input_node_names[i] = input_names[i].c_str();
  }

  // input node dims and input dims
  auto input_node_dims = GetInputDims();

  // input tensor size
  std::vector<size_t> input_tensor_size;
  input_tensor_size.resize(input_node_dims.size());
  for (size_t i = 0; i < num_inputs; ++i) {
    input_tensor_size[i] = 1;
    for (size_t j = 0; j < input_node_dims[i].size(); ++j) {
      input_tensor_size[i] *= input_node_dims[i][j];
    }
  }

  // output names initial and build
  std::vector<const char *> output_node_names;
  std::vector<std::string> output_names;
  size_t num_outputs = session_->GetOutputCount();
  output_node_names.resize(num_outputs);
  for (size_t i = 0; i < num_outputs; i++) {
    output_names.push_back(std::string(""));
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_name = session_->GetOutputNameAllocated(i, allocator);
    output_names[i].append(output_name.get());
    output_node_names[i] = output_names[i].c_str();
  }

  // init and build input tensors
  std::vector<Ort::Value> input_tensors;
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  for (size_t i = 0; i < num_inputs; i++) {
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, input_tensor_values[i].data(), input_tensor_size[i],
        input_node_dims[i].data(), input_node_dims[i].size()));
  }

  // run model
  auto outputTensors = session_->Run(
      Ort::RunOptions{nullptr}, input_node_names.data(), input_tensors.data(),
      num_inputs, output_node_names.data(), num_outputs);
  return outputTensors;
}
