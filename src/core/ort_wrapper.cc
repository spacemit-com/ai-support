#include "src/core/ort_wrapper.h"

#include <utility>  // for move

#include "utils/time.h"
#ifdef HAS_SPACEMIT_EP
#include "spacemit_ort_env.h"
#endif

int OrtWrapper::Init(std::string instanceName,
                     std::basic_string<ORTCHAR_T> modelFilepath) {
  std::unique_ptr<Ort::Env> env(new Ort::Env(
      OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str()));
  // Creation: The Ort::Session is created here
  env_ = std::move(env);
  sessionOptions_.SetIntraOpNumThreads(2);
  sessionOptions_.AddConfigEntry("session.intra_op.allow_spinning", "0");
  sessionOptions_.SetInterOpNumThreads(2);
  sessionOptions_.AddConfigEntry("session.inter_op.allow_spinning", "0");
#ifdef HAS_SPACEMIT_EP
  SessionOptionsSpaceMITEnvInit(sessionOptions_);
  // auto providers = Ort::GetAvailableProviders();
  std::cout << "Enable spacemit ep now" << std::endl;
#else
  std::cout << "Disable spacemit ep now" << std::endl;
#endif
  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
  // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
  // (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible optimizations
  // sessionOptions_.SetGraphOptimizationLevel(
  // GraphOptimizationLevel::ORT_DISABLE_ALL);
  std::unique_ptr<Ort::Session> session(
      new Ort::Session(*env_, modelFilepath.c_str(), sessionOptions_));
  session_ = std::move(session);
  return 0;
}

int OrtWrapper::Init(json config) {
  std::string instanceName;
  if (config.contains("instance_name")) {
    instanceName = config["instance_name"];
  }
  std::basic_string<ORTCHAR_T> modelFilepath = config["model_path"];
  std::unique_ptr<Ort::Env> env(new Ort::Env(
      OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instanceName.c_str()));
  // Creation: The Ort::Session is created here
  env_ = std::move(env);
  if (!config.contains("disable_spcacemit_ep") ||
      config["disable_spcacemit_ep"] == false) {
#ifdef HAS_SPACEMIT_EP
    SessionOptionsSpaceMITEnvInit(sessionOptions_);
    // auto providers = Ort::GetAvailableProviders();
    std::cout << "Enable spacemit ep now" << std::endl;
#else
    std::cout << "[Warning] Unsupport spacemit ep now" << std::endl;
#endif
  } else {
    std::cout << "Disable spacemit ep now" << std::endl;
  }
  if (config.contains("intra_threads_num")) {
    int intraThreadsnum = config["intra_threads_num"];
    sessionOptions_.SetIntraOpNumThreads(intraThreadsnum);
    sessionOptions_.AddConfigEntry("session.intra_op.allow_spinning", "0");
  } else {
    sessionOptions_.SetIntraOpNumThreads(4);
    sessionOptions_.AddConfigEntry("session.intra_op.allow_spinning", "0");
  }
  sessionOptions_.SetInterOpNumThreads(1);
  sessionOptions_.AddConfigEntry("session.inter_op.allow_spinning", "0");
  if (config.contains("profiling_projects")) {
    std::basic_string<ORTCHAR_T> profiling_projects =
        config["profiling_projects"];
    if (profiling_projects.size()) {
      sessionOptions_.EnableProfiling(profiling_projects.c_str());
    }
  }
  if (config.contains("opt_model_path")) {
    std::basic_string<ORTCHAR_T> opt_model_path = config["opt_model_path"];
    if (opt_model_path.size()) {
      sessionOptions_.SetOptimizedModelFilePath(opt_model_path.c_str());
    }
  }
  if (config.contains("log_level")) {
    int log_level = config["log_level"];
    if (log_level >= 0 && log_level <= 4) {
      sessionOptions_.SetLogSeverityLevel(log_level);
    }
  }
  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
  // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
  // (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible optimizations
  if (config.contains("graph_optimization_level")) {
    if (config["graph_optimization_level"] == "ort_disable_all") {
      sessionOptions_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_DISABLE_ALL);
    } else if (config["graph_optimization_level"] == "ort_enable_basic") {
      sessionOptions_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_BASIC);
    } else if (config["graph_optimization_level"] == "ort_enable_extended") {
      sessionOptions_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    } else {
      sessionOptions_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_ALL);
    }
  }
  std::unique_ptr<Ort::Session> session(
      new Ort::Session(*env_, modelFilepath.c_str(), sessionOptions_));
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

  for (int i = 0; i < static_cast<int>(num_inputs); i++) {
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
