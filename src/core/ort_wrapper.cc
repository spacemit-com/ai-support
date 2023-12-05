#include "src/core/ort_wrapper.h"

int OrtWrapper::Init(std::string instanceName, std::string modelFilepath)
{
    std::unique_ptr<Ort::Env> env(new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str()));
    //Creation: The Ort::Session is created here
    env_= std::move(env);
    std::unique_ptr<Ort::Session> session(new Ort::Session(*env_, modelFilepath.c_str(), sessionOptions_));
    session_=std::move(session);
    sessionOptions_.SetIntraOpNumThreads(4);
    sessionOptions_.EnableProfiling(ORT_TSTR("xxx"));
    
    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);
    return 1;
}

std::vector<int64_t> OrtWrapper::GetInputDims()
{    
    Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    auto input_node_dims = tensor_info.GetShape();
    input_node_dims[0]=abs(input_node_dims[0]);
    return input_node_dims;
}

std::vector<std::vector<int64_t>> OrtWrapper::GetOutputDims()
{    
    int num_outputs = session_->GetOutputCount();
    std::vector<std::vector<int64_t>> output_node_dims;
    output_node_dims.resize(num_outputs);
    for (unsigned int i = 0; i < num_outputs; ++i)
    {
        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims[i]=output_dims;
    }
    return output_node_dims;
}

std::vector<Ort::Value> OrtWrapper::Invoke(std::vector<float>& input_tensor_values)
{
#ifdef DEBUG
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif
    //Run Inference

    /* To run inference using ONNX Runtime, the user is responsible for creating and managing the 
    input and output buffers. These buffers could be created and managed via std::vector.
    The linear-format input data should be copied to the buffer for ONNX Runtime inference. */
    

    //init onnxruntime allocator.
    Ort::AllocatorWithDefaultOptions allocator;

    // input node names
    std::vector<const char *> input_node_names;
    auto input_node_name = session_->GetInputNameAllocated(0, allocator);
    const char *input_name = input_node_name.get();
    input_node_names.resize(1);
    input_node_names[0] = input_name;
 
    // input node dims and input dims
    auto input_node_dims = GetInputDims();

    //input tensor size
    int input_tensor_size = 1;
    for (unsigned int i = 0; i < input_node_dims.size(); ++i)
        input_tensor_size *= input_node_dims.at(i);


    //output names initial and build
    std::vector<const char *> output_node_names;
    std::vector<std::string> output_names;
    int num_outputs = session_->GetOutputCount();
    output_node_names.resize(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        output_names.push_back(std::string(""));
    }
    
    for (unsigned int i = 0; i < num_outputs; ++i)
    {
        auto out_name = session_->GetOutputNameAllocated(i, allocator);
        output_names[i].append(out_name.get());
        output_node_names[i] = output_names[i].c_str();
    }

    //init and build input tensors
    std::vector<Ort::Value> input_tensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, input_tensor_values.data(), input_tensor_size, input_node_dims.data(),
        input_node_dims.size()));

    //run model
    auto outputTensors = session_->Run(Ort::RunOptions{nullptr}, 
                 input_node_names.data(),
                 input_tensors.data(), 
                 1, 
                 output_node_names.data(),
                 num_outputs);

#ifdef DEBUG
    std::chrono::steady_clock::time_point end =
    std::chrono::steady_clock::now();
    std::cout << "|-- infer tensor Latency: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
        << " ms" << std::endl;      
#endif       
    return outputTensors;
}