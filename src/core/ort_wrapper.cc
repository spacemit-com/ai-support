#include <string>
#include <vector>
#include <iostream>
#include <cmath>

#include "ort_wrapper.h"

void OrtWrapper::Init(std::string instanceName, std::string modelFilepath)
{
    std::unique_ptr<Ort::Env> env(new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str()));
    env_= std::move(env);
    //Creation: The Ort::Session is created here
    std::unique_ptr<Ort::Session> session(new Ort::Session(*env_, modelFilepath.c_str(), sessionOptions_));
    session_=std::move(session);
    sessionOptions_.SetIntraOpNumThreads(4);
    
    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);

    Ort::TypeInfo inputTypeInfo = session_->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    inputDims_ = inputTensorInfo.GetShape();
    inputDims_[0] = abs(inputDims_[0]);

    Ort::TypeInfo outputTypeInfo = session_->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    outputDims_ = outputTensorInfo.GetShape();
    outputDims_[0] = abs(outputDims_[0]);
}

std::vector<Ort::Value> OrtWrapper::Invoke(std::vector<float>& input_tensor_values)
{
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
    Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<std::vector<int64_t>> input_node_dims;
    auto input_dims = tensor_info.GetShape();
    input_dims[0]=abs(input_dims[0]);
    input_node_dims.push_back(input_dims);

    //input tensor size
    int input_tensor_size = 1;
    for (unsigned int i = 0; i < input_dims.size(); ++i)
        input_tensor_size *= input_dims.at(i);

    //output names
    std::vector<const char *> output_node_names;
    int num_outputs = session_->GetOutputCount();
    auto output_name = session_->GetOutputNameAllocated(0, allocator);
    output_node_names.resize(num_outputs);
    output_node_names[0] = output_name.get();

    //output dims
    std::vector<std::vector<int64_t>> output_node_dims;
    output_node_names.resize(num_outputs);
    for (unsigned int i = 0; i < num_outputs; ++i)
    {
        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_dims[0]=abs(output_dims[0]);
        output_node_dims.push_back(output_dims);
    }

    //init output tensor values
    int output_tensor_size = vectorProduct(output_node_dims[0]);
    std::vector<float> output_tensor_values(output_tensor_size);

    //init input tensors and output tensors
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    //build input tensors and output tensors
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, input_tensor_values.data(), input_tensor_size, input_node_dims[0].data(),
        input_node_dims[0].size()));

    output_tensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, output_tensor_values.data(), output_tensor_size,
        output_node_dims[0].data(), output_node_dims[0].size()));

    //run model
    auto outputTensors = session_->Run(Ort::RunOptions{nullptr}, 
                 input_node_names.data(),
                 input_tensors.data(), 
                 1, 
                 output_node_names.data(),
                 num_outputs);
                 
    return outputTensors;
}