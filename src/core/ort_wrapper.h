#ifndef _ORT_WRAPPER_H_
#define _ORT_WRAPPER_H_

#include <vector>
#include <string>
#include <numeric>
#include <chrono>
#include <iostream>
#include <cmath>
#include <memory>

#include "opencv2/opencv.hpp"

#include "onnxruntime_cxx_api.h"

class OrtWrapper
{
    public:
    OrtWrapper() {};
    ~OrtWrapper() {};
    int Init(std::string instanceName, std::string modelFilepath);
    size_t GetInputCount()
    {return session_->GetInputCount();}
    size_t GetOutputCount()   
    {return session_->GetOutputCount();}
    std::vector<int64_t> GetInputDims();
    std::vector<std::vector<int64_t>> GetOutputDims();

    template <typename T>
    T vectorProduct(const std::vector<T>& v)
    {
        return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }

    std::vector<Ort::Value> Invoke(std::vector<float> &input_values_handler);
    protected:
    private:
    std::unique_ptr<Ort::Env> env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
};
#endif