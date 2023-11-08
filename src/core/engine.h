#ifndef _ENGINE_H_
#define _ENGINE_H_

#include <string>
#include <vector>

#include "src/core/ort_wrapper.h"

#include "opencv2/opencv.hpp"

class Engine
{
    public:
    Engine() {};
    ~Engine() {};
    void Init(std::string instanceName, std::string modelFilepath);
    size_t GetInputCount()
    {return ortwrapper_.GetInputCount();}
    size_t GetOutputCount()   
    {return ortwrapper_.GetOutputCount();}
    std::vector<int64_t> GetInputDims()
    {return ortwrapper_.GetInputDims();}
    std::vector<std::vector<int64_t>> GetOutputDims()
    {return ortwrapper_.GetOutputDims();}
    std::vector<Ort::Value> Interpreter(std::vector<float> &input_values_handler);
    protected:
    private:
    OrtWrapper ortwrapper_;
};

#endif