#include "src/core/engine.h"

int Engine::Init(std::string instanceName, std::string modelFilepath)
{
    return ortwrapper_.Init(instanceName, modelFilepath);
}

std::vector<Ort::Value> Engine::Interpreter(std::vector<float> &input_values_handler)
{
    return ortwrapper_.Invoke(input_values_handler);
}