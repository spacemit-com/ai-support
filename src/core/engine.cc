#include <string>
#include <vector>

#include "engine.h"

#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

void Engine::Init(std::string instanceName, std::string modelFilepath)
{
    ortwrapper_.Init(instanceName, modelFilepath);
}

std::vector<Ort::Value> Engine::Interpreter(std::vector<float> &input_values_handler)
{
    return ortwrapper_.Invoke(input_values_handler);
}