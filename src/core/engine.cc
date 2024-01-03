#include "src/core/engine.h"

int Engine::Init(std::string instanceName, std::string modelFilepath,
                 const bool disable_spacemit_ep, const int intra_threads_num) {
  return ortwrapper_.Init(instanceName, modelFilepath, intra_threads_num,
                          disable_spacemit_ep);
}

int Engine::Init(json config) { return ortwrapper_.Init(config); }

std::vector<Ort::Value> Engine::Interpreter(
    std::vector<std::vector<float>> &input_values_handler) {
  return ortwrapper_.Invoke(input_values_handler);
}
