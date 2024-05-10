#include "engine.h"

#include "src/utils/utils.h"

int Engine::Init(const std::string &instance_name,
                 const std::string &model_file_path,
                 const int intra_threads_num, const int inter_threads_num) {
  return ortwrapper_.Init(instance_name,
#ifdef _WIN32
                          to_wstring(model_file_path), intra_threads_num,
                          inter_threads_num
#else
                          model_file_path, intra_threads_num, inter_threads_num
#endif /* _WIN32 */
  );
}

std::vector<Ort::Value> Engine::Interpreter(
    std::vector<std::vector<float>> &input_values_handler) {
  return ortwrapper_.Invoke(input_values_handler);
}
