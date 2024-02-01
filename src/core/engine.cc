#include "src/core/engine.h"

#ifdef _WIN32
#include <codecvt>
inline std::wstring to_wstring(const std::string& input) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.from_bytes(input);
}
#endif /* _WIN32 */

int Engine::Init(std::string instanceName, std::string modelFilepath) {
  return ortwrapper_.Init(instanceName,
#ifdef _WIN32
                          to_wstring(modelFilepath)
#else
                          modelFilepath
#endif /* _WIN32 */
  );
}

int Engine::Init(json config) { return ortwrapper_.Init(config); }

std::vector<Ort::Value> Engine::Interpreter(
    std::vector<std::vector<float>>& input_values_handler) {
  return ortwrapper_.Invoke(input_values_handler);
}
