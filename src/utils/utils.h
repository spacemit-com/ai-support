#ifndef SUPPORT_SRC_UTILS_UTILS_H_
#define SUPPORT_SRC_UTILS_UTILS_H_

#include <string>
#include <vector>

std::vector<std::string> readLabels(const std::string& labelFilepath);

float sigmoid(float x);

float fast_exp(float x);

#ifdef _WIN32
#include <codecvt>
inline std::wstring to_wstring(const std::string& input) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.from_bytes(input);
}
inline std::wstring to_wstring(const char* input) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  std::string str(input);
  return converter.from_bytes(str);
}
#endif /* _WIN32 */

#endif  // SUPPORT_SRC_UTILS_UTILS_H_
