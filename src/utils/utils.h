#ifndef SUPPORT_SRC_UTILS_UTILS_H_
#define SUPPORT_SRC_UTILS_UTILS_H_

#include <string>

#include "include/utils/utils.h"
#include "src/utils/json.hpp"
using json = nlohmann::json;

bool checkLabelFileExtension(const std::string& filename);

std::vector<std::string> readLabels(const std::string& labelFilepath);

bool checkModelFileExtension(const std::string& filename);

int checkConfigFileExtension(const std::string& filename);

int configCheck(const json& config);

float sigmoid(float x);

float fast_exp(float x);

#endif  // SUPPORT_SRC_UTILS_UTILS_H_
