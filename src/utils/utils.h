#ifndef _SRC_UTILS_UTILS_H_
#define _SRC_UTILS_UTILS_H_

#include <string>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdint>  // for: uint32_t

#include "include/utils/utils.h"
#include "src/utils/json.hpp"
using json = nlohmann::json;

static bool checkLabelFileExtension(const std::string& filename)
{
    size_t pos = filename.rfind('.');
    if (filename.empty())
    {
        std::cout<<"[ ERROR ] The Label file path is empty"<<std::endl;
        return false;
    }
    if (pos == std::string::npos)
        return false;
    std::string ext = filename.substr(pos+1);
    if (ext == "txt") {
        return true;
    } else {
        return false;
    }
}

static std::vector<std::string> readLabels(std::string& labelFilepath)
{
    std::vector<std::string> labels;
    std::string line;
    std::ifstream fp(labelFilepath);
    while (std::getline(fp, line))
    {
        labels.push_back(line);
    }
    return labels;
}

static bool checkModelFileExtension(const std::string& filename)
{
    size_t pos = filename.rfind('.');
    if (filename.empty())
    {
        std::cout<<"[ ERROR ] The Model file path is empty"<<std::endl;
        return false;
    }
    if (pos == std::string::npos)
        return false;
    std::string ext = filename.substr(pos+1);
    if (ext == "onnx") {
        return true;
    } else {
        return false;
    }
}

static int checkConfigFileExtension(const std::string& filename)
{
    size_t pos = filename.rfind('.');
    if (filename.empty())
    {
        std::cout<<"[ ERROR ] The Config file path is empty"<<std::endl;
        return false;
    }
    if (pos == std::string::npos)
        return false;
    std::string ext = filename.substr(pos+1);
    if (ext == "json") {
        return true;
    } else {
        return false;
    }
}

static int configCheck(const json& config)
{
    if(!config.contains("model_path") || !config.contains("label_path"))
    {
        return 1;
    }
    else if(!checkModelFileExtension(config["model_path"]) || !checkLabelFileExtension(config["label_path"]))
    {
        return 1;
    }
    else if(!exists_check(config["model_path"]) || !exists_check(config["label_path"]))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

static float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}

static float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

#endif