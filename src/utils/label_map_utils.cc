#include <string>
#include <vector>
#include <fstream>
#include <stdexcept> // To use runtime_error

bool checkLabelFileExtension(const std::string& filename)
{
    size_t pos = filename.rfind('.');
    if (filename.empty())
    {
        throw std::runtime_error("[ ERROR ] The Label file path is empty");
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

std::vector<std::string> readLabels(std::string& labelFilepath)
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
