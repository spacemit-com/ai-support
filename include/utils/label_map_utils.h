#ifndef _LABEL_MAP_UTILS_
#define _LABEL_MAP_UTILS_

#include <string>

// Function to validate the Label file extension.
extern bool checkLabelFileExtension(const std::string& filename);

// Function to read the labels from the labelFilepath.
extern std::vector<std::string> readLabels(std::string& labelFilepath);

#endif