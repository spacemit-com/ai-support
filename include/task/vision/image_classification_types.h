#ifndef _IMAGE_CLASSIFICATION_TYPES_H_
#define _IMAGE_CLASSIFICATION_TYPES_H_

#include <string>

struct ImageClassificationResult
{
  std::string label_text;
  int label;
  float score;
  int timestamp;
};
#endif