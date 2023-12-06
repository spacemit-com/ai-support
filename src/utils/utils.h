#ifndef _SRC_UTILS_UTILS_H_
#define _SRC_UTILS_UTILS_H_

#include <cmath>
#include <vector>
#include <numeric>
#include <stdexcept>

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

extern float sigmoid(float x);
extern float fast_exp(float x);

#endif