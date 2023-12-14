#ifndef _SRC_UTILS_UTILS_H_
#define _SRC_UTILS_UTILS_H_

#include <cmath>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cstdint>  // for: uint32_t

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
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