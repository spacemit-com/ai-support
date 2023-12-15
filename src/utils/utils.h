#ifndef _SRC_UTILS_UTILS_H_
#define _SRC_UTILS_UTILS_H_

#include <cmath>
#include <vector>
#include <cstdint>  // for: uint32_t

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