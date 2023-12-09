#include "src/utils/utils.h"
#include <cstdint>

#include <cstdint>  // for: uint32_t

float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}

float fast_exp(float x)
{
    union
    {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}
