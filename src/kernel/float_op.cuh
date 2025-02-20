#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

inline __device__ half max(half a, half b)
{
    return __hmax(a, b);
}

inline __device__ __nv_bfloat16 max(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __hmax(a, b);
}

inline __device__ half min(half a, half b)
{
    return __hmin(a, b);
}

inline __device__ __nv_bfloat16 min(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __hmin(a, b);
}

inline __device__ half sin(half x)
{
    return hsin(x);
}

inline __device__ __nv_bfloat16 sin(__nv_bfloat16 x)
{
    return hsin(x);
}

inline __device__ half cos(half x)
{
    return hcos(x);
}

inline __device__ __nv_bfloat16 cos(__nv_bfloat16 x)
{
    return hcos(x);
}

inline __device__ half tan(half x)
{
    return hsin(x) / hcos(x);
}

inline __device__ __nv_bfloat16 tan(__nv_bfloat16 x)
{
    return hsin(x) / hcos(x);
}

inline __device__ half exp(half x)
{
    return hexp(x);
}

inline __device__ __nv_bfloat16 exp(__nv_bfloat16 x)
{
    return hexp(x);
}

inline __device__ half log(half x)
{
    return hlog(x);
}

inline __device__ __nv_bfloat16 log(__nv_bfloat16 x)
{
    return hlog(x);
}

inline __device__ half abs(half x)
{
    return __habs(x);
}

inline __device__ __nv_bfloat16 abs(__nv_bfloat16 x)
{
    return __habs(x);
}

inline __device__ half sqrt(half x)
{
    return hsqrt(x);
}

inline __device__ __nv_bfloat16 sqrt(__nv_bfloat16 x)
{
    return hsqrt(x);
}

inline __device__ half pow(half x, half y)
{
    return pow(static_cast<float>(x), static_cast<float>(y));
}

inline __device__ __nv_bfloat16 pow(__nv_bfloat16 x, __nv_bfloat16 y)
{
    return pow(static_cast<float>(x), static_cast<float>(y));
}

inline __device__ half pow(half x, int y)
{
    return pow(static_cast<float>(x), y);
}

inline __device__ __nv_bfloat16 pow(__nv_bfloat16 x, int y)
{
    return pow(static_cast<float>(x), y);
}

inline __device__ half tanh(half x)
{
    return htanh(x);
}

inline __device__ __nv_bfloat16 tanh(__nv_bfloat16 x)
{
    return htanh(x);
}

inline __device__ half ceil(half x)
{
    return hceil(x);
}

inline __device__ __nv_bfloat16 ceil(__nv_bfloat16 x)
{
    return hceil(x);
}

inline __device__ half floor(half x)
{
    return hfloor(x);
}

inline __device__ __nv_bfloat16 floor(__nv_bfloat16 x)
{
    return hfloor(x);
}
