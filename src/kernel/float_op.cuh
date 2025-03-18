#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

inline __device__ __half_raw max(__half_raw a, __half_raw b)
{
    return __hmax(a, b);
}

inline __device__ __nv_bfloat16_raw max(__nv_bfloat16_raw a, __nv_bfloat16_raw b)
{
    return __hmax(a, b);
}

inline __device__ __half_raw min(__half_raw a, __half_raw b)
{
    return __hmin(a, b);
}

inline __device__ __nv_bfloat16_raw min(__nv_bfloat16_raw a, __nv_bfloat16_raw b)
{
    return __hmin(a, b);
}

inline __device__ __half_raw sin(__half_raw x)
{
    return hsin(x);
}

inline __device__ __nv_bfloat16_raw sin(__nv_bfloat16_raw x)
{
    return hsin(x);
}

inline __device__ __half_raw cos(__half_raw x)
{
    return hcos(x);
}

inline __device__ __nv_bfloat16_raw cos(__nv_bfloat16_raw x)
{
    return hcos(x);
}

inline __device__ __half_raw tan(__half_raw x)
{
    return hsin(x) / hcos(x);
}

inline __device__ __nv_bfloat16_raw tan(__nv_bfloat16_raw x)
{
    return hsin(x) / hcos(x);
}

inline __device__ __half_raw exp(__half_raw x)
{
    return hexp(x);
}

inline __device__ __nv_bfloat16_raw exp(__nv_bfloat16_raw x)
{
    return hexp(x);
}

inline __device__ __half_raw log(__half_raw x)
{
    return hlog(x);
}

inline __device__ __nv_bfloat16_raw log(__nv_bfloat16_raw x)
{
    return hlog(x);
}

inline __device__ __half_raw abs(__half_raw x)
{
    return __habs(x);
}

inline __device__ __nv_bfloat16_raw abs(__nv_bfloat16_raw x)
{
    return __habs(x);
}

inline __device__ __half_raw sqrt(__half_raw x)
{
    return hsqrt(x);
}

inline __device__ __nv_bfloat16_raw sqrt(__nv_bfloat16_raw x)
{
    return hsqrt(x);
}

inline __device__ __half_raw pow(__half_raw x, __half_raw y)
{
    return __float2half(pow(__half2float(x), __half2float(y)));
}

inline __device__ __nv_bfloat16_raw pow(__nv_bfloat16_raw x, __nv_bfloat16_raw y)
{
    return __float2bfloat16(pow(__bfloat162float(x), __bfloat162float(y)));
}

inline __device__ __half_raw pow(__half_raw x, int y)
{
    return __float2half(pow(__half2float(x), y));
}

inline __device__ __nv_bfloat16_raw pow(__nv_bfloat16_raw x, int y)
{
    return __float2bfloat16(pow(__bfloat162float(x), y));
}

inline __device__ __half_raw tanh(__half_raw x)
{
    return htanh(x);
}

inline __device__ __nv_bfloat16_raw tanh(__nv_bfloat16_raw x)
{
    return htanh(x);
}

inline __device__ __half_raw ceil(__half_raw x)
{
    return hceil(x);
}

inline __device__ __nv_bfloat16_raw ceil(__nv_bfloat16_raw x)
{
    return hceil(x);
}

inline __device__ __half_raw floor(__half_raw x)
{
    return hfloor(x);
}

inline __device__ __nv_bfloat16_raw floor(__nv_bfloat16_raw x)
{
    return hfloor(x);
}

inline __host__ __device__ __half_raw operator+(__half_raw const &lh, __half_raw const &rh) { return __hadd(lh, rh); }
inline __host__ __device__ __half_raw operator-(__half_raw const &lh, __half_raw const &rh) { return __hsub(lh, rh); }
inline __host__ __device__ __half_raw operator-(__half_raw const &lh) { return -__half{lh}; }
inline __host__ __device__ __half_raw operator*(__half_raw const &lh, __half_raw const &rh) { return __hmul(lh, rh); }
inline __host__ __device__ __half_raw operator/(__half_raw const &lh, __half_raw const &rh) { return __hdiv(lh, rh); }

inline __host__ __device__ __nv_bfloat16_raw operator+(__nv_bfloat16_raw const &lh, __nv_bfloat16_raw const &rh) { return __nv_bfloat16{lh} + __nv_bfloat16{rh}; }
inline __host__ __device__ __nv_bfloat16_raw operator-(__nv_bfloat16_raw const &lh) { return -__nv_bfloat16{lh}; }
inline __host__ __device__ __nv_bfloat16_raw operator-(__nv_bfloat16_raw const &lh, __nv_bfloat16_raw const &rh) { return __nv_bfloat16{lh} - __nv_bfloat16{rh}; }
inline __host__ __device__ __nv_bfloat16_raw operator*(__nv_bfloat16_raw const &lh, __nv_bfloat16_raw const &rh) { return __nv_bfloat16{lh} * __nv_bfloat16{rh}; }
inline __host__ __device__ __nv_bfloat16_raw operator/(__nv_bfloat16_raw const &lh, __nv_bfloat16_raw const &rh) { return __nv_bfloat16{lh} / __nv_bfloat16{rh}; }


