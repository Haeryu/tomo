#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>

inline __device__ __half max(__half a, __half b)
{
    return __hmax(a, b);
}

inline __device__ __nv_bfloat16 max(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __hmax(a, b);
}

inline __device__ __half min(__half a, __half b)
{
    return __hmin(a, b);
}

inline __device__ __nv_bfloat16 min(__nv_bfloat16 a, __nv_bfloat16 b)
{
    return __hmin(a, b);
}

inline __device__ __half sin(__half x)
{
    return hsin(x);
}

inline __device__ __nv_bfloat16 sin(__nv_bfloat16 x)
{
    return hsin(x);
}

inline __device__ __half cos(__half x)
{
    return hcos(x);
}

inline __device__ __nv_bfloat16 cos(__nv_bfloat16 x)
{
    return hcos(x);
}

inline __device__ __half tan(__half x)
{
    return hsin(x) / hcos(x);
}

inline __device__ __nv_bfloat16 tan(__nv_bfloat16 x)
{
    return hsin(x) / hcos(x);
}

inline __device__ __half exp(__half x)
{
    return hexp(x);
}

inline __device__ __nv_bfloat16 exp(__nv_bfloat16 x)
{
    return hexp(x);
}

inline __device__ __half log(__half x)
{
    return hlog(x);
}

inline __device__ __nv_bfloat16 log(__nv_bfloat16 x)
{
    return hlog(x);
}

inline __device__ __half abs(__half x)
{
    return __habs(x);
}

inline __device__ __nv_bfloat16 abs(__nv_bfloat16 x)
{
    return __habs(x);
}

inline __device__ __half sqrt(__half x)
{
    return hsqrt(x);
}

inline __device__ __nv_bfloat16 sqrt(__nv_bfloat16 x)
{
    return hsqrt(x);
}

inline __device__ __half pow(__half x, __half y)
{
    return __float2half(pow(__half2float(x), __half2float(y)));
}

inline __device__ __nv_bfloat16 pow(__nv_bfloat16 x, __nv_bfloat16 y)
{
    return __float2bfloat16(pow(__bfloat162float(x), __bfloat162float(y)));
}

inline __device__ __half pow(__half x, int y)
{
    return __float2half(pow(__half2float(x), y));
}

inline __device__ __nv_bfloat16 pow(__nv_bfloat16 x, int y)
{
    return __float2bfloat16(pow(__bfloat162float(x), y));
}

inline __device__ __half tanh(__half x)
{
    return htanh(x);
}

inline __device__ __nv_bfloat16 tanh(__nv_bfloat16 x)
{
    return htanh(x);
}

inline __device__ __half ceil(__half x)
{
    return hceil(x);
}

inline __device__ __nv_bfloat16 ceil(__nv_bfloat16 x)
{
    return hceil(x);
}

inline __device__ __half floor(__half x)
{
    return hfloor(x);
}

inline __device__ __nv_bfloat16 floor(__nv_bfloat16 x)
{
    return hfloor(x);
}

// inline __host__ __device__ __half operator+(__half const &lh, __half const &rh) { return __hadd(lh, rh); }
// inline __host__ __device__ __half operator-(__half const &lh, __half const &rh) { return __hsub(lh, rh); }
// inline __host__ __device__ __half operator-(__half const &lh) { return -__half{lh}; }
// inline __host__ __device__ __half operator*(__half const &lh, __half const &rh) { return __hmul(lh, rh); }
// inline __host__ __device__ __half operator/(__half const &lh, __half const &rh) { return __hdiv(lh, rh); }

// inline __host__ __device__ __nv_bfloat16 operator+(__nv_bfloat16 const &lh, __nv_bfloat16 const &rh) { return __nv_bfloat16{lh} + __nv_bfloat16{rh}; }
// inline __host__ __device__ __nv_bfloat16 operator-(__nv_bfloat16 const &lh) { return -__nv_bfloat16{lh}; }
// inline __host__ __device__ __nv_bfloat16 operator-(__nv_bfloat16 const &lh, __nv_bfloat16 const &rh) { return __nv_bfloat16{lh} - __nv_bfloat16{rh}; }
// inline __host__ __device__ __nv_bfloat16 operator*(__nv_bfloat16 const &lh, __nv_bfloat16 const &rh) { return __nv_bfloat16{lh} * __nv_bfloat16{rh}; }
// inline __host__ __device__ __nv_bfloat16 operator/(__nv_bfloat16 const &lh, __nv_bfloat16 const &rh) { return __nv_bfloat16{lh} / __nv_bfloat16{rh}; }
