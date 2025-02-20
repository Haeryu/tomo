#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"
#include "elemwise.h"
#include <algorithm>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

// to support __half_raw in future(...maybe far) easily i removed concept

#include "casting.h"

#include "float_op.cuh"

template <typename From, typename To>
cudaError_t tomoCast(From const *a,
                     size_t len,
                     cudaStream_t stream,
                     To *out)
{
    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {
        thrust::transform(thrust::device.on(stream), a, a + len, out, [] __device__(From const &in)
                          {
            if constexpr (std::is_same_v<std::remove_cvref_t<From>, __half_raw>) {
                return static_cast<To>(static_cast<__half>(in));
            } else if constexpr (std::is_same_v<std::remove_cvref_t<From>, __nv_bfloat16_raw>) {
                return static_cast<To>(static_cast<__nv_bfloat16>(in));
            } else {
                return static_cast<To>(in);
           } });
    }
    catch (const thrust::system_error &e)
    {
        if (e.code().category() == thrust::cuda_category())
        {
            return static_cast<cudaError_t>(e.code().value());
        }
        else
        {
            return cudaErrorUnknown;
        }
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHtoB(__half_raw *a, size_t len, cudaStream_t stream, __nv_bfloat16_raw *out)
{
    return tomoCast(a, len, stream, out);
}
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHtoF(__half_raw *a, size_t len, cudaStream_t stream, float *out)
{
    return tomoCast(a, len, stream, out);
}
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHtoD(__half_raw *a, size_t len, cudaStream_t stream, double *out)
{
    return tomoCast(a, len, stream, out);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoBtoH(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream, __half_raw *out)
{
    return tomoCast(a, len, stream, out);
}
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoBtoF(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream, float *out)
{
    return tomoCast(a, len, stream, out);
}
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoBtoD(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream, double *out)
{
    return tomoCast(a, len, stream, out);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFtoH(float *a, size_t len, cudaStream_t stream, __half_raw *out)
{
    return tomoCast(a, len, stream, out);
}
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFtoB(float *a, size_t len, cudaStream_t stream, __nv_bfloat16_raw *out)
{
    return tomoCast(a, len, stream, out);
}
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFtoD(float *a, size_t len, cudaStream_t stream, double *out)
{
    return tomoCast(a, len, stream, out);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDtoH(double *a, size_t len, cudaStream_t stream, __half_raw *out)
{
    return tomoCast(a, len, stream, out);
}
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDtoB(double *a, size_t len, cudaStream_t stream, __nv_bfloat16_raw *out)
{
    return tomoCast(a, len, stream, out);
}
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDtoF(double *a, size_t len, cudaStream_t stream, float *out)
{
    return tomoCast(a, len, stream, out);
}

TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoF16ToBf16(__half_raw val)
{
    return static_cast<__nv_bfloat16>(static_cast<__half>(val));
}
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoF32ToBf16(float val)
{
    return static_cast<__nv_bfloat16>(val);
}
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoF64ToBf16(double val)
{
    return static_cast<__nv_bfloat16>(val);
}

TOMO_EXTERN_C TOMO_OPS_API __half_raw tomoBf16ToF16(__nv_bfloat16_raw val)
{
    return static_cast<__half_raw>(static_cast<__nv_bfloat16>(val));
}
TOMO_EXTERN_C TOMO_OPS_API float tomoBf16ToF32(__nv_bfloat16_raw val)
{
    return static_cast<float>(static_cast<__nv_bfloat16>(val));
}
TOMO_EXTERN_C TOMO_OPS_API double tomoBf16ToF64(__nv_bfloat16_raw val)
{
    return static_cast<double>(static_cast<__nv_bfloat16>(val));
}
