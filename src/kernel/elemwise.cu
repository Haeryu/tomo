#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"

#include "elemwise.h"

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "float_op.cuh"

cudaError_t tomoElemwise(auto *a,
                         auto const *b,
                         size_t len,
                         auto fn_map,
                         cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*a)>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {
        thrust::transform(thrust::cuda::par_nosync.on(stream), a, a + len, b, a, fn_map);
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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductH(__half_raw *a, __half_raw const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::multiplies<__half_raw>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::multiplies<__nv_bfloat16_raw>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductF(float *a, float const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::multiplies<float>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductD(double *a, double const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::multiplies<double>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivideH(__half_raw *a, __half_raw const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::divides<__half_raw>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivideB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::divides<__nv_bfloat16_raw>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivideF(float *a, float const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::divides<float>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivideD(double *a, double const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::divides<double>(), stream);
}