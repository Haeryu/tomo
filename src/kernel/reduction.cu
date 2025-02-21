#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"
#include "reduction.h"

// #include <cmath>
#include <cfloat> // for FLT_MAX, etc.
#include <math_constants.h>
#include <limits>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include "float_op.cuh"

cudaError_t tomoReduceMap(auto const *a,
                          size_t len,
                          auto *host_out,
                          auto init,
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
        *host_out = thrust::reduce(thrust::cuda::par_nosync.on(stream), a, a + len, init, fn_map);
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

cudaError_t tomoSumReduce(auto const *a,
                          size_t len,
                          auto *host_out,
                          cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*a)>;
    return tomoReduceMap(a, len, host_out, static_cast<T>(0), thrust::plus<T>{}, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceH(__half_raw const *a,
                                                      size_t len,
                                                      __half_raw *host_out,
                                                      cudaStream_t stream)
{
    return tomoSumReduce(a, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceB(__nv_bfloat16_raw const *a,
                                                      size_t len,
                                                      __nv_bfloat16_raw *host_out,
                                                      cudaStream_t stream)
{
    return tomoSumReduce(a, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceF(float const *a,
                                                      size_t len,
                                                      float *host_out,
                                                      cudaStream_t stream)
{
    return tomoSumReduce(a, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceD(double const *a,
                                                      size_t len,
                                                      double *host_out,
                                                      cudaStream_t stream)
{
    return tomoSumReduce(a, len, host_out, stream);
}

cudaError_t tomoMean(auto const *a,
                     size_t len,
                     auto *host_out,
                     cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*a)>;

    auto err = tomoSumReduce(a, len, host_out, stream);

    if (err != cudaSuccess)
    {
        return err;
    }

    if (len > 0)
    {
        if constexpr (std::is_same_v<T, __nv_bfloat16_raw>)
        {
            *host_out = static_cast<__nv_bfloat16>(*host_out) / static_cast<__nv_bfloat16>(len);
        }
        else if constexpr (std::is_same_v<T, __half_raw>)
        {
            *host_out = static_cast<__half>(*host_out) / static_cast<__half>(len);
        }
        else
        {
            *host_out = *host_out / static_cast<T>(len);
        }
    }

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanH(__half_raw const *a,
                                                 size_t len,
                                                 __half_raw *host_out,
                                                 cudaStream_t stream)
{
    return tomoMean(a, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanB(__nv_bfloat16_raw const *a,
                                                 size_t len,
                                                 __nv_bfloat16_raw *host_out,
                                                 cudaStream_t stream)
{
    return tomoMean(a, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanF(float const *a,
                                                 size_t len,
                                                 float *host_out,
                                                 cudaStream_t stream)
{
    return tomoMean(a, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanD(double const *a,
                                                 size_t len,
                                                 double *host_out,
                                                 cudaStream_t stream)
{
    return tomoMean(a, len, host_out, stream);
}

cudaError_t tomoMin(const auto *in,
                    size_t len,
                    auto *host_out,
                    cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*in)>;
    return tomoReduceMap(in, len, host_out, std::numeric_limits<T>::max(), thrust::minimum<T>{}, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinH(__half_raw const *in,
                                                size_t len,
                                                __half_raw *host_out,
                                                cudaStream_t stream)
{
    return tomoMin(in, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinB(__nv_bfloat16_raw const *in,
                                                size_t len,
                                                __nv_bfloat16_raw *host_out,
                                                cudaStream_t stream)
{
    return tomoMin(in, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinF(float const *in,
                                                size_t len,
                                                float *host_out,
                                                cudaStream_t stream)
{
    return tomoMin(in, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinD(double const *in,
                                                size_t len,
                                                double *host_out,
                                                cudaStream_t stream)
{
    return tomoMin(in, len, host_out, stream);
}

cudaError_t tomoMax(auto const *in,
                    size_t len,
                    auto *host_out,
                    cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*in)>;
    return tomoReduceMap(in, len, host_out, std::numeric_limits<T>::lowest(), thrust::maximum<T>{}, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxH(__half_raw const *in,
                                                size_t len,
                                                __half_raw *host_out,
                                                cudaStream_t stream)
{
    return tomoMax(in, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxB(__nv_bfloat16_raw const *in,
                                                size_t len,
                                                __nv_bfloat16_raw *host_out,
                                                cudaStream_t stream)
{
    return tomoMax(in, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxF(float const *in,
                                                size_t len,
                                                float *host_out,
                                                cudaStream_t stream)
{
    return tomoMax(in, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxD(double const *in,
                                                size_t len,
                                                double *host_out,
                                                cudaStream_t stream)
{
    return tomoMax(in, len, host_out, stream);
}

template <typename T>
cudaError_t tomoL2Norm(T const *a,
                       size_t len,
                       T *host_out,
                       cudaStream_t stream)
{
    return tomoReduceMap(a, len, host_out, static_cast<T>(0), [] __host__ __device__(T const &lhs, T const &rhs)
                         { return lhs + rhs * rhs; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormH(__half_raw const *a,
                                                   size_t len,
                                                   __half_raw *host_out,
                                                   cudaStream_t stream)
{
    return tomoL2Norm(a, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormB(__nv_bfloat16_raw const *a,
                                                   size_t len,
                                                   __nv_bfloat16_raw *host_out,
                                                   cudaStream_t stream)
{
    return tomoL2Norm(a, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormF(float const *a,
                                                   size_t len,
                                                   float *host_out,
                                                   cudaStream_t stream)
{
    return tomoL2Norm(a, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormD(double const *a,
                                                   size_t len,
                                                   double *host_out,
                                                   cudaStream_t stream)
{
    return tomoL2Norm(a, len, host_out, stream);
}

template <typename T>
cudaError_t tomoL1Norm(T const *a,
                       size_t len,
                       T *host_out,
                       cudaStream_t stream)
{
    return tomoReduceMap(a, len, host_out, static_cast<T>(0), [] __host__ __device__(const T &lhs, const T &rhs)
                         { return lhs + abs(rhs); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormH(__half_raw const *a,
                                                   size_t len,
                                                   __half_raw *host_out,
                                                   cudaStream_t stream)
{
    return tomoL1Norm(a, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormB(__nv_bfloat16_raw const *a,
                                                   size_t len,
                                                   __nv_bfloat16_raw *host_out,
                                                   cudaStream_t stream)
{
    return tomoL1Norm(a, len, host_out, stream);
}
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormF(float const *a,
                                                   size_t len,
                                                   float *host_out,
                                                   cudaStream_t stream)
{
    return tomoL1Norm(a, len, host_out, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormD(double const *a,
                                                   size_t len,
                                                   double *host_out,
                                                   cudaStream_t stream)
{
    return tomoL1Norm(a, len, host_out, stream);
}
