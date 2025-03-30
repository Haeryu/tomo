#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"
#include "map.h"
#include <algorithm>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>

#include <curand_kernel.h>

// to support __half_raw in future(...maybe far) easily i removed concept

#include "reduction.h"

// #include "float_op.cuh"

cudaError_t tomoElemwiseMap(auto *a,
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
        thrust::transform(thrust::cuda::par_nosync.on(stream), a, a + len, a, fn_map);
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

__device__ auto tomoRelu(auto &val)
{
    using T = std::remove_cvref_t<decltype(val)>;
    return  fmax(static_cast<T>(0.0), (T)val);
}



__device__ auto tomoLeakyRelu(auto &val)
{
    using T = std::remove_cvref_t<decltype(val)>;
    return  (T)fmax(static_cast<T>(0.01) * val,  (T)val);
}

__device__ auto tomoInv(auto &val)
{
    using T = std::remove_cvref_t<decltype(val)>;
    return static_cast<T>(1.0) / val;
}

// __global__ void tomoBinary(auto *a, auto const *b, size_t len, auto fn_binary)
// {
//     auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < len)
//     {
//         a[idx] = fn_binary(a[idx], b[idx]);
//     }
// }

// cudaError_t tomoLaunchBinary(auto *a, auto const *b, size_t len, auto fn_binary, unsigned int threads_per_block, cudaStream_t stream)
// {
//     auto blocks_per_grid = (static_cast<unsigned int>(len) + threads_per_block - 1u) / threads_per_block;
//     tomoBinary<<<blocks_per_grid, threads_per_block, 0, stream>>>(a, b, len, fn_binary);
//     return cudaGetLastError();
// }

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return sin(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16 *)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return sin(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return sin(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return sin(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return cos(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16 *)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return cos(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return cos(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return cos(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return tan(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16 *)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return tan(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return tan(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return tan(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return tomoRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16 *)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return tomoRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return tomoRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return tomoRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return tomoLeakyRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16 *)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return tomoLeakyRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return tomoLeakyRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return tomoLeakyRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return tomoInv(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16 *)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return tomoInv(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return tomoInv(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return tomoInv(x); }, stream);
}

/////////////////////////
//===----------------------------------------------------------------------===//
// 1) ELU
//===----------------------------------------------------------------------===//

__device__ auto deviceElu(auto x, auto alpha)
{
    using T = std::remove_cvref_t<decltype(x)>;
    return (x > static_cast<T>(0.0)) ? x : alpha * ((T)exp(x) - static_cast<T>(1.0));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluH(__half_raw *a, size_t len, __half_raw alpha, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [=] __device__(__half const &x)
                           { return deviceElu(x, (__half )alpha); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw alpha, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return deviceElu(x, (__nv_bfloat16)alpha); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluF(float *a, size_t len, float alpha, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return deviceElu(x, alpha); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluD(double *a, size_t len, double alpha, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return deviceElu(x, alpha); }, stream);
}

//===----------------------------------------------------------------------===//
// 2) SELU
//    Common constants: alpha ≈ 1.6732632423543772, lambda ≈ 1.0507009873554804
//===----------------------------------------------------------------------===//

__device__ auto deviceSelu(auto x, auto alpha, auto lambda)
{
    // x>0 ? lambda*x : lambda * alpha*(exp(x)-1)
    using T = std::remove_cvref_t<decltype(x)>;
    return x > static_cast<T>(0.0) ?  (T)lambda * x :  (T)lambda * ( (T)alpha * ( (T)exp(x) - static_cast<T>(1.0)));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluH(__half_raw *a, size_t len, __half_raw alpha, __half_raw lambda, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [=] __device__(__half const &x)
                           { return deviceSelu(x, (__half )alpha, (__half )lambda); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw alpha, __nv_bfloat16_raw lambda, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return deviceSelu(x, (__nv_bfloat16)alpha, (__nv_bfloat16)lambda); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluF(float *a, size_t len, float alpha, float lambda, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return deviceSelu(x, alpha, lambda); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluD(double *a, size_t len, double alpha, double lambda, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return deviceSelu(x, alpha, lambda); }, stream);
}

//===----------------------------------------------------------------------===//
// 3) Softplus
//===----------------------------------------------------------------------===//

__device__ auto deviceSoftplus(auto x)
{
    // log(1 + exp(x))
    // For numerical stability, consider clamp x or use log1p(exp(x)) if available
    using T = std::remove_cvref_t<decltype(x)>;
    return log(static_cast<T>(1.0) +  (T)exp(x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return deviceSoftplus(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return deviceSoftplus(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return deviceSoftplus(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return deviceSoftplus(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 4) Sigmoid
//===----------------------------------------------------------------------===//

__device__ auto deviceSigmoid(auto x)
{
    using T = std::remove_cvref_t<decltype(x)>;
    return static_cast<T>(1.0) / (static_cast<T>(1.0) +  (T)exp(-x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return deviceSigmoid(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return deviceSigmoid(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return deviceSigmoid(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return deviceSigmoid(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 5) Tanh
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return tanh(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return tanh(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return tanh(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return tanh(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 6) Swish
//    f(x) = x / (1 + exp(-x))
//===----------------------------------------------------------------------===//

__device__ auto deviceSwish(auto x)
{
    using T = std::remove_cvref_t<decltype(x)>;
    return x / (static_cast<T>(1.0) +  (T)exp(-x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return deviceSwish(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return deviceSwish(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return deviceSwish(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return deviceSwish(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 7) GELU (approx)
//    f(x) = 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ) )
//===----------------------------------------------------------------------===//

__device__ auto deviceGelu(auto x)
{
    using T = std::remove_cvref_t<decltype(x)>;
    // constants:
    auto const k0 = static_cast<T>(0.5);
    auto const k1 = static_cast<T>(0.7978845608); // sqrt(2/pi)
    auto const k2 = static_cast<T>(0.044715);

    auto t = k1 * (x + k2 * x * x * x);
    return k0 * x * (static_cast<T>(1.0) +  (T)tanh(t));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return deviceGelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return deviceGelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return deviceGelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return deviceGelu(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 8) Hard Sigmoid
//    f(x) = max(0, min(1, 0.2*x + 0.5))
//===----------------------------------------------------------------------===//

__device__ auto deviceHardSigmoid(auto x)
{
    using T = std::remove_cvref_t<decltype(x)>;
    auto r = static_cast<T>(0.2) * x + static_cast<T>(0.5);
    r = (r < static_cast<T>(0.0)) ? static_cast<T>(0.0) : r;
    r = (r > static_cast<T>(1.0)) ? static_cast<T>(1.0) : r;
    return r;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return deviceHardSigmoid(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return deviceHardSigmoid(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return deviceHardSigmoid(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return deviceHardSigmoid(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 9) Hard Swish
//    f(x) = x * ReLU6(x + 3) / 6
//    ReLU6(z) = min(max(0,z),6)
//===----------------------------------------------------------------------===//

__device__ auto deviceHardSwish(auto x)
{
    using T = std::remove_cvref_t<decltype(x)>;
    auto z = x + static_cast<T>(3.0);
    // clamp z to [0,6]
    z = (z < static_cast<T>(0.0)) ? static_cast<T>(0.0) : z;
    z = (z > static_cast<T>(6.0)) ? static_cast<T>(6.0) : z;
    return x * z / static_cast<T>(6.0);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return deviceHardSwish(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return deviceHardSwish(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return deviceHardSwish(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return deviceHardSwish(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 10) Softsign
//    f(x) = x / (1 + |x|)
//===----------------------------------------------------------------------===//

__device__ auto deviceSoftsign(auto x)
{
    using T = std::remove_cvref_t<decltype(x)>;
    return x / (static_cast<T>(1.0) +  (T)fabs(x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return deviceSoftsign(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return deviceSoftsign(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return deviceSoftsign(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return deviceSoftsign(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 11) Square
//    f(x) = x^2
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return x * x; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return x * x; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return x * x; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return x * x; }, stream);
}

//===----------------------------------------------------------------------===//
// 12) Sqrt
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return sqrt(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return sqrt(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return sqrt(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return sqrt(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 13) Log
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return log(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return log(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return log(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return log(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 14) Exp
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return exp(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return exp(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return exp(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return exp(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 15) Abs
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return (__half )abs((float)x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return (__nv_bfloat16)abs((float)x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return abs(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return abs(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 16) Scale
//    f(x) = scale * x
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleH(__half_raw *a, size_t len, __half_raw scale, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [=] __device__(__half const &x)
                           { return (__half )scale * x; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw scale, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return (__nv_bfloat16)scale * x; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleF(float *a, size_t len, float scale, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return scale * x; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleD(double *a, size_t len, double scale, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return scale * x; }, stream);
}

////////

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowH(__half_raw *a, size_t len, int exponent, cudaStream_t stream)
{
    // We pass 'exponent' by value into the lambda
    return tomoElemwiseMap((__half *)a, len, [=] __device__(__half const &x)
                           { return pow(x, exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowB(__nv_bfloat16_raw *a, size_t len, int exponent, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return pow(x, exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowF(float *a, size_t len, int exponent, cudaStream_t stream)
{
    // We pass 'exponent' by value into the lambda
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return pow(x, exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowD(double *a, size_t len, int exponent, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return pow(x, exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfH(__half_raw *a, size_t len, __half_raw exponent, cudaStream_t stream)
{
    // We pass 'exponent' by value into the lambda
    return tomoElemwiseMap((__half *)a, len, [=] __device__(__half const &x)
                           { return pow(x, (__half)exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw exponent, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return pow(x, (__nv_bfloat16)exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfF(float *a, size_t len, float exponent, cudaStream_t stream)
{
    // We pass 'exponent' by value into the lambda
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return pow(x, exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfD(double *a, size_t len, double exponent, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return pow(x, exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampH(__half_raw *a, size_t len, __half_raw lower, __half_raw upper, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [=] __device__(__half const &x)
                           { return std::clamp(x, (__half )lower, (__half )upper); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw lower, __nv_bfloat16_raw upper, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return std::clamp(x, (__nv_bfloat16)lower, (__nv_bfloat16)upper); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampF(float *a, size_t len, float lower, float upper, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return std::clamp(x, lower, upper); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampD(double *a, size_t len, double lower, double upper, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return std::clamp(x, lower, upper); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return floor(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return floor(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return floor(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return floor(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilH(__half_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [] __device__(__half const &x)
                           { return ceil(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [] __device__(__nv_bfloat16 const &x)
                           { return ceil(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(float const &x)
                           { return ceil(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __device__(double const &x)
                           { return ceil(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftH(__half_raw *a, size_t len, __half_raw offset, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [=] __device__(__half const &x)
                           { return x + (__half )offset; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw offset, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return x + (__nv_bfloat16)offset; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftF(float *a, size_t len, float offset, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return x + offset; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftD(double *a, size_t len, double offset, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return x + offset; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftH(__half_raw *a, size_t len, __half_raw scale, __half_raw offset, cudaStream_t stream)
{
    return tomoElemwiseMap((__half *)a, len, [=] __device__(__half const &x)
                           { return x * (__half )scale + (__half )offset; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw scale, __nv_bfloat16_raw offset, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return x * (__nv_bfloat16)scale + (__nv_bfloat16)offset; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftF(float *a, size_t len, float scale, float offset, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return x * scale + offset; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftD(double *a, size_t len, double scale, double offset, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return x * scale + offset; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtH(__half_raw *a, size_t len, __half_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(__half const &x)
                           { return (__half) !!(x > num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap((__nv_bfloat16*)a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return (__nv_bfloat16) !!(x > (__nv_bfloat16)num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtF(float *a, size_t len, float num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return (float)!!(x > num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtD(double *a, size_t len, double num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return (double)!!(x > num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtUZ(size_t *a, size_t len, size_t num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(size_t const &x)
                           { return (size_t)!!(x > num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtEqH(__half_raw *a, size_t len, __half_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(__half const &x)
                           { return (__half) !!(x >= (__half )num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtEqB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return (__nv_bfloat16) !!(x >= (__nv_bfloat16)num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtEqF(float *a, size_t len, float num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return (float)!!(x >= num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtEqD(double *a, size_t len, double num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return (double)!!(x >= num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtEqUZ(size_t *a, size_t len, size_t num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(size_t const &x)
                           { return (size_t)!!(x >= num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtH(__half_raw *a, size_t len, __half_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(__half const &x)
                           { return (__half) !!(x < (__half )num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return (__nv_bfloat16) !!(x < (__nv_bfloat16)num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtF(float *a, size_t len, float num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return (float)!!(x < num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtD(double *a, size_t len, double num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return (double)!!(x < num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtUZ(size_t *a, size_t len, size_t num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(size_t const &x)
                           { return (size_t)!!(x < num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtEqH(__half_raw *a, size_t len, __half_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(__half const &x)
                           { return (__half) !!(x <= num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtEqB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return (__nv_bfloat16) !!(x <= (__nv_bfloat16)num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtEqF(float *a, size_t len, float num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return (float)!!(x <= num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtEqD(double *a, size_t len, double num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return (double)!!(x <= num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtEqUZ(size_t *a, size_t len, size_t num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(size_t const &x)
                           { return (size_t)!!(x <= num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqH(__half_raw *a, size_t len, __half_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(__half const &x)
                           { return (__half) !!(x == (__half )num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return (__nv_bfloat16) !!(x == (__nv_bfloat16)num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqF(float *a, size_t len, float num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return (float)!!(x == num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqD(double *a, size_t len, double num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return (double)!!(x == num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqUZ(size_t *a, size_t len, size_t num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(size_t const &x)
                           { return (size_t)!!(x == num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoNeqH(__half_raw *a, size_t len, __half_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(__half const &x)
                           { return (__half) !!(x != (__half )num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoNeqB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(__nv_bfloat16 const &x)
                           { return (__nv_bfloat16) !!(x != (__nv_bfloat16)num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoNeqF(float *a, size_t len, float num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(float const &x)
                           { return (float)!!(x != num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoNeqD(double *a, size_t len, double num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(double const &x)
                           { return (double)!!(x != num); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoNeqUZ(size_t *a, size_t len, size_t num, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __device__(size_t const &x)
                           { return (size_t)!!(x != num); }, stream);
}

template <typename T>
cudaError_t tomoMaskedFill(
    T *data,            // [in,out] data to fill in-place
    T const *mask,      // [in] mask array (same length as data)
    T fillValue,        // the value to assign where mask != 0
    size_t len,         // number of elements
    cudaStream_t stream // CUDA stream
)
{
    if (!data || !mask || len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {
        // We zip (data, mask) together and transform into data (in-place).
        thrust::transform(
            thrust::cuda::par_nosync.on(stream),
            thrust::make_zip_iterator(thrust::make_tuple(data, mask)),
            thrust::make_zip_iterator(thrust::make_tuple(data + len, mask + len)),
            data,
            [=] __device__(thrust::tuple<T, T> const &t) -> T
            {
                T d = thrust::get<0>(t); // from data
                T m = thrust::get<1>(t); // from mask
                return (m != T{}) ? fillValue : d;
            });
    }
    catch (const thrust::system_error &e)
    {
        // Convert Thrust exceptions to cudaError_t
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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaskedFillH(
    __half_raw *data,
    __half_raw const *mask,
    __half_raw fillValue,
    size_t len,
    cudaStream_t stream)
{
    return tomoMaskedFill<__half>((__half *)data, (__half *)mask,(__half ) fillValue, len, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaskedFillB(
    __nv_bfloat16_raw *data,
    __nv_bfloat16_raw const *mask,
    __nv_bfloat16_raw fillValue,
    size_t len,
    cudaStream_t stream)
{
    return tomoMaskedFill<__nv_bfloat16>((__nv_bfloat16*)data, (__nv_bfloat16*)mask, (__nv_bfloat16)fillValue, len, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaskedFillF(
    float *data,
    float const *mask,
    float fillValue,
    size_t len,
    cudaStream_t stream)
{
    return tomoMaskedFill<float>(data, mask, fillValue, len, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaskedFillD(
    double *data,
    double const *mask,
    double fillValue,
    size_t len,
    cudaStream_t stream)
{
    return tomoMaskedFill<double>(data, mask, fillValue, len, stream);
}

template <typename T>
cudaError_t tomoTril(
    T *data,     // [in,out] matrix stored in row-major order
    size_t rows, // number of rows
    size_t cols, // number of columns
    T fillValue, // value to assign in the upper triangle (j > i)
    cudaStream_t stream)
{
    size_t len = rows * cols;
    try
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(
            data, thrust::make_counting_iterator<size_t>(0)));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(
            data + len, thrust::make_counting_iterator<size_t>(len)));
        thrust::transform(
            thrust::cuda::par_nosync.on(stream),
            first,
            last,
            data, // output to the same data pointer (in-place)
            [=] __device__(thrust::tuple<T, size_t> const &t) -> T
            {
                T d = thrust::get<0>(t);
                size_t idx = thrust::get<1>(t);
                size_t i = idx / cols;
                size_t j = idx % cols;
                // For lower triangular (tril): if j > i, we're in the upper triangle.
                return (j > i) ? fillValue : d;
            });
    }
    catch (const thrust::system_error &e)
    {
        if (e.code().category() == thrust::cuda_category())
            return static_cast<cudaError_t>(e.code().value());
        else
            return cudaErrorUnknown;
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

template <typename T>
cudaError_t tomoTriu(
    T *data,     // [in,out] matrix stored in row-major order
    size_t rows, // number of rows
    size_t cols, // number of columns
    T fillValue, // value to assign in the lower triangle (i > j)
    cudaStream_t stream)
{
    size_t len = rows * cols;
    try
    {
        auto first = thrust::make_zip_iterator(thrust::make_tuple(
            data, thrust::make_counting_iterator<size_t>(0)));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(
            data + len, thrust::make_counting_iterator<size_t>(len)));
        thrust::transform(
            thrust::cuda::par_nosync.on(stream),
            first,
            last,
            data, // in-place output
            [=] __device__(thrust::tuple<T, size_t> const &t) -> T
            {
                T d = thrust::get<0>(t);
                size_t idx = thrust::get<1>(t);
                size_t i = idx / cols;
                size_t j = idx % cols;
                // For upper triangular (triu): if i > j, we're in the lower triangle.
                return (i > j) ? fillValue : d;
            });
    }
    catch (const thrust::system_error &e)
    {
        if (e.code().category() == thrust::cuda_category())
            return static_cast<cudaError_t>(e.code().value());
        else
            return cudaErrorUnknown;
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

// For __half (assuming __half_raw is defined as in your environment)
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTrilH(
    __half_raw *data,
    size_t rows,
    size_t cols,
    __half_raw fillValue,
    cudaStream_t stream)
{
    return tomoTril<__half>((__half *)data, rows, cols, (__half )fillValue, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTriuH(
    __half_raw *data,
    size_t rows,
    size_t cols,
    __half_raw fillValue,
    cudaStream_t stream)
{
    return tomoTriu<__half>((__half *)data, rows, cols, (__half )fillValue, stream);
}

// For __nv_bfloat16 (assuming __nv_bfloat16_raw is defined appropriately)
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTrilB(
    __nv_bfloat16_raw *data,
    size_t rows,
    size_t cols,
    __nv_bfloat16_raw fillValue,
    cudaStream_t stream)
{
    return tomoTril<__nv_bfloat16>((__nv_bfloat16*)data, rows, cols, (__nv_bfloat16)fillValue, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTriuB(
    __nv_bfloat16_raw *data,
    size_t rows,
    size_t cols,
    __nv_bfloat16_raw fillValue,
    cudaStream_t stream)
{
    return tomoTriu<__nv_bfloat16>((__nv_bfloat16*)data, rows, cols, (__nv_bfloat16)fillValue, stream);
}

// For float
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTrilF(
    float *data,
    size_t rows,
    size_t cols,
    float fillValue,
    cudaStream_t stream)
{
    return tomoTril<float>(data, rows, cols, fillValue, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTriuF(
    float *data,
    size_t rows,
    size_t cols,
    float fillValue,
    cudaStream_t stream)
{
    return tomoTriu<float>(data, rows, cols, fillValue, stream);
}

// For double
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTrilD(
    double *data,
    size_t rows,
    size_t cols,
    double fillValue,
    cudaStream_t stream)
{
    return tomoTril<double>(data, rows, cols, fillValue, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTriuD(
    double *data,
    size_t rows,
    size_t cols,
    double fillValue,
    cudaStream_t stream)
{
    return tomoTriu<double>(data, rows, cols, fillValue, stream);
}

// CUDA kernel to generate the sequence
template <typename T>
__global__ void tomoArangeKernel(T *output, T start, T step, size_t num_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements)
    {
        output[idx] = start + static_cast<T>((float)idx) * step;
    }
}

// Main function to fill the pre-allocated GPU buffer
template <typename T>
cudaError_t tomoArange(T *output, T start, T step, size_t num_elements, cudaStream_t stream)
{
    if (num_elements <= 0)
    {
        return cudaSuccess;
    }
    const int block_size = 256;
    const int num_blocks = (int)(num_elements + block_size - 1) / block_size;
    tomoArangeKernel<T><<<num_blocks, block_size, 0, stream>>>(output, start, step, num_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        return err;
    }
    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArangeH(__half_raw *output, __half_raw start, __half_raw step, size_t num_element, cudaStream_t stream)
{
    return tomoArange<__half>((__half *)output, (__half )start, (__half )step, num_element, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArangeB(__nv_bfloat16_raw *output, __nv_bfloat16_raw start, __nv_bfloat16_raw step, size_t num_element, cudaStream_t stream)
{
    return tomoArange<__nv_bfloat16>((__nv_bfloat16*)output, (__nv_bfloat16)start, (__nv_bfloat16)step, num_element, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArangeF(float *output, float start, float step, size_t num_element, cudaStream_t stream)
{
    return tomoArange<float>(output, start, step, num_element, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArangeD(double *output, double start, double step, size_t num_element, cudaStream_t stream)
{
    return tomoArange<double>(output, start, step, num_element, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArangeUZ(size_t *output, size_t start, size_t step, size_t num_element, cudaStream_t stream)
{
    return tomoArange<size_t>(output, start, step, num_element, stream);
}

__global__ void tomoFillNormalHKernel(__half *a, size_t len, float mean, float stddev, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_normal(&state);
        a[idx] = __float2half(mean + stddev * rand_val);
    }
}

__global__ void tomoFillNormalBKernel(__nv_bfloat16 *a, size_t len, float mean, float stddev, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_normal(&state);
        a[idx] = static_cast<__nv_bfloat16>(mean + stddev * rand_val);
    }
}

__global__ void tomoFillUniformHKernel(__half *a, size_t len, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_uniform(&state);
        a[idx] = __float2half(rand_val);
    }
}

__global__ void tomoFillUniformBKernel(__nv_bfloat16 *a, size_t len, unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_uniform(&state);
        a[idx] = static_cast<__nv_bfloat16>(rand_val);
    }
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillNormalH(__half_raw *a, size_t len, float mean, float stddev, unsigned long long seed, cudaStream_t stream)
{
    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }
    unsigned int threads_per_block = 256;
    unsigned int blocks_per_grid = (unsigned int)(len + threads_per_block - 1) / threads_per_block;
    tomoFillNormalHKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__half *>(a), len, mean, stddev, seed);
    return cudaGetLastError();
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillNormalB(__nv_bfloat16_raw *a, size_t len, float mean, float stddev, unsigned long long seed, cudaStream_t stream)
{
    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }
    unsigned int threads_per_block = 256;
    unsigned int blocks_per_grid = (unsigned int)(len + threads_per_block - 1) / threads_per_block;
    tomoFillNormalBKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 *>(a), len, mean, stddev, seed);
    return cudaGetLastError();
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillUniformH(__half_raw *a, size_t len, unsigned long long seed, cudaStream_t stream)
{
    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }
    unsigned int threads_per_block = 256;
    unsigned int blocks_per_grid = (unsigned int)(len + threads_per_block - 1) / threads_per_block;
    tomoFillUniformHKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__half *>(a), len, seed);
    return cudaGetLastError();
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillUniformB(__nv_bfloat16_raw *a, size_t len, unsigned long long seed, cudaStream_t stream)
{
    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }
    unsigned int threads_per_block = 256;
    unsigned int blocks_per_grid = (unsigned int)(len + threads_per_block - 1) / threads_per_block;
    tomoFillUniformBKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(reinterpret_cast<__nv_bfloat16 *>(a), len, seed);
    return cudaGetLastError();
}


template <typename T>
cudaError_t tomoHasNaN(
    const T* data,      // [in] array to check for NaN
    size_t len,         // number of elements
    bool* result,       // [out] host boolean indicating if any NaN was found
    cudaStream_t stream // CUDA stream
)
{
    if (!data || !result) {
        return cudaErrorInvalidValue;
    }
    if (len == 0) {
        *result = false; 
        return cudaSuccess;
    }

    try {
        auto predicate = [=] __device__ (T x) -> bool {
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                return isnan(x); 
            } else if constexpr (std::is_same_v<T, __half>) {
                return __hisnan(x); 
            } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                return isnan(__bfloat162float(x));
            } else {
                return false; 
            }
        };

        bool has_nan = thrust::any_of(
            thrust::cuda::par_nosync.on(stream), 
            data,                        
            data + len,                 
            predicate                    
        );

        *result = has_nan;
    } catch (const thrust::system_error &e) {
        if (e.code().category() == thrust::cuda_category()) {
            return static_cast<cudaError_t>(e.code().value());
        } else {
            return cudaErrorUnknown;
        }
    } catch (...) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHasNaNH(
    const __half_raw* data,
    size_t len,
    bool* result,
    cudaStream_t stream)
{
    return tomoHasNaN<__half>((const __half*)data, len, result, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHasNaNB(
    const __nv_bfloat16_raw* data,
    size_t len,
    bool* result,
    cudaStream_t stream)
{
    return tomoHasNaN<__nv_bfloat16>((const __nv_bfloat16*)data, len, result, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHasNaNF(
    const float* data,
    size_t len,
    bool* result,
    cudaStream_t stream)
{
    return tomoHasNaN<float>(data, len, result, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHasNaND(
    const double* data,
    size_t len,
    bool* result,
    cudaStream_t stream)
{
    return tomoHasNaN<double>(data, len, result, stream);
}