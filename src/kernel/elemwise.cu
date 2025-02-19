#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"
#include "elemwise.h"
#include <algorithm>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

// to support half in future(...maybe far) easily i removed concept

// __global__ void tomoMap(auto *a, size_t len, auto fn_map)
// {
//     auto idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < len)
//     {
//         a[idx] = fn_map(a[idx]);
//     }
// }

#include "reduction.h"

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
        thrust::transform(thrust::device.on(stream), a, a + len, a, fn_map);
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

__host__ __device__ auto tomoRelu(auto &val) 
{
    using T = std::remove_cvref_t<decltype(val)>;
    return max(static_cast<T>(0), val);
}

__host__ __device__ auto tomoLeakyRelu(auto &val) 
{
    using T = std::remove_cvref_t<decltype(val)>;
    return max(static_cast<T>(0.01) * val, val);
}

__host__ __device__ auto tomoInv(auto &val) 
{
    using T = std::remove_cvref_t<decltype(val)>;
    return static_cast<T>(1) / val;
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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return sin(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return sin(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return cos(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return cos(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return tan(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return tan(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return tomoRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return tomoRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return tomoLeakyRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return tomoLeakyRelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return tomoInv(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return tomoInv(x); }, stream);
}

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
// {
//     return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) 
//                             { return x + y; }, threads_per_block, stream);
// }

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
// {
//     return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) 
//                             { return x + y; }, threads_per_block, stream);
// }

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
// {
//     return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) 
//                             { return x - y; }, threads_per_block, stream);
// }

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
// {
//     return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) 
//                             { return x - y; }, threads_per_block, stream);
// }

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMulAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
// {
//     return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) 
//                             { return x * y; }, threads_per_block, stream);
// }

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMulAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
// {
//     return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) 
//                             { return x * y; }, threads_per_block, stream);
// }

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
// {
//     return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) 
//                             { return x / y; }, threads_per_block, stream);
// }

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
// {
//     return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) 
//                             { return x / y; }, threads_per_block, stream);
// }

/////////////////////////
//===----------------------------------------------------------------------===//
// 1) ELU
//===----------------------------------------------------------------------===//

__host__ __device__ auto deviceElu(auto x, auto alpha) 
{
    using T = std::remove_cvref_t<decltype(x)>;
    return (x > static_cast<T>(0)) ? x : alpha * (exp(x) - static_cast<T>(1));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluF(float *a, size_t len, float alpha, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(float const &x) 
                           { return deviceElu<float>(x, alpha); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluD(double *a, size_t len, double alpha, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(double const &x) 
                           { return deviceElu<double>(x, alpha); }, stream);
}

//===----------------------------------------------------------------------===//
// 2) SELU
//    Common constants: alpha ≈ 1.6732632423543772, lambda ≈ 1.0507009873554804
//===----------------------------------------------------------------------===//

__host__ __device__ auto deviceSelu(auto x, auto alpha, auto lambda) 
{
    // x>0 ? lambda*x : lambda * alpha*(exp(x)-1)
    using T = std::remove_cvref_t<decltype(x)>;
    return x > static_cast<T>(0) ? lambda * x : lambda * (alpha * (exp(x) - static_cast<T>(1)));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluF(float *a, size_t len, float alpha, float lambda, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(float const &x) 
                           { return deviceSelu(x, alpha, lambda); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluD(double *a, size_t len, double alpha, double lambda, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(double const &x) 
                           { return deviceSelu(x, alpha, lambda); }, stream);
}

//===----------------------------------------------------------------------===//
// 3) Softplus
//===----------------------------------------------------------------------===//

__host__ __device__ auto deviceSoftplus(auto x) 
{
    // log(1 + exp(x))
    // For numerical stability, consider clamp x or use log1p(exp(x)) if available
    using T = std::remove_cvref_t<decltype(x)>;
    return log(static_cast<T>(1) + exp(x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return deviceSoftplus(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return deviceSoftplus(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 4) Sigmoid
//===----------------------------------------------------------------------===//

__host__ __device__ auto deviceSigmoid(auto x)
{
    using T = std::remove_cvref_t<decltype(x)>;
    return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return deviceSigmoid(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return deviceSigmoid(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 5) Tanh
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return tanh(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return tanh(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 6) Swish
//    f(x) = x / (1 + exp(-x))
//===----------------------------------------------------------------------===//

__host__ __device__ auto deviceSwish(auto x)
{
    using T = std::remove_cvref_t<decltype(x)>;
    return x / (static_cast<T>(1) + exp(-x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return deviceSwish(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return deviceSwish(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 7) GELU (approx)
//    f(x) = 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ) )
//===----------------------------------------------------------------------===//

__host__ __device__ auto deviceGelu(auto x) 
{
    using T = std::remove_cvref_t<decltype(x)>;
    // constants:
    auto constexpr k0 = static_cast<T>(0.5);
    auto constexpr k1 = static_cast<T>(0.7978845608); // sqrt(2/pi)
    auto constexpr k2 = static_cast<T>(0.044715);

    auto t = k1 * (x + k2 * x * x * x);
    return k0 * x * (static_cast<T>(1) + tanh(t));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return deviceGelu(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return deviceGelu(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 8) Hard Sigmoid
//    f(x) = max(0, min(1, 0.2*x + 0.5))
//===----------------------------------------------------------------------===//

__host__ __device__ auto deviceHardSigmoid(auto x) 
{
    auto r = static_cast<decltype(x)>(0.2) * x + static_cast<decltype(x)>(0.5);
    r = (r < static_cast<decltype(x)>(0)) ? static_cast<decltype(x)>(0) : r;
    r = (r > static_cast<decltype(x)>(1)) ? static_cast<decltype(x)>(1) : r;
    return r;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return deviceHardSigmoid(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return deviceHardSigmoid(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 9) Hard Swish
//    f(x) = x * ReLU6(x + 3) / 6
//    ReLU6(z) = min(max(0,z),6)
//===----------------------------------------------------------------------===//

__host__ __device__ auto deviceHardSwish(auto x) 
{
    using T = std::remove_cvref_t<decltype(x)>;
    auto z = x + static_cast<T>(3);
    // clamp z to [0,6]
    z = (z < static_cast<T>(0)) ? static_cast<T>(0) : z;
    z = (z > static_cast<T>(6)) ? static_cast<T>(6) : z;
    return x * z / static_cast<T>(6);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return deviceHardSwish(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return deviceHardSwish(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 10) Softsign
//    f(x) = x / (1 + |x|)
//===----------------------------------------------------------------------===//

__host__ __device__ auto deviceSoftsign(auto x) 
{
    using T = std::remove_cvref_t<decltype(x)>;
    return x / (static_cast<T>(1) + abs(x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return deviceSoftsign(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return deviceSoftsign(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 11) Square
//    f(x) = x^2
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return x * x; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return x * x; }, stream);
}

//===----------------------------------------------------------------------===//
// 12) Sqrt
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return sqrt(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return sqrt(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 13) Log
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return log(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return log(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 14) Exp
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return exp(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return exp(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 15) Abs
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return fabs(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return fabs(x); }, stream);
}

//===----------------------------------------------------------------------===//
// 16) Scale
//    f(x) = scale * x
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleF(float *a, size_t len, float scale, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(float const &x) 
                           { return scale * x; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleD(double *a, size_t len, double scale, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(double const &x) 
                           { return scale * x; }, stream);
}

//===----------------------------------------------------------------------===//
// 17) Softmax
//    We need two passes:
//      Pass 1: transform each x to exp(x), and compute sum of all exp(x)
//      Pass 2: divide each by the sum
//===----------------------------------------------------------------------===//

// For demonstration, we do a naive global sum across the entire array via atomic.

// __global__ void kernelExpInplaceAndPartialSumF(float const *in, float *out, size_t len, float *partialSum)
// {
//     extern __shared__ float sdata_f[];
//     auto tid = threadIdx.x;
//     auto i = blockIdx.x * blockDim.x + threadIdx.x;
//     auto accum = 0.0f;

//     if (i < len)
//     {
//         auto val = exp(in[i]);
//         out[i] = val; // store exponentiated
//         accum = val;
//     }
//     // reduction in shared memory
//     sdata_f[tid] = accum;
//     __syncthreads();

//     // do block-level reduction
//     for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (tid < stride)
//             sdata_f[tid] += sdata_f[tid + stride];
//         __syncthreads();
//     }

//     // write result for this block to global
//     if (tid == 0)
//     {
//         atomicAdd(partialSum, sdata_f[0]);
//     }
// }

// __global__ void kernelExpInplaceAndPartialSumD(double const *in, double *out, size_t len, double *partialSum)
// {
//     extern __shared__ double sdata2[];
//     auto tid = threadIdx.x;
//     auto i = blockIdx.x * blockDim.x + threadIdx.x;
//     auto accum = 0.0;

//     if (i < len)
//     {
//         auto val = exp(in[i]);
//         out[i] = val; // store exponentiated
//         accum = val;
//     }
//     // reduction in shared memory
//     sdata2[tid] = accum;
//     __syncthreads();

//     // do block-level reduction
//     for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (tid < stride)
//             sdata2[tid] += sdata2[tid + stride];
//         __syncthreads();
//     }

//     // write result for this block to global
//     if (tid == 0)
//     {
//         atomicAdd(partialSum, sdata2[0]);
//     }
// }

// __global__ void kernelDivide(auto *out, size_t len, auto *sum)
// {
//     auto i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < len)
//     {
//         out[i] = out[i] / *sum;
//     }
// }

// cudaError_t tomoSoftmax(auto *a, size_t len, unsigned int threads_per_block, cudaStream_t stream) 
// {
//     using T = std::remove_cvref_t<decltype(*a)>;
//     auto err = cudaSuccess;
//     auto d_sum = static_cast<T *>(nullptr);
//     err = cudaMallocAsync((void **)&d_sum, sizeof(T), stream);
//     if (err != cudaSuccess)
//     {
//         return err;
//     }
//     err = cudaMemsetAsync(d_sum, 0, sizeof(T), stream);
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(d_sum, stream);
//         return err;
//     };

//     unsigned int blocks_per_grid = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;
//     unsigned int sharedSize = threads_per_block * sizeof(T);

//     if constexpr (std::is_same_v<T, float>)
//     {
//         kernelExpInplaceAndPartialSumF<<<blocks_per_grid, threads_per_block, sharedSize, stream>>>(a, a, len, d_sum);
//     }
//     else if constexpr (std::is_same_v<T, double>)
//     {
//         kernelExpInplaceAndPartialSumD<<<blocks_per_grid, threads_per_block, sharedSize, stream>>>(a, a, len, d_sum);
//     }
//     else
//     {
//         static_assert(std::is_floating_point_v<T>);
//     }

//     err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(d_sum, stream);
//         return err;
//     }

//     kernelDivide<<<blocks_per_grid, threads_per_block, 0, stream>>>(a, len, d_sum);
//     err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(d_sum, stream);
//         return err;
//     }

//     cudaFreeAsync(d_sum, stream);

//     return cudaSuccess;
// }

////////

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowF(float *a, size_t len, int exponent, cudaStream_t stream)
{
    // We pass 'exponent' by value into the lambda
    return tomoElemwiseMap(a, len, [=] __host__ __device__(float const &x) 
                           { return pow(x, exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowD(double *a, size_t len, int exponent, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(double const &x) 
                           { return pow(x, exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfF(float *a, size_t len, float exponent, cudaStream_t stream)
{
    // We pass 'exponent' by value into the lambda
    return tomoElemwiseMap(a, len, [=] __host__ __device__(float const &x) 
                           { return pow(x, exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfD(double *a, size_t len, double exponent, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(double const &x) 
                           { return pow(x, exponent); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampF(float *a, size_t len, float lower, float upper, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(float const &x) 
                           { return std::clamp(x, lower, upper); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampD(double *a, size_t len, double lower, double upper, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(double const &x) 
                           { return std::clamp(x, lower, upper); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return floor(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return floor(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilF(float *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(float const &x) 
                           { return ceil(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilD(double *a, size_t len, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [] __host__ __device__(double const &x) 
                           { return ceil(x); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftF(float *a, size_t len, float offset, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(float const &x) 
                           { return x + offset; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftD(double *a, size_t len, double offset, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(double const &x) 
                           { return x + offset; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftF(float *a, size_t len, float scale, float offset, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(float const &x) 
                           { return x * scale + offset; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftD(double *a, size_t len, double scale, double offset, cudaStream_t stream)
{
    return tomoElemwiseMap(a, len, [=] __host__ __device__(double const &x) 
                           { return x * scale + offset; }, stream);
}
