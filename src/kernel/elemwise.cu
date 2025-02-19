#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"
#include "elemwise.h"
#include <algorithm>

// to support half in future(...maybe far) easily i removed concept

__device__ auto tomoRelu(auto val) noexcept
{
    using T = std::remove_cvref_t<decltype(val)>;
    return max(static_cast<T>(0), val);
}

__device__ auto tomoLeakyRelu(auto val) noexcept
{
    using T = std::remove_cvref_t<decltype(val)>;
    return max(static_cast<T>(0.01) * val, val);
}

__device__ auto tomoInv(auto val) noexcept
{
    using T = std::remove_cvref_t<decltype(val)>;
    return static_cast<T>(1) / val;
}

__global__ void tomoMap(auto *a, size_t len, auto fn_map)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        a[idx] = fn_map(a[idx]);
    }
}

cudaError_t tomoLaunchMap(auto *a, size_t len, auto fn_map, unsigned int threads_per_block, cudaStream_t stream)
{
    auto blocks_per_grid = (static_cast<unsigned int>(len) + threads_per_block - 1u) / threads_per_block;
    tomoMap<<<blocks_per_grid, threads_per_block, 0, stream>>>(a, len, fn_map);
    return cudaGetLastError();
}

__global__ void tomoBinary(auto *a, auto const *b, size_t len, auto fn_binary)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        a[idx] = fn_binary(a[idx], b[idx]);
    }
}

cudaError_t tomoLaunchBinary(auto *a, auto const *b, size_t len, auto fn_binary, unsigned int threads_per_block, cudaStream_t stream)
{
    auto blocks_per_grid = (static_cast<unsigned int>(len) + threads_per_block - 1u) / threads_per_block;
    tomoBinary<<<blocks_per_grid, threads_per_block, 0, stream>>>(a, b, len, fn_binary);
    return cudaGetLastError();
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return sin(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return sin(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return cos(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return cos(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return tan(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return tan(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return tomoRelu(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return tomoRelu(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return tomoLeakyRelu(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return tomoLeakyRelu(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return tomoInv(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return tomoInv(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) noexcept
                            { return x + y; }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) noexcept
                            { return x + y; }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) noexcept
                            { return x - y; }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) noexcept
                            { return x - y; }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMulAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) noexcept
                            { return x * y; }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMulAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) noexcept
                            { return x * y; }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) noexcept
                            { return x / y; }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchBinary(a, b, len, [] __device__(auto x, auto y) noexcept
                            { return x / y; }, threads_per_block, stream);
}

/////////////////////////
//===----------------------------------------------------------------------===//
// 1) ELU
//===----------------------------------------------------------------------===//

__device__ auto deviceElu(auto x, auto alpha) noexcept
{
    using T = std::remove_cvref_t<decltype(x)>;
    return (x > static_cast<T>(0)) ? x : alpha * (exp(x) - static_cast<T>(1));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluF(float *a, size_t len, float alpha, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(float x) noexcept
                         { return deviceElu<float>(x, alpha); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluD(double *a, size_t len, double alpha, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(double x) noexcept
                         { return deviceElu<double>(x, alpha); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 2) SELU
//    Common constants: alpha ≈ 1.6732632423543772, lambda ≈ 1.0507009873554804
//===----------------------------------------------------------------------===//

__device__ auto deviceSelu(auto x, auto alpha, auto lambda) noexcept
{
    // x>0 ? lambda*x : lambda * alpha*(exp(x)-1)
    using T = std::remove_cvref_t<decltype(x)>;
    return x > static_cast<T>(0) ? lambda * x : lambda * (alpha * (exp(x) - static_cast<T>(1)));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluF(float *a, size_t len, float alpha, float lambda, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(auto x) noexcept
                         { return deviceSelu(x, alpha, lambda); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluD(double *a, size_t len, double alpha, double lambda, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(auto x) noexcept
                         { return deviceSelu(x, alpha, lambda); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 3) Softplus
//===----------------------------------------------------------------------===//

__device__ auto deviceSoftplus(auto x) noexcept
{
    // log(1 + exp(x))
    // For numerical stability, consider clamp x or use log1p(exp(x)) if available
    using T = std::remove_cvref_t<decltype(x)>;
    return log(static_cast<T>(1) + exp(x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceSoftplus(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceSoftplus(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 4) Sigmoid
//===----------------------------------------------------------------------===//

__device__ auto deviceSigmoid(auto x)
{
    using T = std::remove_cvref_t<decltype(x)>;
    return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceSigmoid(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceSigmoid(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 5) Tanh
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return tanh(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return tanh(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 6) Swish
//    f(x) = x / (1 + exp(-x))
//===----------------------------------------------------------------------===//

__device__ auto deviceSwish(auto x)
{
    using T = std::remove_cvref_t<decltype(x)>;
    return x / (static_cast<T>(1) + exp(-x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceSwish(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceSwish(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 7) GELU (approx)
//    f(x) = 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ) )
//===----------------------------------------------------------------------===//

__device__ auto deviceGelu(auto x) noexcept
{
    using T = std::remove_cvref_t<decltype(x)>;
    // constants:
    auto constexpr k0 = static_cast<T>(0.5);
    auto constexpr k1 = static_cast<T>(0.7978845608); // sqrt(2/pi)
    auto constexpr k2 = static_cast<T>(0.044715);

    auto t = k1 * (x + k2 * x * x * x);
    return k0 * x * (static_cast<T>(1) + tanh(t));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceGelu(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceGelu(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 8) Hard Sigmoid
//    f(x) = max(0, min(1, 0.2*x + 0.5))
//===----------------------------------------------------------------------===//

__device__ auto deviceHardSigmoid(auto x) noexcept
{
    auto r = static_cast<decltype(x)>(0.2) * x + static_cast<decltype(x)>(0.5);
    r = (r < static_cast<decltype(x)>(0)) ? static_cast<decltype(x)>(0) : r;
    r = (r > static_cast<decltype(x)>(1)) ? static_cast<decltype(x)>(1) : r;
    return r;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceHardSigmoid(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceHardSigmoid(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 9) Hard Swish
//    f(x) = x * ReLU6(x + 3) / 6
//    ReLU6(z) = min(max(0,z),6)
//===----------------------------------------------------------------------===//

__device__ auto deviceHardSwish(auto x) noexcept
{
    using T = std::remove_cvref_t<decltype(x)>;
    auto z = x + static_cast<T>(3);
    // clamp z to [0,6]
    z = (z < static_cast<T>(0)) ? static_cast<T>(0) : z;
    z = (z > static_cast<T>(6)) ? static_cast<T>(6) : z;
    return x * z / static_cast<T>(6);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceHardSwish(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceHardSwish(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 10) Softsign
//    f(x) = x / (1 + |x|)
//===----------------------------------------------------------------------===//

__device__ auto deviceSoftsign(auto x) noexcept
{
    using T = std::remove_cvref_t<decltype(x)>;
    return x / (static_cast<T>(1) + abs(x));
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceSoftsign(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return deviceSoftsign(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 11) Square
//    f(x) = x^2
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return x * x; }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return x * x; }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 12) Sqrt
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return sqrt(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return sqrt(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 13) Log
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return log(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return log(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 14) Exp
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return exp(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return exp(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 15) Abs
//===----------------------------------------------------------------------===//

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return fabs(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return fabs(x); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 16) Scale
//    f(x) = scale * x
//===----------------------------------------------------------------------===//

__device__ auto deviceScale(auto x, auto scale) noexcept
{
    return scale * x;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleF(float *a, size_t len, float scale, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(auto x) noexcept
                         { return deviceScale(x, scale); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleD(double *a, size_t len, double scale, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(auto x) noexcept
                         { return deviceScale(x, scale); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 17) Softmax
//    We need two passes:
//      Pass 1: transform each x to exp(x), and compute sum of all exp(x)
//      Pass 2: divide each by the sum
//===----------------------------------------------------------------------===//

// For demonstration, we do a naive global sum across the entire array via atomic.

__global__ void kernelExpInplaceAndPartialSumF(float const *in, float *out, size_t len, float *partialSum)
{
    extern __shared__ float sdata_f[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto accum = 0.0f;

    if (i < len)
    {
        auto val = exp(in[i]);
        out[i] = val; // store exponentiated
        accum = val;
    }
    // reduction in shared memory
    sdata_f[tid] = accum;
    __syncthreads();

    // do block-level reduction
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sdata_f[tid] += sdata_f[tid + stride];
        __syncthreads();
    }

    // write result for this block to global
    if (tid == 0)
    {
        atomicAdd(partialSum, sdata_f[0]);
    }
}

__global__ void kernelExpInplaceAndPartialSumD(double const *in, double *out, size_t len, double *partialSum)
{
    extern __shared__ double sdata2[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto accum = 0.0;

    if (i < len)
    {
        auto val = exp(in[i]);
        out[i] = val; // store exponentiated
        accum = val;
    }
    // reduction in shared memory
    sdata2[tid] = accum;
    __syncthreads();

    // do block-level reduction
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
            sdata2[tid] += sdata2[tid + stride];
        __syncthreads();
    }

    // write result for this block to global
    if (tid == 0)
    {
        atomicAdd(partialSum, sdata2[0]);
    }
}

__global__ void kernelDivide(auto *out, size_t len, auto *sum)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        out[i] = out[i] / *sum;
    }
}

cudaError_t tomoSoftmax(auto *a, size_t len, unsigned int threads_per_block, cudaStream_t stream) noexcept
{
    using T = std::remove_cvref_t<decltype(*a)>;
    auto err = cudaSuccess;
    auto d_sum = static_cast<T *>(nullptr);
    err = cudaMallocAsync((void **)&d_sum, sizeof(T), stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMemsetAsync(d_sum, 0, sizeof(T), stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_sum, stream);
        return err;
    };

    unsigned int blocks_per_grid = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;
    unsigned int sharedSize = threads_per_block * sizeof(T);

    if constexpr (std::is_same_v<T, float>)
    {
        kernelExpInplaceAndPartialSumF<<<blocks_per_grid, threads_per_block, sharedSize, stream>>>(a, a, len, d_sum);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        kernelExpInplaceAndPartialSumD<<<blocks_per_grid, threads_per_block, sharedSize, stream>>>(a, a, len, d_sum);
    }
    else
    {
        static_assert(std::is_floating_point_v<T>);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_sum, stream);
        return err;
    }

    kernelDivide<<<blocks_per_grid, threads_per_block, 0, stream>>>(a, len, d_sum);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_sum, stream);
        return err;
    }

    cudaFreeAsync(d_sum, stream);

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftmaxF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoSoftmax(a, len, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftmaxD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoSoftmax(a, len, threads_per_block, stream);
}

////////

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowF(float *a, size_t len, int exponent, unsigned int threads_per_block, cudaStream_t stream)
{
    // We pass 'exponent' by value into the lambda
    return tomoLaunchMap(a, len, [=] __device__(auto x) noexcept
                         { return pow(x, exponent); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowD(double *a, size_t len, int exponent, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(auto x) noexcept
                         { return pow(x, exponent); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfF(float *a, size_t len, float exponent, unsigned int threads_per_block, cudaStream_t stream)
{
    // We pass 'exponent' by value into the lambda
    return tomoLaunchMap(a, len, [=] __device__(auto x) noexcept
                         { return pow(x, exponent); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfD(double *a, size_t len, double exponent, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(auto x) noexcept
                         { return pow(x, exponent); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampF(float *a, size_t len, float lower, float upper, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(auto x) noexcept
                         { return std::clamp(x, lower, upper); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampD(double *a, size_t len, double lower, double upper, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(auto x) noexcept
                         { return std::clamp(x, lower, upper); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return floor(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return floor(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return ceil(x); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [] __device__(auto x) noexcept
                         { return ceil(x); }, threads_per_block, stream);
}
