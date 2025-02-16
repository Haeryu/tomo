#include <cuda_runtime.h>

#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"
#include "elemwise.h"
// #include <cuda_fp16.h>

// to support half in future(...maybe long) easily i removed concept

__device__ auto tomoRelu(auto val) noexcept
{

    return max(static_cast<decltype(val)>(0), val);
}

__device__ auto tomoLeakyRelu(auto val) noexcept
{
    return max(static_cast<decltype(val)>(0.01) * val, val);
}

__device__ auto tomoInv(auto val) noexcept
{
    return static_cast<decltype(val)>(1) / val;
}

__global__ void tomoMap(auto *a, size_t len, auto fn_map)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        a[idx] = fn_map(a[idx]);
    }
}

cudaError_t tomoLaunchMap(auto *a, size_t len, auto fn_map, unsigned int threads_per_block, cudaStream_t stream)
{
    unsigned int blocks_per_grid = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;
    tomoMap<<<blocks_per_grid, threads_per_block, 0, stream>>>(a, len, fn_map);
    return cudaGetLastError();
}

__global__ void tomoBinary(auto *a, auto const *b, size_t len, auto fn_binary)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        a[idx] = fn_binary(a[idx], b[idx]);
    }
}

cudaError_t tomoLaunchBinary(auto *a, auto const *b, size_t len, auto fn_binary, unsigned int threads_per_block, cudaStream_t stream)
{
    unsigned int blocks_per_grid = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;
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

__device__ auto deviceElu(std::floating_point auto x, std::floating_point auto alpha) noexcept
{
    return (x > static_cast<decltype(x)>(0)) ? x : alpha * (exp(x) - static_cast<decltype(x)>(1));
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

__device__ auto deviceSelu(std::floating_point auto x, std::floating_point auto alpha, std::floating_point auto lambda) noexcept
{
    // x>0 ? lambda*x : lambda * alpha*(exp(x)-1)
    if (x > static_cast<decltype(x)>(0))
    {
        return lambda * x;
    }
    else
    {
        return lambda * (alpha * (exp(x) - static_cast<decltype(x)>(1)));
    }
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluF(float *a, size_t len, float alpha, float lambda, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(float x) noexcept
                         { return deviceSelu<float>(x, alpha, lambda); }, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluD(double *a, size_t len, double alpha, double lambda, unsigned int threads_per_block, cudaStream_t stream)
{
    return tomoLaunchMap(a, len, [=] __device__(double x) noexcept
                         { return deviceSelu<double>(x, alpha, lambda); }, threads_per_block, stream);
}

//===----------------------------------------------------------------------===//
// 3) Softplus
//===----------------------------------------------------------------------===//

__device__ auto deviceSoftplus(std::floating_point auto x) noexcept
{
    // log(1 + exp(x))
    // For numerical stability, consider clamp x or use log1p(exp(x)) if available
    return log(static_cast<decltype(x)>(1) + exp(x));
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

__device__ auto deviceSigmoid(std::floating_point auto x)
{
    return static_cast<decltype(x)>(1) / (static_cast<decltype(x)>(1) + exp(-x));
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
                         { return tanhf(x); }, threads_per_block, stream);
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

__device__ auto deviceSwish(std::floating_point auto x)
{
    return x / (static_cast<decltype(x)>(1) + exp(-x));
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

__device__ auto deviceGelu(std::floating_point auto x) noexcept
{
    // constants:
    const auto k0 = static_cast<decltype(x)>(0.5);
    const auto k1 = static_cast<decltype(x)>(0.7978845608); // sqrt(2/pi)
    const auto k2 = static_cast<decltype(x)>(0.044715);

    auto t = k1 * (x + k2 * x * x * x);
    return k0 * x * (static_cast<decltype(x)>(1) + tanh(t));
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

__device__ auto deviceHardSigmoid(std::floating_point auto x) noexcept
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

__device__ auto deviceHardSwish(std::floating_point auto x) noexcept
{
    auto z = x + static_cast<decltype(x)>(3);
    // clamp z to [0,6]
    z = (z < static_cast<decltype(x)>(0)) ? static_cast<decltype(x)>(0) : z;
    z = (z > static_cast<decltype(x)>(6)) ? static_cast<decltype(x)>(6) : z;
    return x * z / static_cast<decltype(x)>(6);
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

__device__ auto deviceSoftsign(std::floating_point auto x) noexcept
{
    return x / (static_cast<decltype(x)>(1) + fabs(x));
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
                         { return sqrtf(x); }, threads_per_block, stream);
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
                         { return logf(x); }, threads_per_block, stream);
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
                         { return expf(x); }, threads_per_block, stream);
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
                         { return fabsf(x); }, threads_per_block, stream);
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

__device__ auto deviceScale(std::floating_point auto x, std::floating_point auto scale) noexcept
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

__global__ void kernelExpInplaceAndPartialSum(std::floating_point auto const *in, std::floating_point auto *out, size_t len, std::floating_point auto *partialSum)
{
    extern __shared__ float sdata2[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    auto accum = static_cast<decltype(*in)>(0.0);

    if (i < len)
    {
        float val = exp(in[i]);
        out[i] = val; // store exponentiated
        accum = val;
    }
    // reduction in shared memory
    sdata2[tid] = accum;
    __syncthreads();

    // do block-level reduction
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
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
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        out[i] = out[i] / *sum;
    }
}

cudaError_t tomoSoftmax(auto *a, size_t len, unsigned int threads_per_block, cudaStream_t stream)
{
    auto err = cudaSuccess;
    auto d_sum = static_cast<decltype(a)>(nullptr);
    err = cudaMallocAsync((void **)&d_sum, sizeof(decltype(*a)), stream);
    if (err != cudaSuccess)
    {
        return err; // can't even allocate
    }
    err = cudaMemsetAsync(d_sum, 0, sizeof(decltype(*a)), stream);
    if (err != cudaSuccess)
    {
        // We must free before returning
        cudaFreeAsync(d_sum, stream);
        return err;
    };

    unsigned int blocks_per_grid = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;
    unsigned int sharedSize = threads_per_block * sizeof(decltype(*a));

    kernelExpInplaceAndPartialSum<<<blocks_per_grid, threads_per_block, sharedSize, stream>>>(a, a, len, d_sum);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // Free before returning
        cudaFreeAsync(d_sum, stream);
        return err;
    }

    kernelDivide<<<blocks_per_grid, threads_per_block, 0, stream>>>(a, len, d_sum);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // If we failed to launch or run kernel, free memory
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
