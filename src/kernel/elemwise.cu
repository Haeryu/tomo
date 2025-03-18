#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"

#include "elemwise.h"

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <numbers>

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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddH(__half_raw *a, __half_raw const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::plus<__half_raw>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::plus<__nv_bfloat16_raw>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddF(float *a, float const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::plus<float>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddD(double *a, double const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::plus<double>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubH(__half_raw *a, __half_raw const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::minus<__half_raw>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::minus<__nv_bfloat16_raw>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubF(float *a, float const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::minus<float>(), stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubD(double *a, double const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, thrust::minus<double>(), stream);
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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualH(__half_raw *a, __half_raw const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, [] __device__(__half_raw a, __half_raw b) -> __half_raw
                        { return (__half_raw) !!(a == b); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, [] __device__(__nv_bfloat16_raw a, __nv_bfloat16_raw b) -> __nv_bfloat16_raw
                        { return (__nv_bfloat16_raw) !!(a == b); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualF(float *a, float const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, [] __device__(float a, float b) -> float
                        { return (float)!!(a == b); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualD(double *a, double const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, [] __device__(double a, double b) -> double
                        { return (double)!!(a == b); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualUz(size_t *a, size_t const *b, size_t len, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, [] __device__(size_t a, size_t b) -> size_t
                        { return (size_t)!!(a == b); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualApproxH(__half_raw *a, __half_raw const *b, size_t len, __half_raw eps, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, [=] __device__(__half_raw a, __half_raw b) -> __half_raw
                        { return (__half_raw) !!((a - b) < eps); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualApproxB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, __nv_bfloat16_raw eps, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, [=] __device__(__nv_bfloat16_raw a, __nv_bfloat16_raw b) -> __nv_bfloat16_raw
                        { return (__nv_bfloat16_raw) !!((a - b) < eps); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualApproxF(float *a, float const *b, size_t len, float eps, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, [=] __device__(float a, float b) -> float
                        { return (float)!!((a - b) < eps); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualApproxD(double *a, double const *b, size_t len, double eps, cudaStream_t stream)
{
    return tomoElemwise(a, b, len, [=] __device__(double a, double b) -> double
                        { return (double)!!((a - b) < eps); }, stream);
}
//---------------------------------------------------------------------
// ReLU Backward
// Computes: grad = grad * (x > 0 ? 1 : 0)
//---------------------------------------------------------------------

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluBackwardH(
    const __half_raw *x, __half_raw *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(__half_raw grad_val, __half_raw x_val) -> __half_raw
                        {
            float xf = __half2float(x_val);
            float gradf = __half2float(grad_val);
            float result = (xf > 0.0f) ? gradf : 0.0f;
            return __float2half(result); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluBackwardB(
    const __nv_bfloat16_raw *x, __nv_bfloat16_raw *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(__nv_bfloat16_raw grad_val, __nv_bfloat16_raw x_val) -> __nv_bfloat16_raw
                        {
            float xf = __bfloat162float(x_val);
            float gradf = __bfloat162float(grad_val);
            float result = (xf > 0.0f) ? gradf : 0.0f;
            return __float2bfloat16(result); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluBackwardF(
    const float *x, float *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(float grad_val, float x_val) -> float
                        { return grad_val * (x_val > 0.0f ? 1.0f : 0.0f); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluBackwardD(
    const double *x, double *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(double grad_val, double x_val) -> double
                        { return grad_val * (x_val > 0.0 ? 1.0 : 0.0); }, stream);
}

//---------------------------------------------------------------------
// Leaky ReLU Backward
// Computes: grad = grad * (x > 0 ? 1 : slope)
// Here we use slope = 0.01
//---------------------------------------------------------------------

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluBackwardH(
    const __half_raw *x, __half_raw *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(__half_raw grad_val, __half_raw x_val) -> __half_raw
                        {
            float xf = __half2float(x_val);
            float gradf = __half2float(grad_val);
            float slope = 0.01f;
            float result = gradf * (xf > 0.0f ? 1.0f : slope);
            return __float2half(result); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluBackwardB(
    const __nv_bfloat16_raw *x, __nv_bfloat16_raw *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(__nv_bfloat16_raw grad_val, __nv_bfloat16_raw x_val) -> __nv_bfloat16_raw
                        {
            float xf = __bfloat162float(x_val);
            float gradf = __bfloat162float(grad_val);
            float slope = 0.01f;
            float result = gradf * (xf > 0.0f ? 1.0f : slope);
            return __float2bfloat16(result); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluBackwardF(
    const float *x, float *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(float grad_val, float x_val) -> float
                        {
            float slope = 0.01f;
            return grad_val * (x_val > 0.0f ? 1.0f : slope); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluBackwardD(
    const double *x, double *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(double grad_val, double x_val) -> double
                        {
            double slope = 0.01;
            return grad_val * (x_val > 0.0 ? 1.0 : slope); }, stream);
}

//---------------------------------------------------------------------
// GELU Backward
// Approximate GELU derivative is computed as:
//   gelu_deriv = 0.5*(1 + tanh(z)) + 0.5*x*(1 - tanh(z)^2)*kAlpha*(1 + 0.134145*x^2)
// with z = kAlpha*(x + 0.044715*x^3) and kAlpha = sqrt(2/pi)
// Then: grad = grad * gelu_deriv
//---------------------------------------------------------------------

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluBackwardH(
    const __half_raw *x, __half_raw *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(__half_raw grad_val, __half_raw x_val) -> __half_raw
                        {
            float xf = __half2float(x_val);
            float gradf = __half2float(grad_val);
            float kAlpha = sqrtf(2.0f / std::numbers::pi);
            float z = kAlpha * (xf + 0.044715f * xf * xf * xf);
            float tanh_z = tanhf(z);
            float gelu_deriv = 0.5f * (1.0f + tanh_z) +
                               0.5f * xf * (1.0f - tanh_z * tanh_z) * kAlpha * (1.0f + 0.134145f * xf * xf);
            return __float2half(gradf * gelu_deriv); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluBackwardB(
    const __nv_bfloat16_raw *x, __nv_bfloat16_raw *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(__nv_bfloat16_raw grad_val, __nv_bfloat16_raw x_val) -> __nv_bfloat16_raw
                        {
            float xf = __bfloat162float(x_val);
            float gradf = __bfloat162float(grad_val);
            float kAlpha = sqrtf(2.0f / std::numbers::pi);
            float z = kAlpha * (xf + 0.044715f * xf * xf * xf);
            float tanh_z = tanhf(z);
            float gelu_deriv = 0.5f * (1.0f + tanh_z) +
                               0.5f * xf * (1.0f - tanh_z * tanh_z) * kAlpha * (1.0f + 0.134145f * xf * xf);
            return __float2bfloat16(gradf * gelu_deriv); }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluBackwardF(
    const float *x, float *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(float grad_val, float x_val) -> float
                        {
            float kAlpha = sqrtf(2.0f / std::numbers::pi);
            float z = kAlpha * (x_val + 0.044715f * x_val * x_val * x_val);
            float tanh_z = tanhf(z);
            float gelu_deriv = 0.5f * (1.0f + tanh_z) +
                               0.5f * x_val * (1.0f - tanh_z * tanh_z) * kAlpha * (1.0f + 0.134145f * x_val * x_val);
            return grad_val * gelu_deriv; }, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluBackwardD(
    const double *x, double *grad,
    size_t len, cudaStream_t stream)
{
    return tomoElemwise(grad, x, len, [] __device__(double grad_val, double x_val) -> double
                        {
            double kAlpha = sqrt(2.0 / std::numbers::pi);
            double z = kAlpha * (x_val + 0.044715 * x_val * x_val * x_val);
            double tanh_z = tanh(z);
            double gelu_deriv = 0.5 * (1.0 + tanh_z) +
                                  0.5 * x_val * (1.0 - tanh_z * tanh_z) * kAlpha * (1.0 + 0.134145 * x_val * x_val);
            return grad_val * gelu_deriv; }, stream);
}
