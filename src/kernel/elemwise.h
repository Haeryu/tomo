#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tomo_dll.h"

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddH(__half_raw *a, __half_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddF(float *a, float const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddD(double *a, double const *b, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubH(__half_raw *a, __half_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubF(float *a, float const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubD(double *a, double const *b, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductH(__half_raw *a, __half_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductF(float *a, float const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductD(double *a, double const *b, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivideH(__half_raw *a, __half_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivideB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivideF(float *a, float const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivideD(double *a, double const *b, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualH(__half_raw *a, __half_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualF(float *a, float const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualD(double *a, double const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualUz(size_t *a, size_t const *b, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualApproxH(__half_raw *a, __half_raw const *b, size_t len, __half_raw eps, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualApproxB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, __nv_bfloat16_raw eps, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualApproxF(float *a, float const *b, size_t len, float eps, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqualApproxD(double *a, double const *b, size_t len, double eps, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluBackwardH(
    const __half_raw *x, __half_raw *grad,
    size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluBackwardB(
    const __nv_bfloat16_raw *x, __nv_bfloat16_raw *grad,
    size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluBackwardF(
    const float *x, float *grad,
    size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluBackwardD(
    const double *x, double *grad,
    size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluBackwardH(
    const __half_raw *x, __half_raw *grad,
    size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluBackwardB(
    const __nv_bfloat16_raw *x, __nv_bfloat16_raw *grad,
    size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluBackwardF(
    const float *x, float *grad,
    size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluBackwardD(
    const double *x, double *grad,
    size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluBackwardH(
    const __half_raw *x, __half_raw *grad,
    size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluBackwardB(
    const __nv_bfloat16_raw *x, __nv_bfloat16_raw *grad,
    size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluBackwardF(
    const float *x, float *grad,
    size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluBackwardD(
    const double *x, double *grad,
    size_t len, cudaStream_t stream);