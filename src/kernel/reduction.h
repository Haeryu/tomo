#pragma once

#include <cuda_runtime.h>

#include "tomo_dll.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceH(half const *a, size_t len, half *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceB(__nv_bfloat16 const *a, size_t len, __nv_bfloat16 *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceF(float const *a, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceD(double const *a, size_t len, double *host_out, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanH(half const *a, size_t len, half *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanB(__nv_bfloat16 const *a, size_t len, __nv_bfloat16 *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanF(float const *a, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanD(double const *a, size_t len, double *host_out, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinH(half const *in, size_t len, half *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinB(__nv_bfloat16 const *in, size_t len, __nv_bfloat16 *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinF(float const *in, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinD(double const *in, size_t len, double *host_out, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxH(half const *in, size_t len, half *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxB(__nv_bfloat16 const *in, size_t len, __nv_bfloat16 *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxF(float const *in, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxD(double const *in, size_t len, double *host_out, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormH(half const *a, size_t len, half *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormB(__nv_bfloat16 const *a, size_t len, __nv_bfloat16 *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormF(float const *a, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormD(double const *a, size_t len, double *host_out, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormH(half const *a, size_t len, half *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormB(__nv_bfloat16 const *a, size_t len, __nv_bfloat16 *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormF(float const *a, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormD(double const *a, size_t len, double *host_out, cudaStream_t stream);
