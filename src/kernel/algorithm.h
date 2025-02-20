#pragma once

#include <cuda_runtime.h>

#include "tomo_dll.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillH(half *a, size_t len, half val, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillB(__nv_bfloat16 *a, size_t len, __nv_bfloat16 val, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillF(float *a, size_t len, float val, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillD(double *a, size_t len, double val, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindH(half *a, size_t len, half val, cudaStream_t stream, size_t *i);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindB(__nv_bfloat16 *a, size_t len, __nv_bfloat16 val, cudaStream_t stream, size_t *i);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindF(float *a, size_t len, float val, cudaStream_t stream, size_t *i);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindD(double *a, size_t len, double val, cudaStream_t stream, size_t *i);
