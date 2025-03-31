#pragma once

#include <cuda_runtime.h>

#include "tomo_dll.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillH(__half_raw *a, size_t len, __half_raw val, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw val, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillF(float *a, size_t len, float val, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillD(double *a, size_t len, double val, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillUZ(size_t *a, size_t len, size_t val, cudaStream_t stream);


TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindH(__half_raw *a, size_t len, __half_raw val, cudaStream_t stream, size_t *i);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw val, cudaStream_t stream, size_t *i);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindF(float *a, size_t len, float val, cudaStream_t stream, size_t *i);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindD(double *a, size_t len, double val, cudaStream_t stream, size_t *i);
