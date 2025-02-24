#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tomo_dll.h"

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductH(__half_raw *a, __half_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductB(__nv_bfloat16_raw *a, __nv_bfloat16_raw const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductF(float *a, float const *b, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoProductD(double *a, double const *b, size_t len, cudaStream_t stream);