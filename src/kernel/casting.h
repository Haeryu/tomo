#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tomo_dll.h"

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHtoB(__half_raw *a, size_t len, cudaStream_t stream, __nv_bfloat16_raw *out);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHtoF(__half_raw *a, size_t len, cudaStream_t stream, float *out);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHtoD(__half_raw *a, size_t len, cudaStream_t stream, double *out);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoBtoH(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream, __half_raw *out);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoBtoF(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream, float *out);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoBtoD(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream, double *out);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFtoH(float *a, size_t len, cudaStream_t stream, __half_raw *out);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFtoB(float *a, size_t len, cudaStream_t stream, __nv_bfloat16_raw *out);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFtoD(float *a, size_t len, cudaStream_t stream, double *out);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDtoH(double *a, size_t len, cudaStream_t stream, __half_raw *out);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDtoB(double *a, size_t len, cudaStream_t stream, __nv_bfloat16_raw *out);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDtoF(double *a, size_t len, cudaStream_t stream, float *out);

TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw f16ToBf16(__half_raw val);
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw f32ToBf16(float val);
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw f64ToBf16(double val);

TOMO_EXTERN_C TOMO_OPS_API __half_raw bf16Tof16(__nv_bfloat16_raw val);
TOMO_EXTERN_C TOMO_OPS_API float bf16Tof32(__nv_bfloat16_raw val);
TOMO_EXTERN_C TOMO_OPS_API double bf16Tof64(__nv_bfloat16_raw val);
