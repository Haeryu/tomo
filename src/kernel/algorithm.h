#include <cuda_runtime.h>

#include "tomo_dll.h"

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillF(float *a, size_t len, float val, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillD(double *a, size_t len, double val, cudaStream_t stream);

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGenerateF(float *a, size_t len, float start, float step, cudaStream_t stream);
// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGenerateD(double *a, size_t len, double start, double step, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindF(float *a, size_t len, float val, cudaStream_t stream, size_t *i);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindD(double *a, size_t len, double val, cudaStream_t stream, size_t *i);
