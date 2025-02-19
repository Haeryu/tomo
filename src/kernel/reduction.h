#include <cuda_runtime.h>

#include "tomo_dll.h"

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceF(float const *a, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceD(double const *a, size_t len, double *host_out, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanF(const float *a, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanD(double const *a, size_t len, double *host_out, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinF(float const *in, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinD(double const *in, size_t len, double *host_out, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxF(float const *in, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxD(double const *in, size_t len, double *host_out, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormF(float const *a, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormD(double const *a, size_t len, double *host_out, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormF(float const *a, size_t len, float *host_out, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormD(double const *a, size_t len, double *host_out, cudaStream_t stream);
