#include <cuda_runtime.h>

#include "tomo_dll.h"

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceF(float const *a, size_t len, float *host_out, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceD(double const *a, size_t len, double *host_out, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanF(const float *a, size_t len, float *host_out, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanD(double const *a, size_t len, double *host_out, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinF(float const *in, size_t len, float *host_out, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinD(double const *in, size_t len, double *host_out, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxF(float const *in, size_t len, float *host_out, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxD(double const *in, size_t len, double *host_out, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormF(float const *a, size_t len, float *host_out, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormD(double const *a, size_t len, double *host_out, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMinF(float const *in, size_t len, float *host_min_val, size_t *host_min_idx, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMinD(double const *in, size_t len, double *host_min_val, size_t *host_min_idx, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMaxF(float const *in, size_t len, float *host_min_val, size_t *host_min_idx, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMaxD(double const *in, size_t len, double *host_min_val, size_t *host_min_idx, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormF(float const *a, size_t len, float *host_out, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormD(double const *a, size_t len, double *host_out, unsigned int threads_per_block, cudaStream_t stream);
