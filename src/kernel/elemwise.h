#include <cuda_runtime.h>

#include "tomo_dll.h"

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAddAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSubAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMulAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMulAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivAssignF(float *a, float const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoDivAssignD(double *a, double const *b, size_t len, unsigned int threads_per_block, cudaStream_t stream);


// 1) ELU
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluF(float *a, size_t len, float alpha, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluD(double *a, size_t len, double alpha, unsigned int threads_per_block, cudaStream_t stream);

// 2) SELU
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluF(float *a, size_t len, float alpha, float lambda, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluD(double *a, size_t len, double alpha, double lambda, unsigned int threads_per_block, cudaStream_t stream);

// 3) Softplus
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 4) Sigmoid
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 5) Tanh
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 6) Swish
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 7) GELU
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 8) Hard Sigmoid
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 9) Hard Swish
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 10) Softsign
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 11) Square
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 12) Sqrt
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 13) Log
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 14) Exp
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 15) Abs
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

// 16) Scale
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleF(float *a, size_t len, float scale, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleD(double *a, size_t len, double scale, unsigned int threads_per_block, cudaStream_t stream);

// 17) Softmax (requires multi-pass)
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftmaxF(float *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftmaxD(double *a, size_t len, unsigned int threads_per_block, cudaStream_t stream);

