#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "tomo_dll.h"

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvD(double *a, size_t len, cudaStream_t stream);

// 1) ELU
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluH(__half_raw *a, size_t len, __half_raw alpha, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw alpha, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluF(float *a, size_t len, float alpha, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluD(double *a, size_t len, double alpha, cudaStream_t stream);

// 2) SELU
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluH(__half_raw *a, size_t len, __half_raw alpha, __half_raw lambda, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw alpha, __nv_bfloat16_raw lambda, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluF(float *a, size_t len, float alpha, float lambda, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluD(double *a, size_t len, double alpha, double lambda, cudaStream_t stream);

// 3) Softplus
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusD(double *a, size_t len, cudaStream_t stream);

// 4) Sigmoid
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidD(double *a, size_t len, cudaStream_t stream);

// 5) Tanh
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhD(double *a, size_t len, cudaStream_t stream);

// 6) Swish
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishD(double *a, size_t len, cudaStream_t stream);

// 7) GELU
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluD(double *a, size_t len, cudaStream_t stream);

// 8) Hard Sigmoid
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidD(double *a, size_t len, cudaStream_t stream);

// 9) Hard Swish
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishD(double *a, size_t len, cudaStream_t stream);

// 10) Softsign
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignD(double *a, size_t len, cudaStream_t stream);

// 11) Square
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareD(double *a, size_t len, cudaStream_t stream);

// 12) Sqrt
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtD(double *a, size_t len, cudaStream_t stream);

// 13) Log
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogD(double *a, size_t len, cudaStream_t stream);

// 14) Exp
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpD(double *a, size_t len, cudaStream_t stream);

// 15) Abs
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsD(double *a, size_t len, cudaStream_t stream);

// 16) Scale
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleH(__half_raw *a, size_t len, __half_raw scale, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw scale, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleF(float *a, size_t len, float scale, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleD(double *a, size_t len, double scale, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowH(__half_raw *a, size_t len, int exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowB(__nv_bfloat16_raw *a, size_t len, int exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowF(float *a, size_t len, int exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowD(double *a, size_t len, int exponent, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfH(__half_raw *a, size_t len, __half_raw exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfF(float *a, size_t len, float exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfD(double *a, size_t len, double exponent, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampH(__half_raw *a, size_t len, __half_raw lower, __half_raw upper, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw lower, __nv_bfloat16_raw upper, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampF(float *a, size_t len, float lower, float upper, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampD(double *a, size_t len, double lower, double upper, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilH(__half_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilB(__nv_bfloat16_raw *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftH(__half_raw *a, size_t len, __half_raw offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftF(float *a, size_t len, float offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftD(double *a, size_t len, double offset, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftH(__half_raw *a, size_t len, __half_raw scale, __half_raw offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw scale, __nv_bfloat16_raw offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftF(float *a, size_t len, float scale, float offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftD(double *a, size_t len, double scale, double offset, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtH(__half_raw *a, size_t len, __half_raw num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtF(float *a, size_t len, float num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtD(double *a, size_t len, double num, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtEqH(__half_raw *a, size_t len, __half_raw num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtEqB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtEqF(float *a, size_t len, float num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGtEqD(double *a, size_t len, double num, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtH(__half_raw *a, size_t len, __half_raw num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtF(float *a, size_t len, float num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtD(double *a, size_t len, double num, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtEqH(__half_raw *a, size_t len, __half_raw num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtEqB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtEqF(float *a, size_t len, float num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLtEqD(double *a, size_t len, double num, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqH(__half_raw *a, size_t len, __half_raw num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqB(__nv_bfloat16_raw *a, size_t len, __nv_bfloat16_raw num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqF(float *a, size_t len, float num, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEqD(double *a, size_t len, double num, cudaStream_t stream);