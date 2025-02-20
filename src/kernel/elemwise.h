#pragma once

#include <cuda_runtime.h>

#include "tomo_dll.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSinD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCosD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoReluD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLeakyReluD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoInvD(double *a, size_t len, cudaStream_t stream);

// 1) ELU
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluH(half *a, size_t len, half alpha, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluB(__nv_bfloat16 *a, size_t len, __nv_bfloat16 alpha, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluF(float *a, size_t len, float alpha, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoEluD(double *a, size_t len, double alpha, cudaStream_t stream);

// 2) SELU
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluH(half *a, size_t len, half alpha, half lambda, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluB(__nv_bfloat16 *a, size_t len, __nv_bfloat16 alpha, __nv_bfloat16 lambda, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluF(float *a, size_t len, float alpha, float lambda, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSeluD(double *a, size_t len, double alpha, double lambda, cudaStream_t stream);

// 3) Softplus
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftplusD(double *a, size_t len, cudaStream_t stream);

// 4) Sigmoid
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSigmoidD(double *a, size_t len, cudaStream_t stream);

// 5) Tanh
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTanhD(double *a, size_t len, cudaStream_t stream);

// 6) Swish
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwishD(double *a, size_t len, cudaStream_t stream);

// 7) GELU
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGeluD(double *a, size_t len, cudaStream_t stream);

// 8) Hard Sigmoid
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSigmoidD(double *a, size_t len, cudaStream_t stream);

// 9) Hard Swish
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoHardSwishD(double *a, size_t len, cudaStream_t stream);

// 10) Softsign
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSoftsignD(double *a, size_t len, cudaStream_t stream);

// 11) Square
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSquareD(double *a, size_t len, cudaStream_t stream);

// 12) Sqrt
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSqrtD(double *a, size_t len, cudaStream_t stream);

// 13) Log
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLogD(double *a, size_t len, cudaStream_t stream);

// 14) Exp
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoExpD(double *a, size_t len, cudaStream_t stream);

// 15) Abs
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAbsD(double *a, size_t len, cudaStream_t stream);

// 16) Scale
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleH(half *a, size_t len, half scale, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleB(__nv_bfloat16 *a, size_t len, __nv_bfloat16 scale, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleF(float *a, size_t len, float scale, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleD(double *a, size_t len, double scale, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowH(half *a, size_t len, int exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowB(__nv_bfloat16 *a, size_t len, int exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowF(float *a, size_t len, int exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowD(double *a, size_t len, int exponent, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfH(half *a, size_t len, half exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfB(__nv_bfloat16 *a, size_t len, __nv_bfloat16 exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfF(float *a, size_t len, float exponent, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoPowfD(double *a, size_t len, double exponent, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampH(half *a, size_t len, half lower, half upper, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampB(__nv_bfloat16 *a, size_t len, __nv_bfloat16 lower, __nv_bfloat16 upper, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampF(float *a, size_t len, float lower, float upper, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoClampD(double *a, size_t len, double lower, double upper, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFloorD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilH(half *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilB(__nv_bfloat16 *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilF(float *a, size_t len, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCeilD(double *a, size_t len, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftH(half *a, size_t len, half offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftB(__nv_bfloat16 *a, size_t len, __nv_bfloat16 offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftF(float *a, size_t len, float offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoShiftD(double *a, size_t len, double offset, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftH(half *a, size_t len, half scale, half offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftB(__nv_bfloat16 *a, size_t len, __nv_bfloat16 scale, __nv_bfloat16 offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftF(float *a, size_t len, float scale, float offset, cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoScaleShiftD(double *a, size_t len, double scale, double offset, cudaStream_t stream);
