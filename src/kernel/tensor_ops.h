#pragma once

#include <cuda_runtime.h>

#include "tomo_dll.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

////////////////////////////////////////////////////////////////////////////////
// BROADCAST-TO WRAPPERS
////////////////////////////////////////////////////////////////////////////////

/*
   Half version
   - We assume your half type is __half_raw as in your snippet.
   - If you have a different type or include <cuda_fp16.h>, adapt accordingly.
*/
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoBroadcastToH(
    __half_raw const *d_in,
    __half_raw *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

/*
   Bfloat16 version
   - We assume your bfloat16 type is __nv_bfloat16_raw as in your snippet.
*/
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoBroadcastToB(
    __nv_bfloat16_raw const *d_in,
    __nv_bfloat16_raw *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

/*
   Float version
*/
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoBroadcastToF(
    float const *d_in,
    float *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

/*
   Double version
*/
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoBroadcastToD(
    double const *d_in,
    double *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
// SUM-TO WRAPPERS
////////////////////////////////////////////////////////////////////////////////

// Half
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumToH(
    __half_raw const *d_in,
    __half_raw *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

// Bfloat16
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumToB(
    __nv_bfloat16_raw const *d_in,
    __nv_bfloat16_raw *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

// Float
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumToF(
    float const *d_in,
    float *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

// Double
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumToD(
    double const *d_in,
    double *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearH(
    __half_raw const *A, __half_raw const *B, size_t M, size_t K, size_t N, __half_raw const *bias, __half_raw *C,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearB(
    __nv_bfloat16_raw const *A, __nv_bfloat16_raw const *B, size_t M, size_t K, size_t N, __nv_bfloat16_raw const *bias, __nv_bfloat16_raw *C,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearF(
    float const *A, float const *B, size_t M, size_t K, size_t N, float const *bias, float *C,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearD(
    double const *A, double const *B, size_t M, size_t K, size_t N, double const *bias, double *C,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeH(__half_raw const *A, size_t M, size_t N, __half_raw *C, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeB(__nv_bfloat16_raw const *A, size_t M, size_t N, __nv_bfloat16_raw *C, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeF(float const *A, size_t M, size_t N, float *C, cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeD(double const *A, size_t M, size_t N, double *C, cudaStream_t stream);

// ----- Max wrappers -----

// Half
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxToH(
    __half_raw const *d_in,
    __half_raw *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

// Bfloat16
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxToB(
    __nv_bfloat16_raw const *d_in,
    __nv_bfloat16_raw *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

// Float
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxToF(
    float const *d_in,
    float *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

// Double
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxToD(
    double const *d_in,
    double *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

// ----- Min wrappers -----

// Half
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinToH(
    __half_raw const *d_in,
    __half_raw *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);
// Bfloat16
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinToB(
    __nv_bfloat16_raw const *d_in,
    __nv_bfloat16_raw *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);
// Float
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinToF(
    float const *d_in,
    float *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);
// Double
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinToD(
    double const *d_in,
    double *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size,
    size_t out_size,
    size_t nd,
    cudaStream_t stream);

// Half (FP16)
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTensordotH(
    __half_raw const *d_a,
    __half_raw const *d_b,
    __half_raw *d_out,
    size_t const *a_shape, size_t a_shape_len,
    size_t const *b_shape, size_t b_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *a_strides, size_t a_strides_len,
    size_t const *b_strides, size_t b_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *contracted_axes_a, size_t contracted_axes_a_len,
    size_t const *contracted_axes_b, size_t contracted_axes_b_len,
    size_t a_nd, size_t b_nd, size_t out_nd,
    size_t out_size, size_t num_contracted,
    cudaStream_t stream);

// Bfloat16
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTensordotB(
    __nv_bfloat16_raw const *d_a,
    __nv_bfloat16_raw const *d_b,
    __nv_bfloat16_raw *d_out,
    size_t const *a_shape, size_t a_shape_len,
    size_t const *b_shape, size_t b_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *a_strides, size_t a_strides_len,
    size_t const *b_strides, size_t b_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *contracted_axes_a, size_t contracted_axes_a_len,
    size_t const *contracted_axes_b, size_t contracted_axes_b_len,
    size_t a_nd, size_t b_nd, size_t out_nd,
    size_t out_size, size_t num_contracted,
    cudaStream_t stream);

// Float
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTensordotF(
    float const *d_a,
    float const *d_b,
    float *d_out,
    size_t const *a_shape, size_t a_shape_len,
    size_t const *b_shape, size_t b_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *a_strides, size_t a_strides_len,
    size_t const *b_strides, size_t b_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *contracted_axes_a, size_t contracted_axes_a_len,
    size_t const *contracted_axes_b, size_t contracted_axes_b_len,
    size_t a_nd, size_t b_nd, size_t out_nd,
    size_t out_size, size_t num_contracted,
    cudaStream_t stream);

// Double
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTensordotD(
    double const *d_a,
    double const *d_b,
    double *d_out,
    size_t const *a_shape, size_t a_shape_len,
    size_t const *b_shape, size_t b_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *a_strides, size_t a_strides_len,
    size_t const *b_strides, size_t b_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *contracted_axes_a, size_t contracted_axes_a_len,
    size_t const *contracted_axes_b, size_t contracted_axes_b_len,
    size_t a_nd, size_t b_nd, size_t out_nd,
    size_t out_size, size_t num_contracted,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeExH(
    __half_raw const *d_in, __half_raw *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *out_shape, size_t const out_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const *out_strides, size_t const out_strides_len,
    size_t const *perm, size_t const perm_len,
    size_t const nd,
    size_t const in_size, size_t const out_size,
    cudaStream_t const stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeExB(
    __nv_bfloat16_raw const *d_in, __nv_bfloat16_raw *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *out_shape, size_t const out_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const *out_strides, size_t const out_strides_len,
    size_t const *perm, size_t const perm_len,
    size_t const nd,
    size_t const in_size, size_t const out_size,
    cudaStream_t const stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeExF(
    float const *d_in, float *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *out_shape, size_t const out_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const *out_strides, size_t const out_strides_len,
    size_t const *perm, size_t const perm_len,
    size_t const nd,
    size_t const in_size, size_t const out_size,
    cudaStream_t const stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeExD(
    double const *d_in, double *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *out_shape, size_t const out_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const *out_strides, size_t const out_strides_len,
    size_t const *perm, size_t const perm_len,
    size_t const nd,
    size_t const in_size, size_t const out_size,
    cudaStream_t const stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoRollaxisH(
    __half_raw const *d_in, __half_raw *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const axis, size_t const start,
    size_t const nd, size_t const in_size, size_t const out_size,
    cudaStream_t const stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoRollaxisB(
    __nv_bfloat16_raw const *d_in, __nv_bfloat16_raw *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const axis, size_t const start,
    size_t const nd, size_t const in_size, size_t const out_size,
    cudaStream_t const stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoRollaxisF(
    float const *d_in, float *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const axis, size_t const start,
    size_t const nd, size_t const in_size, size_t const out_size,
    cudaStream_t const stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoRollaxisD(
    double const *d_in, double *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const axis, size_t const start,
    size_t const nd, size_t const in_size, size_t const out_size,
    cudaStream_t const stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwapaxesH(
    __half_raw const *d_in, __half_raw *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t axis1, size_t axis2,
    size_t nd, size_t in_size, size_t out_size,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwapaxesB(
    __nv_bfloat16_raw const *d_in, __nv_bfloat16_raw *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t axis1, size_t axis2,
    size_t nd, size_t in_size, size_t out_size,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwapaxesF(
    float const *d_in, float *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t axis1, size_t axis2,
    size_t nd, size_t in_size, size_t out_size,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwapaxesD(
    double const *d_in, double *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t axis1, size_t axis2,
    size_t nd, size_t in_size, size_t out_size,
    cudaStream_t stream);

// __half version for im2col
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2colH(
    __half_raw const *d_img, __half_raw *d_col,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dy, size_t dx,
    cudaStream_t stream);

// __nv_bfloat16 version for im2col
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2colB(
    __nv_bfloat16_raw const *d_img, __nv_bfloat16_raw *d_col,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dy, size_t dx,
    cudaStream_t stream);

// float version for im2col
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2colF(
    float const *d_img, float *d_col,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dy, size_t dx,
    cudaStream_t stream);

// double version for im2col
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2colD(
    double const *d_img, double *d_col,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dy, size_t dx,
    cudaStream_t stream);

// __half version for col2im
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2imH(
    __half_raw const *d_col, __half_raw *d_img,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dx, size_t dy,
    cudaStream_t stream);
// __nv_bfloat16 version for col2im
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2imB(
    __nv_bfloat16_raw const *d_col, __nv_bfloat16_raw *d_img,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dx, size_t dy,
    cudaStream_t stream);

// float version for col2im
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2imF(
    float const *d_col, float *d_img,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dx, size_t dy,
    cudaStream_t stream);
// double version for col2im
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2imD(
    double const *d_col, double *d_img,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dx, size_t dy,
    cudaStream_t stream);