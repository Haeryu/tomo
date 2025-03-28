#pragma once

#include <cuda_runtime.h>

#include "tomo_dll.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ----- GetItem Wrappers -----
// Half
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemH(
    __half_raw const *x,
    __half_raw *y,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream);

// Bfloat16
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemB(
    __nv_bfloat16_raw const *x,
    __nv_bfloat16_raw *y,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream);

// Float
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemF(
    float const *x,
    float *y,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream);

// Double
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemD(
    double const *x,
    double *y,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream);

    // Half
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSetItemH(
    __half_raw const *src,
    __half_raw *dest,
    size_t const *src_shape, size_t src_shape_len,
    size_t const *dest_shape, size_t dest_shape_len,
    size_t const *src_strides, size_t src_strides_len,
    size_t const *dest_strides, size_t dest_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t src_size,
    cudaStream_t stream);

// Bfloat16
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSetItemB(
    __nv_bfloat16_raw const *src,
    __nv_bfloat16_raw *dest,
    size_t const *src_shape, size_t src_shape_len,
    size_t const *dest_shape, size_t dest_shape_len,
    size_t const *src_strides, size_t src_strides_len,
    size_t const *dest_strides, size_t dest_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t src_size,
    cudaStream_t stream);

// Float
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSetItemF(
    float const *src,
    float *dest,
    size_t const *src_shape, size_t src_shape_len,
    size_t const *dest_shape, size_t dest_shape_len,
    size_t const *src_strides, size_t src_strides_len,
    size_t const *dest_strides, size_t dest_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t src_size,
    cudaStream_t stream);

// Double
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSetItemD(
    double const *src,
    double *dest,
    size_t const *src_shape, size_t src_shape_len,
    size_t const *dest_shape, size_t dest_shape_len,
    size_t const *src_strides, size_t src_strides_len,
    size_t const *dest_strides, size_t dest_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t src_size,
    cudaStream_t stream);

// ----- GetItemGrad Wrappers -----
// Half
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemGradH(
    __half_raw const *gy,
    __half_raw *gx,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream);

// Bfloat16
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemGradB(
    __nv_bfloat16_raw const *gy,
    __nv_bfloat16_raw *gx,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream);

// Float
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemGradF(
    float const *gy,
    float *gx,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream);

// Double
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemGradD(
    double const *gy,
    double *gx,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoOneHotH(
    size_t const *indices,
    __half_raw *one_hot,
    size_t batch_size,
    size_t num_classes,
    cudaStream_t stream);
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoOneHotB(
    size_t const *indices,
    __nv_bfloat16_raw *one_hot,
    size_t batch_size,
    size_t num_classes,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoOneHotF(
    size_t const *indices,
    float *one_hot,
    size_t batch_size,
    size_t num_classes,
    cudaStream_t stream);

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoOneHotD(
    size_t const *indices,
    double *one_hot,
    size_t batch_size,
    size_t num_classes,
    cudaStream_t stream);