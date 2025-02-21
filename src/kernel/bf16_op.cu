#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"
#include "bf16_op.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "float_op.cuh"

TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoBf16Add(__nv_bfloat16_raw lh, __nv_bfloat16_raw rh)
{
    return __nv_bfloat16{lh} + __nv_bfloat16{rh};
}
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoBf16Neg(__nv_bfloat16_raw lh)
{
    return -__nv_bfloat16{lh};
}
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoBf16Sub(__nv_bfloat16_raw lh, __nv_bfloat16_raw rh)
{
    return __nv_bfloat16{lh} - __nv_bfloat16{rh};
}
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoBf16Mul(__nv_bfloat16_raw lh, __nv_bfloat16_raw rh)
{
    return __nv_bfloat16{lh} * __nv_bfloat16{rh};
}
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoBf16Div(__nv_bfloat16_raw lh, __nv_bfloat16_raw rh)
{
    return __nv_bfloat16{lh} / __nv_bfloat16{rh};
}