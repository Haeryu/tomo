#pragma once

#include <cuda_runtime.h>

#include "tomo_dll.h"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoBf16Add(__nv_bfloat16_raw lh, __nv_bfloat16_raw rh);
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoBf16Neg(__nv_bfloat16_raw lh);
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoBf16Sub(__nv_bfloat16_raw lh, __nv_bfloat16_raw rh);
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoBf16Mul(__nv_bfloat16_raw lh, __nv_bfloat16_raw rh);
TOMO_EXTERN_C TOMO_OPS_API __nv_bfloat16_raw tomoBf16Div(__nv_bfloat16_raw lh, __nv_bfloat16_raw rh);