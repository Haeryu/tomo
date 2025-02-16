#pragma once

#ifdef TOMO_OPS_EXPORTS
#define TOMO_OPS_API __declspec(dllexport)
#else
#define TOMO_OPS_API __declspec(dllimport)
#endif

#ifdef __cplusplus
#define TOMO_EXTERN_C extern "C"
#else
#define TOMO_EXTERN_C extern
#endif

