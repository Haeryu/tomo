#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"
#include "algorithm.h"

#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

cudaError_t tomoFill(auto *a, size_t len, auto val, cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*a)>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {
        thrust::fill(thrust::device.on(stream), a, a + len, val);
    }
    catch (const thrust::system_error &e)
    {
        if (e.code().category() == thrust::cuda_category())
        {
            return static_cast<cudaError_t>(e.code().value());
        }
        else
        {
            return cudaErrorUnknown;
        }
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillF(float *a, size_t len, float val, cudaStream_t stream)
{
    return tomoFill(a, len, val, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFillD(double *a, size_t len, double val, cudaStream_t stream)
{
    return tomoFill(a, len, val, stream);
}

// cudaError_t tomoGenerate(auto *a, size_t len, auto f, cudaStream_t stream)
// {
//     using T = std::remove_cvref_t<decltype(*a)>;

//     if (len == 0)
//     {
//         return cudaErrorInvalidValue;
//     }

//     try
//     {
//         thrust::generate(thrust::device.on(stream), a, a + len, f);
//     }
//     catch (const thrust::system_error &e)
//     {
//         if (e.code().category() == thrust::cuda_category())
//         {
//             return static_cast<cudaError_t>(e.code().value());
//         }
//         else
//         {
//             return cudaErrorUnknown;
//         }
//     }
//     catch (...)
//     {
//         return cudaErrorUnknown;
//     }

//     return cudaSuccess;
// }

cudaError_t tomoSortDesc(auto *a, size_t len, cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*a)>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {
        thrust::sort(thrust::device.on(stream), a, a + len, thrust::less<int>());
    }
    catch (const thrust::system_error &e)
    {
        if (e.code().category() == thrust::cuda_category())
        {
            return static_cast<cudaError_t>(e.code().value());
        }
        else
        {
            return cudaErrorUnknown;
        }
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescF(float *a, size_t len, cudaStream_t stream)
{
    return tomoSortDesc(a, len, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortDescD(double *a, size_t len, cudaStream_t stream)
{
    return tomoSortDesc(a, len, stream);
}

cudaError_t tomoSortAsc(auto *a, size_t len, cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*a)>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {
        thrust::sort(thrust::device.on(stream), a, a + len, thrust::greater<int>());
    }
    catch (const thrust::system_error &e)
    {
        if (e.code().category() == thrust::cuda_category())
        {
            return static_cast<cudaError_t>(e.code().value());
        }
        else
        {
            return cudaErrorUnknown;
        }
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscF(float *a, size_t len, cudaStream_t stream)
{
    return tomoSortAsc(a, len, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSortAscD(double *a, size_t len, cudaStream_t stream)
{
    return tomoSortAsc(a, len, stream);
}

cudaError_t tomoFind(auto *a, size_t len, auto val, cudaStream_t stream, size_t *i)
{
    using T = std::remove_cvref_t<decltype(*a)>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {
        auto it = thrust::find(thrust::device.on(stream), a, a + len, val);
        *i = std::distance(a, it);
    }
    catch (const thrust::system_error &e)
    {
        if (e.code().category() == thrust::cuda_category())
        {
            return static_cast<cudaError_t>(e.code().value());
        }
        else
        {
            return cudaErrorUnknown;
        }
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindF(float *a, size_t len, float val, cudaStream_t stream, size_t *i)
{
    return tomoFind(a, len, val, stream, i);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoFindD(double *a, size_t len, double val, cudaStream_t stream, size_t *i)
{
    return tomoFind(a, len, val, stream, i);
}
