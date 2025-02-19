#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"
#include "reduction.h"

// #include <cmath>
#include <cfloat> // for FLT_MAX, etc.
#include <math_constants.h>
#include <limits>

#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

cudaError_t tomoSumReduce(auto const *a,
                          size_t len,
                          auto *host_out,
                          unsigned int threads_per_block,
                          cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*a)>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {

        *host_out = thrust::reduce(thrust::device.on(stream), a, a + len, static_cast<T>(0), thrust::plus<T>{});
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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceF(float const *a,
                                                      size_t len,
                                                      float *host_out,
                                                      unsigned int threads_per_block,
                                                      cudaStream_t stream)
{
    return tomoSumReduce(a, len, host_out, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceD(double const *a,
                                                      size_t len,
                                                      double *host_out,
                                                      unsigned int threads_per_block,
                                                      cudaStream_t stream)
{
    return tomoSumReduce(a, len, host_out, threads_per_block, stream);
}

cudaError_t tomoMean(auto const *a,
                     size_t len,
                     auto *host_out,
                     unsigned int threads_per_block,
                     cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*a)>;
    // 1) first do tomoSumF
    auto err = tomoSumReduce(a, len, host_out, threads_per_block, stream);

    if (err != cudaSuccess)
    {
        return err;
    }

    // 2) hostOut now has the sum (assuming we did a stream sync).
    // divide by len on CPU side:
    if (len > 0)
    {
        *host_out /= static_cast<T>(len);
    }

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanF(const float *a,
                                                 size_t len,
                                                 float *host_out,
                                                 unsigned int threads_per_block,
                                                 cudaStream_t stream)
{
    return tomoMean(a, len, host_out, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMeanD(double const *a,
                                                 size_t len,
                                                 double *host_out,
                                                 unsigned int threads_per_block,
                                                 cudaStream_t stream)
{
    return tomoMean(a, len, host_out, threads_per_block, stream);
}

cudaError_t tomoMin(const auto *in,
                    size_t len,
                    auto *host_out,
                    unsigned int threads_per_block,
                    cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*in)>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {
        *host_out = thrust::reduce(thrust::device.on(stream), in, in + len, std::numeric_limits<T>::max(), thrust::minimum<T>{});
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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinF(float const *in,
                                                size_t len,
                                                float *host_out,
                                                unsigned int threads_per_block,
                                                cudaStream_t stream)
{
    return tomoMin(in, len, host_out, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinD(double const *in,
                                                size_t len,
                                                double *host_out,
                                                unsigned int threads_per_block,
                                                cudaStream_t stream)
{
    return tomoMin(in, len, host_out, threads_per_block, stream);
}

cudaError_t tomoMax(auto const *in,
                    size_t len,
                    auto *host_out,
                    unsigned int threads_per_block,
                    cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*in)>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {
        *host_out = thrust::reduce(thrust::device.on(stream), in, in + len, std::numeric_limits<T>::lowest(), thrust::maximum<T>{});
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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxF(float const *in,
                                                size_t len,
                                                float *host_out, // final result on host
                                                unsigned int threads_per_block,
                                                cudaStream_t stream)
{
    return tomoMax(in, len, host_out, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxD(double const *in,
                                                size_t len,
                                                double *host_out, // final result on host
                                                unsigned int threads_per_block,
                                                cudaStream_t stream)
{
    return tomoMax(in, len, host_out, threads_per_block, stream);
}

// Then a second pass that does the same approach on blockResults...
// or do it on the CPU if #blocks is small enough.

template <typename T>
cudaError_t tomoL2Norm(T const *a,
                       size_t len,
                       T *host_out, // where we store final L2 norm on the host
                       unsigned int threads_per_block,
                       cudaStream_t stream)
{

    // using T = std::remove_cvref_t<decltype(*a)>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {
        *host_out = thrust::reduce(thrust::device.on(stream), a, a + len, static_cast<T>(0), [] __host__ __device__(T const &lhs, T const &rhs)
                                   { return lhs + rhs * rhs; });
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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormF(float const *a,
                                                   size_t len,
                                                   float *host_out, // where we store final L2 norm on the host
                                                   unsigned int threads_per_block,
                                                   cudaStream_t stream)
{
    return tomoL2Norm(a, len, host_out, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL2NormD(double const *a,
                                                   size_t len,
                                                   double *host_out,
                                                   unsigned int threads_per_block,
                                                   cudaStream_t stream)
{
    return tomoL2Norm(a, len, host_out, threads_per_block, stream);
}

template <typename T>
cudaError_t tomoL1Norm(T const *a,
                       size_t len,
                       T *host_out,
                       unsigned int threads_per_block,
                       cudaStream_t stream)
{
    // using T = std::remove_cvref_t<decltype(*a)>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    try
    {
        *host_out = thrust::reduce(thrust::device.on(stream), a, a + len, static_cast<T>(0), [] __host__ __device__(const T &lhs, const T &rhs)
                                   { return lhs + abs(rhs); });
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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormF(float const *a,
                                                   size_t len,
                                                   float *host_out,
                                                   unsigned int threads_per_block,
                                                   cudaStream_t stream)
{
    return tomoL1Norm(a, len, host_out, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoL1NormD(double const *a,
                                                   size_t len,
                                                   double *host_out,
                                                   unsigned int threads_per_block,
                                                   cudaStream_t stream)
{
    return tomoL1Norm(a, len, host_out, threads_per_block, stream);
}

// cudaError_t tomoArgMin(auto const *in,
//                        size_t len,
//                        auto *host_min_val,   // where we store the final min value
//                        size_t *host_min_idx, // where we store the final index
//                        unsigned int threads_per_block,
//                        cudaStream_t stream)
// {
//     using T = std::remove_cvref_t<decltype(*in)>;
//     using ValIndexT = ValIndex<T>;

//     if (len == 0)
//     {
//         return cudaErrorInvalidValue;
//     }

//     // 1) #blocks for first pass
//     auto blocks = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;

//     // 2) block-level results array
//     auto block_results = static_cast<ValIndexT *>(nullptr);
//     auto err = cudaMallocAsync((void **)&block_results, blocks * sizeof(ValIndexT), stream);
//     if (err != cudaSuccess)
//     {
//         return err;
//     }

//     // 3) kernelBlockArgMinF
//     auto shared_size = threads_per_block * sizeof(ValIndexT);

//     if constexpr (std::is_same_v<T, float>)
//     {
//         kernelBlockArgMinF<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
//     }
//     else if constexpr (std::is_same_v<T, double>)
//     {
//         kernelBlockArgMinD<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
//     }
//     else
//     {
//         static_assert(std::is_floating_point_v<T>);
//     }

//     err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(block_results, stream);
//         return err;
//     }

//     // 4) single-block final reduce
//     auto d_final = static_cast<ValIndexT *>(nullptr);
//     err = cudaMallocAsync((void **)&d_final, sizeof(ValIndexT), stream);
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(block_results, stream);
//         return err;
//     }

//     // pick #threads in second pass
//     auto threads2 = (blocks < threads_per_block) ? blocks : threads_per_block;
//     auto sharedSize2 = threads2 * sizeof(ValIndexT);

//     if constexpr (std::is_same_v<T, float>)
//     {
//         kernelSingleBlockArgMinF<<<1, threads2, sharedSize2, stream>>>(block_results, blocks, d_final);
//     }
//     else if constexpr (std::is_same_v<T, double>)
//     {
//         kernelSingleBlockArgMinD<<<1, threads2, sharedSize2, stream>>>(block_results, blocks, d_final);
//     }
//     else
//     {
//         static_assert(std::is_floating_point_v<T>);
//     }

//     err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(d_final, stream);
//         cudaFreeAsync(block_results, stream);
//         return err;
//     }

//     // 5) copy final result
//     ValIndexT h_final;
//     err = cudaMemcpyAsync(&h_final, d_final, sizeof(ValIndexT), cudaMemcpyDeviceToHost, stream);
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(d_final, stream);
//         cudaFreeAsync(block_results, stream);
//         return err;
//     }

//     // sync if we need result now
//     err = cudaStreamSynchronize(stream);
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(d_final, stream);
//         cudaFreeAsync(block_results, stream);
//         return err;
//     }

//     // 6) store to host
//     *host_min_val = h_final.val;
//     *host_min_idx = h_final.idx;

//     // 7) cleanup
//     cudaFreeAsync(d_final, stream);
//     cudaFreeAsync(block_results, stream);

//     return cudaSuccess;
// }

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMinF(float const *in,
//                                                    size_t len,
//                                                    float *host_min_val,  // where we store the final min value
//                                                    size_t *host_min_idx, // where we store the final index
//                                                    unsigned int threads_per_block,
//                                                    cudaStream_t stream)

// {
//     return tomoArgMin(in, len, host_min_val, host_min_idx, threads_per_block, stream);
// }

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMinD(double const *in,
//                                                    size_t len,
//                                                    double *host_min_val, // where we store the final min value
//                                                    size_t *host_min_idx, // where we store the final index
//                                                    unsigned int threads_per_block,
//                                                    cudaStream_t stream)

// {
//     return tomoArgMin(in, len, host_min_val, host_min_idx, threads_per_block, stream);
// }

// __global__ void kernelBlockArgMaxF(float const *in, size_t len, ValIndex<float> *block_results)
// {
//     extern __shared__ ValIndex<float> sdata_val_f[];
//     auto tid = threadIdx.x;
//     auto i = blockIdx.x * blockDim.x + threadIdx.x;

//     // each thread loads value, or +inf if out of range
//     auto v = (i < len) ? in[i] : CUDART_INF_F;
//     sdata_val_f[tid].val = v;
//     sdata_val_f[tid].idx = (i < len) ? i : static_cast<ptrdiff_t>(-1); // -1 if out-of-range
//     __syncthreads();

//     // do block-level reduction
//     for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (tid < stride)
//         {
//             // compare sdata[tid + stride].val with sdata[tid].val
//             if (sdata_val_f[tid + stride].val > sdata_val_f[tid].val)
//             {
//                 sdata_val_f[tid] = sdata_val_f[tid + stride];
//             }
//         }
//         __syncthreads();
//     }

//     // final block min in sdata[0]
//     if (tid == 0)
//     {
//         block_results[blockIdx.x] = sdata_val_f[0];
//     }
// }

// __global__ void kernelBlockArgMaxD(double const *in, size_t len, ValIndex<double> *block_results)
// {
//     extern __shared__ ValIndex<double> sdata_val_d[];
//     auto tid = threadIdx.x;
//     auto i = blockIdx.x * blockDim.x + threadIdx.x;

//     // each thread loads value, or +inf if out of range
//     auto v = (i < len) ? in[i] : CUDART_INF;
//     sdata_val_d[tid].val = v;
//     sdata_val_d[tid].idx = (i < len) ? i : static_cast<ptrdiff_t>(-1); // -1 if out-of-range
//     __syncthreads();

//     // do block-level reduction
//     for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (tid < stride)
//         {
//             // compare sdata[tid + stride].val with sdata[tid].val
//             if (sdata_val_d[tid + stride].val > sdata_val_d[tid].val)
//             {
//                 sdata_val_d[tid] = sdata_val_d[tid + stride];
//             }
//         }
//         __syncthreads();
//     }

//     // final block min in sdata[0]
//     if (tid == 0)
//     {
//         block_results[blockIdx.x] = sdata_val_d[0];
//     }
// }

// __global__ void kernelSingleBlockArgMaxF(const ValIndex<float> *block_results, size_t num_blocks, ValIndex<float> *final_out)
// {
//     extern __shared__ ValIndex<float> sdata_val_f[];
//     auto tid = threadIdx.x;

//     // load or +inf if out-of-range
//     auto v = (tid < num_blocks) ? block_results[tid].val : CUDART_INF_F;
//     sdata_val_f[tid].val = v;
//     sdata_val_f[tid].idx = (tid < num_blocks) ? block_results[tid].idx : -1;
//     __syncthreads();

//     for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (tid < stride)
//         {
//             if (sdata_val_f[tid + stride].val > sdata_val_f[tid].val)
//             {
//                 sdata_val_f[tid] = sdata_val_f[tid + stride];
//             }
//         }
//         __syncthreads();
//     }

//     if (tid == 0)
//     {
//         // store final result
//         final_out->val = sdata_val_f[0].val;
//         final_out->idx = sdata_val_f[0].idx;
//     }
// }

// __global__ void kernelSingleBlockArgMaxD(const ValIndex<double> *block_results, size_t num_blocks, ValIndex<double> *final_out)
// {
//     extern __shared__ ValIndex<double> sdata_val_d[];
//     auto tid = threadIdx.x;

//     // load or +inf if out-of-range
//     auto v = (tid < num_blocks) ? block_results[tid].val : CUDART_INF;
//     sdata_val_d[tid].val = v;
//     sdata_val_d[tid].idx = (tid < num_blocks) ? block_results[tid].idx : -1;
//     __syncthreads();

//     for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (tid < stride)
//         {
//             if (sdata_val_d[tid + stride].val > sdata_val_d[tid].val)
//             {
//                 sdata_val_d[tid] = sdata_val_d[tid + stride];
//             }
//         }
//         __syncthreads();
//     }

//     if (tid == 0)
//     {
//         // store final result
//         final_out->val = sdata_val_d[0].val;
//         final_out->idx = sdata_val_d[0].idx;
//     }
// }

// cudaError_t tomoArgMax(auto const *in,
//                        size_t len,
//                        auto *host_min_val,   // where we store the final min value
//                        size_t *host_min_idx, // where we store the final index
//                        unsigned int threads_per_block,
//                        cudaStream_t stream)
// {
//     using T = std::remove_cvref_t<decltype(*in)>;
//     using ValIndexT = ValIndex<T>;

//     if (len == 0)
//     {
//         return cudaErrorInvalidValue;
//     }

//     // 1) #blocks for first pass
//     auto blocks = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;

//     // 2) block-level results array
//     auto block_results = static_cast<ValIndexT *>(nullptr);
//     auto err = cudaMallocAsync((void **)&block_results, blocks * sizeof(ValIndexT), stream);
//     if (err != cudaSuccess)
//     {
//         return err;
//     }

//     // 3) kernelBlockArgMinF
//     auto shared_size = threads_per_block * sizeof(ValIndexT);

//     if constexpr (std::is_same_v<T, float>)
//     {
//         kernelBlockArgMaxF<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
//     }
//     else if constexpr (std::is_same_v<T, double>)
//     {
//         kernelBlockArgMaxD<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
//     }
//     else
//     {
//         static_assert(std::is_floating_point_v<T>);
//     }

//     err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(block_results, stream);
//         return err;
//     }

//     // 4) single-block final reduce
//     auto d_final = static_cast<ValIndexT *>(nullptr);
//     err = cudaMallocAsync((void **)&d_final, sizeof(ValIndexT), stream);
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(block_results, stream);
//         return err;
//     }

//     // pick #threads in second pass
//     auto threads2 = (blocks < threads_per_block) ? blocks : threads_per_block;
//     auto sharedSize2 = threads2 * sizeof(ValIndexT);

//     if constexpr (std::is_same_v<T, float>)
//     {
//         kernelSingleBlockArgMaxF<<<1, threads2, sharedSize2, stream>>>(block_results, blocks, d_final);
//     }
//     else if constexpr (std::is_same_v<T, double>)
//     {
//         kernelSingleBlockArgMaxD<<<1, threads2, sharedSize2, stream>>>(block_results, blocks, d_final);
//     }
//     else
//     {
//         static_assert(std::is_floating_point_v<T>);
//     }

//     err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(d_final, stream);
//         cudaFreeAsync(block_results, stream);
//         return err;
//     }

//     // 5) copy final result
//     ValIndexT h_final;
//     err = cudaMemcpyAsync(&h_final, d_final, sizeof(ValIndexT), cudaMemcpyDeviceToHost, stream);
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(d_final, stream);
//         cudaFreeAsync(block_results, stream);
//         return err;
//     }

//     // sync if we need result now
//     err = cudaStreamSynchronize(stream);
//     if (err != cudaSuccess)
//     {
//         cudaFreeAsync(d_final, stream);
//         cudaFreeAsync(block_results, stream);
//         return err;
//     }

//     // 6) store to host
//     *host_min_val = h_final.val;
//     *host_min_idx = static_cast<size_t>(h_final.idx);

//     // 7) cleanup
//     cudaFreeAsync(d_final, stream);
//     cudaFreeAsync(block_results, stream);

//     return cudaSuccess;
// }

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMaxF(float const *in,
//                                                    size_t len,
//                                                    float *host_min_val,  // where we store the final min value
//                                                    size_t *host_min_idx, // where we store the final index
//                                                    unsigned int threads_per_block,
//                                                    cudaStream_t stream)

// {
//     return tomoArgMax(in, len, host_min_val, host_min_idx, threads_per_block, stream);
// }

// TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMaxD(double const *in,
//                                                    size_t len,
//                                                    double *host_min_val, // where we store the final min value
//                                                    size_t *host_min_idx, // where we store the final index
//                                                    unsigned int threads_per_block,
//                                                    cudaStream_t stream)

// {
//     return tomoArgMax(in, len, host_min_val, host_min_idx, threads_per_block, stream);
// }