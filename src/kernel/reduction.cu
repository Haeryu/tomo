#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"
#include "reduction.h"

// #include <cmath>
#include <cfloat> // for FLT_MAX, etc.
#include <math_constants.h>

__global__ void kernelReduceSumF(float const *in, size_t len, float *partial_sum)
{
    extern __shared__ float sdata_f[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    // 1) Load data or 0 if out-of-range
    auto accum = 0.0f;
    if (i < len)
    {
        accum = in[i];
    }
    sdata_f[tid] = accum;
    __syncthreads();

    // 2) Intra-block reduction
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_f[tid] += sdata_f[tid + stride];
        }
        __syncthreads();
    }

    // 3) Write one partial sum per block
    if (tid == 0)
    {
        atomicAdd(partial_sum, sdata_f[0]);
    }
}

__global__ void kernelReduceSumD(double const *in, size_t len, double *partial_sum)
{
    extern __shared__ double kernel_reduce_sdatad[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    auto accum = 0.0;
    if (i < len)
    {
        accum = in[i];
    }
    kernel_reduce_sdatad[tid] = accum;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            kernel_reduce_sdatad[tid] += kernel_reduce_sdatad[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(partial_sum, kernel_reduce_sdatad[0]);
    }
}

__device__ inline void atomicMinFloat(float *addr, float val)
{
    // Alias the float pointer as an int pointer
    auto int_addr = reinterpret_cast<int *>(addr);
    auto old = *int_addr; // non-atomic read of the current bits
    while (true)
    {
        auto assumed = old;
        // Convert the bits to float
        auto f_old = __int_as_float(assumed);

        // If the old value is already less than or equal to 'val', we can stop
        if (f_old <= val)
        {
            break;
        }

        // Otherwise, we attempt to store 'val' by CAS
        auto newVal = __float_as_int(val);
        old = atomicCAS(int_addr, assumed, newVal);

        // If CAS succeeded, old == assumed, we are done
        if (old == assumed)
        {
            break;
        }
        // otherwise, another thread changed it, so repeat
    }
}

__device__ inline void atomicMinDouble(double *addr, double val)
{
    auto addr_as_ull = reinterpret_cast<unsigned long long *>(addr);
    auto old = *addr_as_ull; // non-atomic read

    while (true)
    {
        auto assumed = old;
        auto d_old = __longlong_as_double(assumed);

        // If the old value is already <= val, we're done
        if (d_old <= val)
        {
            break;
        }

        auto new_val = __double_as_longlong(val);
        old = atomicCAS(addr_as_ull, assumed, new_val);

        // if no change occurred, we succeeded
        if (old == assumed)
        {
            break;
        }
    }
}

// __device__ inline void atomicMaxFloat(float *addr, float val)
// {
//     auto int_addr = reinterpret_cast<int *>(addr);
//     auto old = *int_addr;
//     while (true)
//     {
//         auto assumed = old;
//         auto f_old = __int_as_float(assumed);

//         if (f_old >= val)
//         {
//             break;
//         }

//         auto newVal = __float_as_int(val);
//         old = atomicCAS(int_addr, assumed, newVal);

//         if (old == assumed)
//         {
//             break;
//         }
//     }
// }

// __device__ inline void atomicMaxDouble(double *addr, double val)
// {
//     auto addr_as_ull = reinterpret_cast<unsigned long long *>(addr);
//     auto old = *addr_as_ull;

//     while (true)
//     {
//         auto assumed = old;
//         auto d_old = __longlong_as_double(assumed);

//         if (d_old >= val)
//         {
//             break;
//         }

//         auto newVal = __double_as_longlong(val);
//         old = atomicCAS(addr_as_ull, assumed, newVal);

//         if (old == assumed)
//         {
//             break;
//         }
//     }
// }

// // TODO
// __global__ void kernelReduceMinF(float const *in, size_t len, float *partial_min)
// {
//     extern __shared__ float sdata_f[];
//     auto tid = threadIdx.x;
//     auto i = blockIdx.x * blockDim.x + threadIdx.x;

//     // initialize accum to +INF if out-of-range
//     auto accum = (i < len) ? in[i] : CUDART_INF_F;
//     sdata_f[tid] = accum;
//     __syncthreads();

//     for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (tid < stride)
//         {
//             sdata_f[tid] = fmin(sdata_f[tid], sdata_f[tid + stride]);
//         }
//         __syncthreads();
//     }

//     if (tid == 0)
//     {
//         atomicMinFloat(partial_min, sdata_f[0]);
//     }
// }

// __global__ void kernelReduceMinD(double const *in, size_t len, double *partial_min)
// {
//     extern __shared__ double sdata_d[];
//     auto tid = threadIdx.x;
//     auto i = blockIdx.x * blockDim.x + threadIdx.x;

//     // If out of range, set +inf
//     auto accum = (i < len) ? in[i] : DBL_MAX;
//     sdata_d[tid] = accum;
//     __syncthreads();

//     // Intra-block reduce
//     for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (tid < stride)
//         {
//             sdata_d[tid] = fmin(sdata_d[tid], sdata_d[tid + stride]);
//         }
//         __syncthreads();
//     }

//     // final block-level result in sdata_d[0]
//     if (tid == 0)
//     {
//         atomicMinDouble(partial_min, sdata_d[0]);
//     }
// }

// // TODO
// __global__ void kernelReduceMaxF(const float *in, size_t len, float *partial_max)
// {
//     extern __shared__ float sdata_f[];
//     auto tid = threadIdx.x;
//     auto i = blockIdx.x * blockDim.x + threadIdx.x;

//     auto accum = (i < len) ? in[i] : -CUDART_INF_F;
//     sdata_f[tid] = accum;
//     __syncthreads();

//     for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (tid < stride)
//         {
//             sdata_f[tid] = fmax(sdata_f[tid], sdata_f[tid + stride]);
//         }
//         __syncthreads();
//     }

//     if (tid == 0)
//     {
//         atomicMinFloat(partial_max, sdata_f[0]);
//     }
// }

// __global__ void kernelReduceMaxD(const double *in, size_t len, double *partial_max)
// {
//     extern __shared__ double sdata_d[];
//     auto tid = threadIdx.x;
//     auto i = blockIdx.x * blockDim.x + threadIdx.x;

//     // If out of range, set -inf
//     double accum = (i < len) ? in[i] : -DBL_MAX;
//     sdata_d[tid] = accum;
//     __syncthreads();

//     for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
//     {
//         if (tid < stride)
//         {
//             sdata_d[tid] = fmax(sdata_d[tid], sdata_d[tid + stride]);
//         }
//         __syncthreads();
//     }

//     if (tid == 0)
//     {
//         // NOTE: we want "atomicMaxDouble", not "atomicMinDouble" here.
//         // Probably a copy/paste fix from your float code. So let's correct it:
//         atomicMaxDouble(partial_max, sdata_d[0]);
//     }
// }

__global__ void kernelReduceSumOfSquaresF(const float *in, size_t len, float *partialSum)
{
    extern __shared__ float sdata_f[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    auto accum = 0.0f;
    if (i < len)
    {
        float v = in[i];
        accum = v * v;
    }
    sdata_f[tid] = accum;
    __syncthreads();

    // reduce like usual
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_f[tid] += sdata_f[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(partialSum, sdata_f[0]);
    }
}

__global__ void kernelReduceSumOfSquaresD(double const *in, size_t len, double *partial_sum)
{
    extern __shared__ double sdata_d[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    auto accum = 0.0;
    if (i < len)
    {
        auto v = in[i];
        accum = v * v; // square
    }
    sdata_d[tid] = accum;
    __syncthreads();

    // reduce
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_d[tid] += sdata_d[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(partial_sum, sdata_d[0]); // GPU CC >= 6.0
    }
}

cudaError_t tomoSumReduce(auto const *a,
                          size_t len,
                          auto *hostOut, // pointer to store result on host
                          unsigned int threads_per_block,
                          cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*a)>;
    if (len == 0)
    {
        *hostOut = static_cast<T>(0.0);
        return cudaSuccess;
    }

    auto d_sum = static_cast<T *>(nullptr);

    // 1) Allocate device memory for partial sum
    auto err = cudaMallocAsync((void **)&d_sum, sizeof(T), stream);
    if (err != cudaSuccess)
    {
        return err;
    }

    // 2) Memset to 0
    err = cudaMemsetAsync(d_sum, 0, sizeof(T), stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_sum, stream);
        return err;
    }

    // 3) Launch kernel
    auto blocks = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;
    auto sharedSize = threads_per_block * sizeof(T);

    if constexpr (std::is_same_v<T, float>)
    {
        kernelReduceSumF<<<blocks, threads_per_block, sharedSize, stream>>>(a, len, d_sum);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        kernelReduceSumD<<<blocks, threads_per_block, sharedSize, stream>>>(a, len, d_sum);
    }
    else
    {
        static_assert(std::is_floating_point_v<T>);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_sum, stream);
        return err;
    }

    // 4) Copy back result
    err = cudaMemcpyAsync(hostOut, d_sum, sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_sum, stream);
        return err;
    }

    // 5) Sync if you need the result immediately
    // err = cudaStreamSynchronize(stream);
    // If you want truly async usage, skip the sync here, but be aware
    // that hostOut is not valid until the stream completes.
    // if (err != cudaSuccess)
    // {
    //     cudaFreeAsync(d_sum, stream);
    //     return err;
    // }

    // 6) Clean up
    cudaFreeAsync(d_sum, stream);

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceF(float const *a,
                                                      size_t len,
                                                      float *host_out, // pointer to store result on host
                                                      unsigned int threads_per_block,
                                                      cudaStream_t stream)
{
    return tomoSumReduce(a, len, host_out, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSumReduceD(double const *a,
                                                      size_t len,
                                                      double *host_out, // pointer to store result on host
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

__global__ void kernelBlockMinF(float const *in, size_t len, float *block_results)
{
    extern __shared__ float sdata_f[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    auto accum = (i < len) ? in[i] : CUDART_INF_F;
    sdata_f[tid] = accum;
    __syncthreads();

    // reduce within block
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_f[tid] = fmin(sdata_f[tid], sdata_f[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // store the block's min
        block_results[blockIdx.x] = sdata_f[0];
    }
}

__global__ void kernelBlockMinD(double const *in, size_t len, double *block_results)
{
    extern __shared__ double sdata_d[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    // use +INF if out-of-range
    auto accum = (i < len) ? in[i] : CUDART_INF;
    sdata_d[tid] = accum;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_d[tid] = fmin(sdata_d[tid], sdata_d[tid + stride]);
        }
        __syncthreads();
    }

    // each block writes a single result
    if (tid == 0)
    {
        block_results[blockIdx.x] = sdata_d[0];
    }
}

__global__ void kernelBlockMaxF(float const *in, size_t len, float *block_results)
{
    extern __shared__ float sdata_f[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    auto accum = (i < len) ? in[i] : -CUDART_INF_F;
    sdata_f[tid] = accum;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_f[tid] = fmax(sdata_f[tid], sdata_f[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        block_results[blockIdx.x] = sdata_f[0];
    }
}

__global__ void kernelBlockMaxD(double const *in, size_t len, double *block_results)
{
    extern __shared__ double sdata_d[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    auto accum = (i < len) ? in[i] : -CUDART_INF;
    sdata_d[tid] = accum;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_d[tid] = fmax(sdata_d[tid], sdata_d[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        block_results[blockIdx.x] = sdata_d[0];
    }
}

__global__ void kernelSingleBlockMinF(float const *blockResults, size_t numBlocks, float *finalMin)
{
    extern __shared__ float sdata_f[];
    auto tid = threadIdx.x;
    auto i = tid;

    // load from blockResults array or +inf if out-of-range
    double accum = (i < numBlocks) ? blockResults[i] : CUDART_INF_F;
    sdata_f[tid] = accum;
    __syncthreads();

    // do normal block reduction
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_f[tid] = fmin(sdata_f[tid], sdata_f[tid + stride]);
        }
        __syncthreads();
    }

    // final result in sdata[0]
    if (tid == 0)
    {
        *finalMin = sdata_f[0];
    }
}

__global__ void kernelSingleBlockMinD(double const *blockResults, size_t numBlocks, double *finalMin)
{
    extern __shared__ double sdata_d[];
    auto tid = threadIdx.x;
    auto i = tid;

    // load from blockResults array or +inf if out-of-range
    auto accum = (i < numBlocks) ? blockResults[i] : CUDART_INF;
    sdata_d[tid] = accum;
    __syncthreads();

    // do normal block reduction
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_d[tid] = fmin(sdata_d[tid], sdata_d[tid + stride]);
        }
        __syncthreads();
    }

    // final result in sdata[0]
    if (tid == 0)
    {
        *finalMin = sdata_d[0];
    }
}

__global__ void kernelSingleBlockMaxF(float const *block_results, size_t num_blocks, float *final_max)
{
    extern __shared__ float sdata_f[];
    auto tid = threadIdx.x;
    auto i = tid;

    auto accum = (i < num_blocks) ? block_results[i] : -CUDART_INF_F;
    sdata_f[tid] = accum;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_f[tid] = fmax(sdata_f[tid], sdata_f[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        *final_max = sdata_f[0];
    }
}

__global__ void kernelSingleBlockMaxD(double const *block_results, size_t num_blocks, double *final_max)
{
    extern __shared__ double sdata_d[];
    auto tid = threadIdx.x;
    auto i = tid;

    auto accum = (i < num_blocks) ? block_results[i] : -CUDART_INF;
    sdata_d[tid] = accum;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_d[tid] = fmax(sdata_d[tid], sdata_d[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        *final_max = sdata_d[0];
    }
}

cudaError_t tomoMin(const auto *in,
                    size_t len,
                    auto *host_out, // final result on host
                    unsigned int threads_per_block,
                    cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*in)>;
    if (len == 0)
    {
        // maybe define min of empty array as +inf or do an error
        // *hostOut = CUDART_INF;
        return cudaErrorInvalidValue;
    }

    // 1) #blocks in first pass
    auto blocks = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;

    // 2) Allocate array "d_blockResults" to hold the min of each block
    auto block_results = static_cast<T *>(nullptr);
    auto err = cudaMallocAsync((void **)&block_results, blocks * sizeof(T), stream);
    if (err != cudaSuccess)
    {
        return err;
    }

    // 3) Launch kernelBlockMinD to fill block-wise results
    {
        auto shared_size = threads_per_block * sizeof(T);

        if constexpr (std::is_same_v<T, float>)
        {
            kernelBlockMinF<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            kernelBlockMinD<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
        }
        else
        {
            static_assert(std::is_floating_point_v<T>);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFreeAsync(block_results, stream);
            return err;
        }
    }

    // 4) Now we do a second pass to reduce that array of size = blocks
    // We'll do it in a single-block kernel for simplicity:
    auto d_final = static_cast<T *>(nullptr);
    err = cudaMallocAsync((void **)&d_final, sizeof(T), stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(block_results, stream);
        return err;
    }

    {
        // single-block kernel with at least 'blocks' threads if blocks < threads.
        // but let's pick threads = next power of two >= blocks, or just do blocks again if blocks < threads.
        auto threads2 = threads_per_block;
        if (blocks < threads_per_block)
        {
            threads2 = blocks;
        }
        auto shared_size = threads2 * sizeof(T);

        using T = std::remove_cvref_t<decltype(*block_results)>;
        if constexpr (std::is_same_v<T, float>)
        {
            kernelSingleBlockMinF<<<1, threads2, shared_size, stream>>>(block_results, blocks, d_final);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            kernelSingleBlockMinD<<<1, threads2, shared_size, stream>>>(block_results, blocks, d_final);
        }
        else
        {
            static_assert(std::is_floating_point_v<T>);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFreeAsync(d_final, stream);
            cudaFreeAsync(block_results, stream);
            return err;
        }
    }

    // 5) Copy final result to host
    err = cudaMemcpyAsync(host_out, d_final, sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_final, stream);
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // 6) Sync if needed
    // err = cudaStreamSynchronize(stream);
    // if (err != cudaSuccess)
    // {
    //     cudaFreeAsync(d_final, stream);
    //     cudaFreeAsync(d_blockResults, stream);
    //     return err;
    // }

    // 7) Clean up
    cudaFreeAsync(d_final, stream);
    cudaFreeAsync(block_results, stream);

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinF(float const *in,
                                                size_t len,
                                                float *host_out, // final result on host
                                                unsigned int threads_per_block,
                                                cudaStream_t stream)
{
    return tomoMin(in, len, host_out, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMinD(double const *in,
                                                size_t len,
                                                double *host_out, // final result on host
                                                unsigned int threads_per_block,
                                                cudaStream_t stream)
{
    return tomoMin(in, len, host_out, threads_per_block, stream);
}

cudaError_t tomoMax(auto const *in,
                    size_t len,
                    auto *hostout, // final result on host
                    unsigned int threads_per_block,
                    cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*in)>;
    if (len == 0)
    {
        // maybe define min of empty array as +inf or do an error
        // *hostOut = CUDART_INF;
        return cudaErrorInvalidValue;
    }

    auto blocks = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;

    auto block_results = static_cast<T *>(nullptr);
    auto err = cudaMallocAsync((void **)&block_results, blocks * sizeof(T), stream);
    if (err != cudaSuccess)
    {
        return err;
    }

    // First pass: block-level max
    {
        auto shared_size = threads_per_block * sizeof(T);

        if constexpr (std::is_same_v<T, float>)
        {
            kernelBlockMaxF<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            kernelBlockMaxD<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
        }
        else
        {
            static_assert(std::is_floating_point_v<T>);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFreeAsync(block_results, stream);
            return err;
        }
    }

    // Second pass: single-block reduce of blockResults
    auto final = static_cast<T *>(nullptr);
    err = cudaMallocAsync((void **)&final, sizeof(T), stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(block_results, stream);
        return err;
    }

    {
        auto threads2 = threads_per_block;
        if (blocks < threads_per_block)
        {
            threads2 = blocks;
        }
        auto shared_size = threads2 * sizeof(T);

        if constexpr (std::is_same_v<T, float>)
        {
            kernelSingleBlockMaxF<<<1, threads2, shared_size, stream>>>(block_results, blocks, final);
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            kernelSingleBlockMaxD<<<1, threads2, shared_size, stream>>>(block_results, blocks, final);
        }
        else
        {
            static_assert(std::is_floating_point_v<T>);
        }
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            cudaFreeAsync(final, stream);
            cudaFreeAsync(block_results, stream);
            return err;
        }
    }

    err = cudaMemcpyAsync(hostout, final, sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(final, stream);
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // err = cudaStreamSynchronize(stream);
    // if (err != cudaSuccess)
    // {
    //     cudaFreeAsync(final, stream);
    //     cudaFreeAsync(block_results, stream);
    //     return err;
    // }

    // cleanup
    cudaFreeAsync(final, stream);
    cudaFreeAsync(block_results, stream);

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

cudaError_t tomoL2Norm(auto const *a,
                       size_t len,
                       auto *host_out, // where we store final L2 norm on the host
                       unsigned int threads_per_block,
                       cudaStream_t stream)
{

    using T = std::remove_cvref_t<decltype(*a)>;
    // Edge case: length == 0
    if (len == 0)
    {
        // maybe define min of empty array as +inf or do an error
        // *host_out = 0.f;
        return cudaErrorInvalidValue;
    }

    auto sum = static_cast<T *>(nullptr);

    // 1) Allocate device memory for partial sum
    auto err = cudaMallocAsync((void **)&sum, sizeof(T), stream);
    if (err != cudaSuccess)
    {
        return err;
    }

    // 2) Zero it out
    err = cudaMemsetAsync(sum, 0, sizeof(T), stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(sum, stream);
        return err;
    }

    // 3) Launch the kernel
    auto blocks = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;
    auto sharedSize = threads_per_block * sizeof(T);

    if constexpr (std::is_same_v<T, float>)
    {
        kernelReduceSumOfSquaresF<<<blocks, threads_per_block, sharedSize, stream>>>(a, len, sum);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        kernelReduceSumOfSquaresD<<<blocks, threads_per_block, sharedSize, stream>>>(a, len, sum);
    }
    else
    {
        static_assert(std::is_floating_point_v<T>);
    }

    err = cudaGetLastError(); // check for launch errors
    if (err != cudaSuccess)
    {
        cudaFreeAsync(sum, stream);
        return err;
    }

    // 4) Copy the partialSum result back to the host
    err = cudaMemcpyAsync(host_out, sum, sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(sum, stream);
        return err;
    }

    // 5) We must synchronize if we need *hostOut now
    //    (Otherwise, we do an event or let caller sync.)
    // err = cudaStreamSynchronize(stream);
    // if (err != cudaSuccess)
    // {
    //     cudaFreeAsync(sum, stream);
    //     return err;
    // }

    // 6) The sum of squares is now in *hostOut
    //    L2 norm = sqrtf( sum_of_squares ).
    *host_out = sqrt(*host_out);

    // 7) Cleanup
    cudaFreeAsync(sum, stream);

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
struct ValIndex
{
    ptrdiff_t idx;
    T val;
};

__global__ void kernelBlockArgMinF(float const *in, size_t len, ValIndex<float> *block_results)
{
    extern __shared__ ValIndex<float> sdata_val_f[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread loads value, or +inf if out of range
    auto v = (i < len) ? in[i] : CUDART_INF_F;
    sdata_val_f[tid].val = v;
    sdata_val_f[tid].idx = (i < len) ? i : static_cast<ptrdiff_t>(-1); // -1 if out-of-range
    __syncthreads();

    // do block-level reduction
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            // compare sdata[tid + stride].val with sdata[tid].val
            if (sdata_val_f[tid + stride].val < sdata_val_f[tid].val)
            {
                sdata_val_f[tid] = sdata_val_f[tid + stride];
            }
        }
        __syncthreads();
    }

    // final block min in sdata[0]
    if (tid == 0)
    {
        block_results[blockIdx.x] = sdata_val_f[0];
    }
}

__global__ void kernelBlockArgMinD(double const *in, size_t len, ValIndex<double> *block_results)
{
    extern __shared__ ValIndex<double> sdata_val_d[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread loads value, or +inf if out of range
    auto v = (i < len) ? in[i] : CUDART_INF;
    sdata_val_d[tid].val = v;
    sdata_val_d[tid].idx = (i < len) ? i : static_cast<ptrdiff_t>(-1); // -1 if out-of-range
    __syncthreads();

    // do block-level reduction
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            // compare sdata[tid + stride].val with sdata[tid].val
            if (sdata_val_d[tid + stride].val < sdata_val_d[tid].val)
            {
                sdata_val_d[tid] = sdata_val_d[tid + stride];
            }
        }
        __syncthreads();
    }

    // final block min in sdata[0]
    if (tid == 0)
    {
        block_results[blockIdx.x] = sdata_val_d[0];
    }
}

__global__ void kernelSingleBlockArgMinF(const ValIndex<float> *block_results, size_t num_blocks, ValIndex<float> *final_out)
{
    extern __shared__ ValIndex<float> sdata_val_f[];
    auto tid = threadIdx.x;

    // load or +inf if out-of-range
    auto v = (tid < num_blocks) ? block_results[tid].val : CUDART_INF_F;
    sdata_val_f[tid].val = v;
    sdata_val_f[tid].idx = (tid < num_blocks) ? block_results[tid].idx : -1;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (sdata_val_f[tid + stride].val < sdata_val_f[tid].val)
            {
                sdata_val_f[tid] = sdata_val_f[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // store final result
        final_out->val = sdata_val_f[0].val;
        final_out->idx = sdata_val_f[0].idx;
    }
}

__global__ void kernelSingleBlockArgMinD(const ValIndex<double> *block_results, size_t num_blocks, ValIndex<double> *final_out)
{
    extern __shared__ ValIndex<double> sdata_val_d[];
    auto tid = threadIdx.x;

    // load or +inf if out-of-range
    auto v = (tid < num_blocks) ? block_results[tid].val : CUDART_INF;
    sdata_val_d[tid].val = v;
    sdata_val_d[tid].idx = (tid < num_blocks) ? block_results[tid].idx : -1;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (sdata_val_d[tid + stride].val < sdata_val_d[tid].val)
            {
                sdata_val_d[tid] = sdata_val_d[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // store final result
        final_out->val = sdata_val_d[0].val;
        final_out->idx = sdata_val_d[0].idx;
    }
}

cudaError_t tomoArgMin(auto const *in,
                       size_t len,
                       auto *host_min_val,   // where we store the final min value
                       size_t *host_min_idx, // where we store the final index
                       unsigned int threads_per_block,
                       cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*in)>;
    using ValIndexT = ValIndex<T>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    // 1) #blocks for first pass
    auto blocks = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;

    // 2) block-level results array
    auto block_results = static_cast<ValIndexT *>(nullptr);
    auto err = cudaMallocAsync((void **)&block_results, blocks * sizeof(ValIndexT), stream);
    if (err != cudaSuccess)
    {
        return err;
    }

    // 3) kernelBlockArgMinF
    auto shared_size = threads_per_block * sizeof(ValIndexT);

    if constexpr (std::is_same_v<T, float>)
    {
        kernelBlockArgMinF<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        kernelBlockArgMinD<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
    }
    else
    {
        static_assert(std::is_floating_point_v<T>);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // 4) single-block final reduce
    auto d_final = static_cast<ValIndexT *>(nullptr);
    err = cudaMallocAsync((void **)&d_final, sizeof(ValIndexT), stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // pick #threads in second pass
    auto threads2 = (blocks < threads_per_block) ? blocks : threads_per_block;
    auto sharedSize2 = threads2 * sizeof(ValIndexT);

    if constexpr (std::is_same_v<T, float>)
    {
        kernelSingleBlockArgMinF<<<1, threads2, sharedSize2, stream>>>(block_results, blocks, d_final);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        kernelSingleBlockArgMinD<<<1, threads2, sharedSize2, stream>>>(block_results, blocks, d_final);
    }
    else
    {
        static_assert(std::is_floating_point_v<T>);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_final, stream);
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // 5) copy final result
    ValIndexT h_final;
    err = cudaMemcpyAsync(&h_final, d_final, sizeof(ValIndexT), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_final, stream);
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // sync if we need result now
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_final, stream);
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // 6) store to host
    *host_min_val = h_final.val;
    *host_min_idx = h_final.idx;

    // 7) cleanup
    cudaFreeAsync(d_final, stream);
    cudaFreeAsync(block_results, stream);

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMinF(float const *in,
                                                   size_t len,
                                                   float *host_min_val,  // where we store the final min value
                                                   size_t *host_min_idx, // where we store the final index
                                                   unsigned int threads_per_block,
                                                   cudaStream_t stream)

{
    return tomoArgMin(in, len, host_min_val, host_min_idx, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMinD(double const *in,
                                                   size_t len,
                                                   double *host_min_val, // where we store the final min value
                                                   size_t *host_min_idx, // where we store the final index
                                                   unsigned int threads_per_block,
                                                   cudaStream_t stream)

{
    return tomoArgMin(in, len, host_min_val, host_min_idx, threads_per_block, stream);
}

__global__ void kernelBlockArgMaxF(float const *in, size_t len, ValIndex<float> *block_results)
{
    extern __shared__ ValIndex<float> sdata_val_f[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread loads value, or +inf if out of range
    auto v = (i < len) ? in[i] : CUDART_INF_F;
    sdata_val_f[tid].val = v;
    sdata_val_f[tid].idx = (i < len) ? i : static_cast<ptrdiff_t>(-1); // -1 if out-of-range
    __syncthreads();

    // do block-level reduction
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            // compare sdata[tid + stride].val with sdata[tid].val
            if (sdata_val_f[tid + stride].val > sdata_val_f[tid].val)
            {
                sdata_val_f[tid] = sdata_val_f[tid + stride];
            }
        }
        __syncthreads();
    }

    // final block min in sdata[0]
    if (tid == 0)
    {
        block_results[blockIdx.x] = sdata_val_f[0];
    }
}

__global__ void kernelBlockArgMaxD(double const *in, size_t len, ValIndex<double> *block_results)
{
    extern __shared__ ValIndex<double> sdata_val_d[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread loads value, or +inf if out of range
    auto v = (i < len) ? in[i] : CUDART_INF;
    sdata_val_d[tid].val = v;
    sdata_val_d[tid].idx = (i < len) ? i : static_cast<ptrdiff_t>(-1); // -1 if out-of-range
    __syncthreads();

    // do block-level reduction
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            // compare sdata[tid + stride].val with sdata[tid].val
            if (sdata_val_d[tid + stride].val > sdata_val_d[tid].val)
            {
                sdata_val_d[tid] = sdata_val_d[tid + stride];
            }
        }
        __syncthreads();
    }

    // final block min in sdata[0]
    if (tid == 0)
    {
        block_results[blockIdx.x] = sdata_val_d[0];
    }
}

__global__ void kernelSingleBlockArgMaxF(const ValIndex<float> *block_results, size_t num_blocks, ValIndex<float> *final_out)
{
    extern __shared__ ValIndex<float> sdata_val_f[];
    auto tid = threadIdx.x;

    // load or +inf if out-of-range
    auto v = (tid < num_blocks) ? block_results[tid].val : CUDART_INF_F;
    sdata_val_f[tid].val = v;
    sdata_val_f[tid].idx = (tid < num_blocks) ? block_results[tid].idx : -1;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (sdata_val_f[tid + stride].val > sdata_val_f[tid].val)
            {
                sdata_val_f[tid] = sdata_val_f[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // store final result
        final_out->val = sdata_val_f[0].val;
        final_out->idx = sdata_val_f[0].idx;
    }
}

__global__ void kernelSingleBlockArgMaxD(const ValIndex<double> *block_results, size_t num_blocks, ValIndex<double> *final_out)
{
    extern __shared__ ValIndex<double> sdata_val_d[];
    auto tid = threadIdx.x;

    // load or +inf if out-of-range
    auto v = (tid < num_blocks) ? block_results[tid].val : CUDART_INF;
    sdata_val_d[tid].val = v;
    sdata_val_d[tid].idx = (tid < num_blocks) ? block_results[tid].idx : -1;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (sdata_val_d[tid + stride].val > sdata_val_d[tid].val)
            {
                sdata_val_d[tid] = sdata_val_d[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        // store final result
        final_out->val = sdata_val_d[0].val;
        final_out->idx = sdata_val_d[0].idx;
    }
}

cudaError_t tomoArgMax(auto const *in,
                       size_t len,
                       auto *host_min_val,   // where we store the final min value
                       size_t *host_min_idx, // where we store the final index
                       unsigned int threads_per_block,
                       cudaStream_t stream)
{
    using T = std::remove_cvref_t<decltype(*in)>;
    using ValIndexT = ValIndex<T>;

    if (len == 0)
    {
        return cudaErrorInvalidValue;
    }

    // 1) #blocks for first pass
    auto blocks = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;

    // 2) block-level results array
    auto block_results = static_cast<ValIndexT *>(nullptr);
    auto err = cudaMallocAsync((void **)&block_results, blocks * sizeof(ValIndexT), stream);
    if (err != cudaSuccess)
    {
        return err;
    }

    // 3) kernelBlockArgMinF
    auto shared_size = threads_per_block * sizeof(ValIndexT);

    if constexpr (std::is_same_v<T, float>)
    {
        kernelBlockArgMaxF<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        kernelBlockArgMaxD<<<blocks, threads_per_block, shared_size, stream>>>(in, len, block_results);
    }
    else
    {
        static_assert(std::is_floating_point_v<T>);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // 4) single-block final reduce
    auto d_final = static_cast<ValIndexT *>(nullptr);
    err = cudaMallocAsync((void **)&d_final, sizeof(ValIndexT), stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // pick #threads in second pass
    auto threads2 = (blocks < threads_per_block) ? blocks : threads_per_block;
    auto sharedSize2 = threads2 * sizeof(ValIndexT);

    if constexpr (std::is_same_v<T, float>)
    {
        kernelSingleBlockArgMaxF<<<1, threads2, sharedSize2, stream>>>(block_results, blocks, d_final);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        kernelSingleBlockArgMaxD<<<1, threads2, sharedSize2, stream>>>(block_results, blocks, d_final);
    }
    else
    {
        static_assert(std::is_floating_point_v<T>);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_final, stream);
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // 5) copy final result
    ValIndexT h_final;
    err = cudaMemcpyAsync(&h_final, d_final, sizeof(ValIndexT), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_final, stream);
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // sync if we need result now
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(d_final, stream);
        cudaFreeAsync(block_results, stream);
        return err;
    }

    // 6) store to host
    *host_min_val = h_final.val;
    *host_min_idx = static_cast<size_t>(h_final.idx);

    // 7) cleanup
    cudaFreeAsync(d_final, stream);
    cudaFreeAsync(block_results, stream);

    return cudaSuccess;
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMaxF(float const *in,
                                                   size_t len,
                                                   float *host_min_val,  // where we store the final min value
                                                   size_t *host_min_idx, // where we store the final index
                                                   unsigned int threads_per_block,
                                                   cudaStream_t stream)

{
    return tomoArgMax(in, len, host_min_val, host_min_idx, threads_per_block, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgMaxD(double const *in,
                                                   size_t len,
                                                   double *host_min_val, // where we store the final min value
                                                   size_t *host_min_idx, // where we store the final index
                                                   unsigned int threads_per_block,
                                                   cudaStream_t stream)

{
    return tomoArgMax(in, len, host_min_val, host_min_idx, threads_per_block, stream);
}

__global__ void kernelReduceSumOfAbsF(const float *in, size_t len, float *partial_sum)
{
    extern __shared__ float sdata_f[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    auto accum = 0.0f;
    if (i < len)
    {
        float v = in[i];
        accum = fabsf(v);
    }
    sdata_f[tid] = accum;
    __syncthreads();

    // standard block reduction
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_f[tid] += sdata_f[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(partial_sum, sdata_f[0]);
    }
}

__global__ void kernelReduceSumOfAbsD(const double *in, size_t len, double *partial_sum)
{
    extern __shared__ double sdata_d[];
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    auto accum = 0.0;
    if (i < len)
    {
        float v = in[i];
        accum = fabsf(v);
    }
    sdata_d[tid] = accum;
    __syncthreads();

    // standard block reduction
    for (auto stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata_d[tid] += sdata_d[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(partial_sum, sdata_d[0]);
    }
}

cudaError_t tomoL1Norm(auto const *a,
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

    auto partial_sum = static_cast<T *>(nullptr);

    // 1) Allocate partial sum
    auto err = cudaMallocAsync((void **)&partial_sum, sizeof(T), stream);
    if (err != cudaSuccess)
        return err;

    err = cudaMemsetAsync(partial_sum, 0, sizeof(T), stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(partial_sum, stream);
        return err;
    }

    // 2) Launch kernel
    auto blocks = (static_cast<unsigned int>(len) + threads_per_block - 1) / threads_per_block;
    auto sharedSize = threads_per_block * sizeof(T);

    if constexpr (std::is_same_v<T, float>)
    {
        kernelReduceSumOfAbsF<<<blocks, threads_per_block, sharedSize, stream>>>(a, len, partial_sum);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        kernelReduceSumOfAbsD<<<blocks, threads_per_block, sharedSize, stream>>>(a, len, partial_sum);
    }
    else
    {
        static_assert(std::is_floating_point_v<T>);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cudaFreeAsync(partial_sum, stream);
        return err;
    }

    // 3) Copy to host
    err = cudaMemcpyAsync(host_out, partial_sum, sizeof(T), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess)
    {
        cudaFreeAsync(partial_sum, stream);
        return err;
    }

    // (Optional) sync if you need immediate result:
    // err = cudaStreamSynchronize(stream);

    cudaFreeAsync(partial_sum, stream);
    return err;
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
