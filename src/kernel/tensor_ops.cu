#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"

#include "tensor_ops.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include "float_op.cuh"

// template <typename T>
// void tomoBroadcastTo(const T *d_in,
//                      T *d_out,
//                      size_t const *in_shape,
//                      size_t in_shape_len,
//                      size_t const *out_shape,
//                      size_t out_shape_len,
//                      size_t const *in_strides,
//                      size_t in_strides_len,
//                      size_t in_size,
//                      size_t out_size,
//                      size_t nd,
//                      cudaStream_t stream)
// {
//     // Copy shape/strides to device arrays
//     size_t *d_in_shape, *d_out_shape, *d_in_stride;

//     // Allocate with stream-ordered memory
//     cudaMallocAsync(&d_in_shape, in_shape_len * sizeof(size_t), stream);
//     cudaMallocAsync(&d_out_shape, out_shape_len * sizeof(size_t), stream);
//     cudaMallocAsync(&d_in_stride, in_strides_len * sizeof(size_t), stream);

//     // Copy data asynchronously
//     cudaMemcpyAsync(d_in_shape, in_shape, in_shape_len * sizeof(size_t), cudaMemcpyHostToDevice, stream);
//     cudaMemcpyAsync(d_out_shape, out_shape, out_shape_len * sizeof(size_t), cudaMemcpyHostToDevice, stream);
//     cudaMemcpyAsync(d_in_stride, in_strides, in_strides_len * sizeof(size_t), cudaMemcpyHostToDevice, stream);

//     // Create counting iterators
//     auto first = thrust::counting_iterator<size_t>{0};
//     auto last = thrust::counting_iterator<size_t>{out_size};

//     // The device lambda (broadcast logic).
//     // We'll capture all by [=], but be mindful that older NVCC
//     // may require you to pass certain pointers explicitly.
//     thrust::transform(thrust::cuda::par_nosync.on(stream),
//                       first, last,
//                       thrust::device_pointer_cast(d_out),
//                       [=] __device__(size_t out_index)
//                       {
//                           // 1) Unravel outIndex -> outCoords
//                           //    row-major from the back
//                           int coords[32]; // if nd <= 32 for demo
//                           auto tmp = out_index;
//                           for (auto d = (ptrdiff_t)nd - 1; d >= 0; d--)
//                           {
//                               auto dim_size = d_out_shape[d];
//                               coords[d] = tmp % dim_size;
//                               tmp /= dim_size;
//                           }

//                           // 2) Build inIndex by mapping coords. If inSh[d] == 1, clamp to 0
//                           auto in_index = (size_t)0;
//                           for (auto d = (size_t)0; d < nd; d++)
//                           {
//                               auto c = (d_in_shape[d] == 1) ? (size_t)0 : coords[d];
//                               in_index += c * d_in_stride[d];
//                           }

//                           // 3) Return in_data[inIndex]
//                           return d_in[in_index];
//                       });

//     cudaFreeAsync(d_in_shape, stream);
//     cudaFreeAsync(d_out_shape, stream);
//     cudaFreeAsync(d_in_stride, stream);
// }

// template <typename T>
// cudaError_t tomoBroadcastToErr(const T *d_in,
//                                T *d_out,
//                                size_t const *in_shape,
//                                size_t in_shape_len,
//                                size_t const *out_shape,
//                                size_t out_shape_len,
//                                size_t const *in_strides,
//                                size_t in_strides_len,
//                                size_t in_size,
//                                size_t out_size,
//                                size_t nd,
//                                cudaStream_t stream)
// {

//     try
//     {
//         tomoBroadcastTo<T>(d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len, in_strides, in_strides_len, in_size, out_size, nd, stream);
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

// #define CHECK_CUDA(call) do { \
//     cudaError_t err = call; \
//     if (err != cudaSuccess) return err; \
// } while(0)

// template <typename T>
// void tomoSumTo(const T *d_in,
//                T *d_out,
//                size_t const *in_shape,
//                size_t in_shape_len,
//                size_t const *out_shape,
//                size_t out_shape_len,
//                size_t const *in_strides,
//                size_t in_strides_len,
//                size_t const *out_strides,
//                size_t out_strides_len,
//                size_t in_size,
//                size_t out_size,
//                size_t nd,
//                cudaStream_t stream)
// {
//     // Copy shape/stride info to device
//     // Copy shape/strides to device arrays
//     size_t *d_in_shape, *d_out_shape, *d_in_stride, *d_out_stride, *keys;

//     // Allocate with stream-ordered memory
//     cudaMallocAsync(&d_in_shape, in_shape_len * sizeof(size_t), stream);
//     cudaMallocAsync(&d_out_shape, out_shape_len * sizeof(size_t), stream);
//     cudaMallocAsync(&d_in_stride, in_strides_len * sizeof(size_t), stream);
//     cudaMallocAsync(&d_out_stride, out_strides_len * sizeof(size_t), stream);
//     cudaMallocAsync(&keys, in_size * sizeof(size_t), stream);

//     // Copy data asynchronously
//     cudaMemcpyAsync(d_in_shape, in_shape, in_shape_len * sizeof(size_t), cudaMemcpyHostToDevice, stream);
//     cudaMemcpyAsync(d_out_shape, out_shape, out_shape_len * sizeof(size_t), cudaMemcpyHostToDevice, stream);
//     cudaMemcpyAsync(d_in_stride, in_strides, in_strides_len * sizeof(size_t), cudaMemcpyHostToDevice, stream);
//     cudaMemcpyAsync(d_out_stride, out_strides, out_strides_len * sizeof(size_t), cudaMemcpyHostToDevice, stream);

//     // 1) Build a device_vector of "keys" for each input index

//     // The lambda that computes the key from inIndex
//     thrust::transform(thrust::cuda::par_nosync.on(stream),
//                       thrust::make_counting_iterator<size_t>(0),
//                       thrust::make_counting_iterator<size_t>(in_size),
//                       keys,
//                       [=] __device__(size_t inIndex)
//                       {
//                           // decode inIndex -> coords
//                           size_t coords[32];
//                           auto tmp = inIndex;
//                           for (auto d = (ptrdiff_t)nd - 1; d >= 0; --d)
//                           {
//                               size_t dimSize = d_in_shape[d];
//                               coords[d] = tmp % dimSize;
//                               tmp /= dimSize;
//                           }

//                           // clamp coords if out_shape[d] == 1 => coords[d] = 0
//                           size_t outCoords[32];
//                           for (auto d = (size_t)0; d < nd; ++d)
//                           {
//                               outCoords[d] = (d_out_shape[d] == 1) ? 0 : coords[d];
//                           }

//                           // re-linearize outCoords => outIndex
//                           auto outIndex = (size_t)0;
//                           for (auto d = (size_t)0; d < nd; ++d)
//                           {
//                               outIndex += outCoords[d] * d_out_stride[d];
//                           }
//                           return outIndex;
//                       });

//     // 2) Sort by key while carrying values
//     //    The values are just d_in[inIndex].
//     T *vals;
//     cudaMallocAsync(&vals, in_size * sizeof(T), stream);
//     thrust::copy_n(thrust::cuda::par_nosync.on(stream), d_in, in_size, vals);

//     thrust::sort_by_key(thrust::cuda::par_nosync.on(stream), keys, keys + in_size, vals);

//     // 3) reduce_by_key
//     size_t *out_keys;
//     T *out_vals;
//     cudaMallocAsync(&out_keys, in_size * sizeof(size_t), stream);
//     cudaMallocAsync(&out_vals, in_size * sizeof(T), stream);

//     auto new_end = thrust::reduce_by_key(thrust::cuda::par_nosync.on(stream),
//                                          keys, keys + in_size,
//                                          vals,
//                                          out_keys,
//                                          out_vals);

//     auto num_unique = new_end.first - out_keys;

//     // 4) Scatter results into d_out
//     //    out_keys[i] = some index in [0..out_size),
//     //    out_vals[i] = sum of all input mapped to that key.

//     // Real approach: we can get raw pointers:

//     // Overwrite with sums (or accumulate)
//     thrust::for_each(thrust::cuda::par_nosync.on(stream),
//                      thrust::make_counting_iterator<size_t>(0),
//                      thrust::make_counting_iterator<size_t>(num_unique),
//                      [=] __device__(size_t i)
//                      {
//                          auto k = out_keys[i];
//                          auto v = out_vals[i];
//                          d_out[k] = v;
//                      });

//     cudaFreeAsync(vals, stream);
//     cudaFreeAsync(out_keys, stream);
//     cudaFreeAsync(out_vals, stream);
//     cudaFreeAsync(d_in_shape, stream);
//     cudaFreeAsync(d_out_shape, stream);
//     cudaFreeAsync(d_in_stride, stream);
//     cudaFreeAsync(d_out_stride, stream);
//     cudaFreeAsync(keys, stream);
// }

// template <typename T>
// cudaError_t tomoSumToErr(const T *d_in,
//                          T *d_out,
//                          size_t const *in_shape,
//                          size_t in_shape_len,
//                          size_t const *out_shape,
//                          size_t out_shape_len,
//                          size_t const *in_strides,
//                          size_t in_strides_len,
//                          size_t const *out_strides,
//                          size_t out_strides_len,
//                          size_t in_size,
//                          size_t out_size,
//                          size_t nd,
//                          cudaStream_t stream)
// {

//     try
//     {
//         tomoSumTo<T>(d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len, in_strides, in_strides_len, out_strides, out_strides_len, in_size, out_size, nd, stream);
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

#define MAX_ND 32
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) return err; \
} while(0)

template <typename T>
cudaError_t tomoBroadcastTo(
    const T *d_in, T *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t in_size, size_t out_size, size_t nd,
    cudaStream_t stream) {
    // Validate inputs

    // Device buffers (could be cached in a real system)
    size_t *d_in_shape, *d_out_shape, *d_in_stride;
    CHECK_CUDA(cudaMallocAsync(&d_in_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_in_stride, nd * sizeof(size_t), stream));

    CHECK_CUDA(cudaMemcpyAsync(d_in_shape, in_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_shape, out_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_in_stride, in_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    auto first = thrust::counting_iterator<size_t>{0};
    auto last = thrust::counting_iterator<size_t>{out_size};

    thrust::transform(thrust::cuda::par_nosync.on(stream),
                  thrust::counting_iterator<size_t>{0},
                  thrust::counting_iterator<size_t>{out_size},
                  thrust::device_pointer_cast(d_out),
                  [=] __device__(size_t out_index) {
                      size_t coords[MAX_ND];
                      size_t tmp = out_index;
                      for (ptrdiff_t d = nd - 1; d >= 0; --d) {
                          size_t dim_size = d_out_shape[d];
                          coords[d] = tmp % dim_size;
                          tmp /= dim_size;
                      }
                      size_t in_index = 0;
                      for (size_t d = 0; d < nd; ++d) {
                          size_t c;
                          if (d_in_shape[d] == 1) {
                              c = 0;  // Broadcasting
                          } else if (d_out_shape[d] % d_in_shape[d] == 0) {
                              c = coords[d] % d_in_shape[d];  // Tiling
                          } else {
                              c = coords[d];  // Direct mapping (if shapes match)
                          }
                          in_index = in_index + c * d_in_stride[d];
                      }
                      return d_in[in_index];
                  });

    CHECK_CUDA(cudaFreeAsync(d_in_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_out_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_in_stride, stream));
    return cudaSuccess;
}

template <typename T>
cudaError_t tomoBroadcastToErr(const T *d_in,
                               T *d_out,
                               size_t const *in_shape,
                               size_t in_shape_len,
                               size_t const *out_shape,
                               size_t out_shape_len,
                               size_t const *in_strides,
                               size_t in_strides_len,
                               size_t in_size,
                               size_t out_size,
                               size_t nd,
                               cudaStream_t stream)
{

    try
    {
        tomoBroadcastTo<T>(d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len, in_strides, in_strides_len, in_size, out_size, nd, stream);
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

// Custom reduction kernel for tomoSumTo
template <typename T>
__global__ void tomoSumToKernel(
    const T *d_in, T *d_out,
    const size_t *d_in_shape, const size_t *d_out_shape,
    const size_t *d_in_strides, const size_t *d_out_strides,
    size_t in_size, size_t out_size, size_t nd) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_size) return;

    // Unravel out_idx to out_coords
    size_t out_coords[MAX_ND];
    size_t tmp = out_idx;
    for (ptrdiff_t d = nd - 1; d >= 0; --d) {
        size_t dim_size = d_out_shape[d];
        out_coords[d] = tmp % dim_size;
        tmp /= dim_size;
    }

    // Compute sum over input elements mapping to this output index
    T sum = T(0);
    size_t in_coords[MAX_ND];
    for (size_t in_idx = 0; in_idx < in_size; ++in_idx) {
        size_t unravel = in_idx;
        bool matches = true;
        for (ptrdiff_t d = nd - 1; d >= 0; --d) {
            size_t dim_size = d_in_shape[d];
            in_coords[d] = unravel % dim_size;
            unravel /= dim_size;
            if (d_out_shape[d] != 1) {
                size_t out_c = out_coords[d];
                size_t in_c = (d_in_shape[d] == 1) ? 0 : in_coords[d];
                if (out_c != in_c) {
                    matches = false;
                    break;
                }
            }
        }
        if (matches) {
            size_t in_offset = 0;
            for (size_t d = 0; d < nd; ++d) {
                in_offset = in_offset + in_coords[d] * d_in_strides[d];
            }
            sum = sum + d_in[in_offset];
        }
    }
    d_out[out_idx] = sum;
}

template <typename T>
cudaError_t tomoSumTo(
    const T *d_in, T *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size, size_t out_size, size_t nd,
    cudaStream_t stream) {
    // Validate inputs
    if (out_strides_len != nd) return cudaErrorInvalidValue;

    // Device buffers
    size_t *d_in_shape, *d_out_shape, *d_in_strides, *d_out_strides;
    CHECK_CUDA(cudaMallocAsync(&d_in_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_in_strides, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_strides, nd * sizeof(size_t), stream));

    CHECK_CUDA(cudaMemcpyAsync(d_in_shape, in_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_shape, out_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_in_strides, in_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_strides, out_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    // Launch kernel
    const int threads = 256;
    const int blocks = ((int)out_size + threads - 1) / threads;
    tomoSumToKernel<<<blocks, threads, 0, stream>>>(
        d_in, d_out, d_in_shape, d_out_shape, d_in_strides, d_out_strides,
        in_size, out_size, nd);

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFreeAsync(d_in_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_out_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_in_strides, stream));
    CHECK_CUDA(cudaFreeAsync(d_out_strides, stream));
    return cudaSuccess;
}

template <typename T>
cudaError_t tomoSumToErr(const T *d_in,
                         T *d_out,
                         size_t const *in_shape,
                         size_t in_shape_len,
                         size_t const *out_shape,
                         size_t out_shape_len,
                         size_t const *in_strides,
                         size_t in_strides_len,
                         size_t const *out_strides,
                         size_t out_strides_len,
                         size_t in_size,
                         size_t out_size,
                         size_t nd,
                         cudaStream_t stream)
{

    try
    {
        tomoSumTo<T>(d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len, in_strides, in_strides_len, out_strides, out_strides_len, in_size, out_size, nd, stream);
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
    cudaStream_t stream)
{
    return tomoBroadcastToErr<__half_raw>(
        d_in,
        d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        in_size,
        out_size,
        nd,
        stream);
}

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
    cudaStream_t stream)
{
    return tomoBroadcastToErr<__nv_bfloat16_raw>(
        d_in,
        d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        in_size,
        out_size,
        nd,
        stream);
}

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
    cudaStream_t stream)
{
    return tomoBroadcastToErr<float>(
        d_in,
        d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        in_size,
        out_size,
        nd,
        stream);
}

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
    cudaStream_t stream)
{
    return tomoBroadcastToErr<double>(
        d_in,
        d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        in_size,
        out_size,
        nd,
        stream);
}

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
    cudaStream_t stream)
{
    return tomoSumToErr<__half_raw>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        in_size,
        out_size,
        nd,
        stream);
}

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
    cudaStream_t stream)
{
    return tomoSumToErr<__nv_bfloat16_raw>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        in_size,
        out_size,
        nd,
        stream);
}

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
    cudaStream_t stream)
{
    return tomoSumToErr<float>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        in_size,
        out_size,
        nd,
        stream);
}

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
    cudaStream_t stream)
{
    return tomoSumToErr<double>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        in_size,
        out_size,
        nd,
        stream);
}
