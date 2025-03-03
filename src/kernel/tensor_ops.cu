#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"

#include "tensor_ops.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

template <typename T>
void tomoBroadcastTo(const T *d_in,
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
    // Copy shape/strides to device arrays
    auto d_in_shape = thrust::device_vector<size_t>{in_shape, in_shape + in_shape_len};
    auto d_out_shape = thrust::device_vector<size_t>{out_shape, out_shape + out_shape_len};
    auto d_in_stride = thrust::device_vector<size_t>{in_strides, in_strides + in_strides_len};

    // Create counting iterators
    auto first = thrust::counting_iterator<size_t>{0};
    auto last = thrust::counting_iterator<size_t>{out_size};

    // We need device copies of shape/stride
    auto in_sh = thrust::raw_pointer_cast(d_in_shape.data());
    auto out_sh = thrust::raw_pointer_cast(d_out_shape.data());
    auto in_str = thrust::raw_pointer_cast(d_in_stride.data());

    // The device lambda (broadcast logic).
    // We'll capture all by [=], but be mindful that older NVCC
    // may require you to pass certain pointers explicitly.
    thrust::transform(thrust::cuda::par_nosync.on(stream),
                      first, last,
                      thrust::device_pointer_cast(d_out),
                      [=] __device__(size_t out_index)
                      {
                          // 1) Unravel outIndex -> outCoords
                          //    row-major from the back
                          int coords[32]; // if nd <= 32 for demo
                          auto tmp = out_index;
                          for (auto d = (ptrdiff_t)nd - 1; d >= 0; d--)
                          {
                              auto dim_size = out_sh[d];
                              coords[d] = tmp % dim_size;
                              tmp /= dim_size;
                          }

                          // 2) Build inIndex by mapping coords. If inSh[d] == 1, clamp to 0
                          auto in_index = (size_t)0;
                          for (auto d = (size_t)0; d < nd; d++)
                          {
                              auto c = (in_sh[d] == 1) ? (size_t)0 : coords[d];
                              in_index += c * in_str[d];
                          }

                          // 3) Return in_data[inIndex]
                          return d_in[in_index];
                      });
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

template <typename T>
void tomoSumTo(const T *d_in,
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
    // Copy shape/stride info to device
    auto d_in_shape = thrust::device_vector<size_t>{in_shape, in_shape + in_shape_len};
    auto d_out_shape = thrust::device_vector<size_t>{out_shape, out_shape + out_shape_len};
    auto d_in_stride = thrust::device_vector<size_t>{in_strides, in_strides + in_strides_len};
    auto d_out_stride = thrust::device_vector<size_t>{out_strides, out_strides + out_strides_len};

    // 1) Build a device_vector of "keys" for each input index
    auto keys = thrust::device_vector<size_t>(in_size);

    // We'll need shape/stride from device memory
    auto inSh = thrust::raw_pointer_cast(d_in_shape.data());
    auto outSh = thrust::raw_pointer_cast(d_out_shape.data());
    auto inStr = thrust::raw_pointer_cast(d_in_stride.data());
    auto outStr = thrust::raw_pointer_cast(d_out_stride.data());

    // The lambda that computes the key from inIndex
    thrust::transform(thrust::cuda::par_nosync.on(stream),
                      thrust::make_counting_iterator<size_t>(0),
                      thrust::make_counting_iterator<size_t>(in_size),
                      keys.begin(),
                      [=] __device__(size_t inIndex)
                      {
                                                    // decode inIndex -> coords
                          size_t coords[32];
                          auto tmp = inIndex;
                          for (auto d = (ptrdiff_t)nd - 1; d >= 0; d--)
                          {
                              size_t dimSize = inSh[d];
                              coords[d] = tmp % dimSize;
                              tmp /= dimSize;
                          }

                          // clamp coords if out_shape[d] == 1 => coords[d] = 0
                          size_t outCoords[32];
                          for (auto d = (size_t)0; d < nd; d++)
                          {
                              outCoords[d] = (outSh[d] == 1) ? 0 : coords[d];
                          }

                          // re-linearize outCoords => outIndex
                          auto outIndex = (size_t)0;
                          for (auto d = (size_t)0; d < nd; d++)
                          {
                              outIndex += outCoords[d] * outStr[d];
                          }
                          return outIndex;
                      });

    // 2) Sort by key while carrying values
    //    The values are just d_in[inIndex].
    auto vals = thrust::device_vector<T>(in_size);
    thrust::copy_n(thrust::cuda::par_nosync.on(stream), thrust::device_pointer_cast(d_in), in_size, vals.begin());

    thrust::sort_by_key(thrust::cuda::par_nosync.on(stream), keys.begin(), keys.end(), vals.begin());

    // 3) reduce_by_key
    auto out_keys = thrust::device_vector<size_t>(in_size);
    auto out_vals = thrust::device_vector<T>(in_size);

    auto new_end = thrust::reduce_by_key(thrust::cuda::par_nosync.on(stream),
                                         keys.begin(), keys.end(),
                                         vals.begin(),
                                         out_keys.begin(),
                                         out_vals.begin());

    auto num_unique = new_end.first - out_keys.begin();

    // 4) Scatter results into d_out
    //    out_keys[i] = some index in [0..out_size),
    //    out_vals[i] = sum of all input mapped to that key.

    // Real approach: we can get raw pointers:
    auto out_keys_ptr = thrust::raw_pointer_cast(out_keys.data());
    auto out_vals_ptr = thrust::raw_pointer_cast(out_vals.data());
    auto d_out_ptr = d_out; // from outer scope

    // Overwrite with sums (or accumulate)
    thrust::for_each(thrust::cuda::par_nosync.on(stream),
                     thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(num_unique),
                     [=] __device__(size_t i)
                     {
                         auto k = out_keys_ptr[i];
                         auto v = out_vals_ptr[i];
                         d_out_ptr[k] = v;
                     });
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
