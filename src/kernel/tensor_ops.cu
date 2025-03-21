#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"

#include "tensor_ops.h"

#include <limits>
#include <algorithm>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include "float_op.cuh"

#define MAX_ND 32

#define CHECK_CUDA(call)        \
    do                          \
    {                           \
        cudaError_t err = call; \
        if (err != cudaSuccess) \
            return err;         \
    } while (0)

template <typename T>
cudaError_t tomoBroadcastTo(
    const T *d_in, T *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t in_size, size_t out_size, size_t nd,
    cudaStream_t stream)
{
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
                      [=] __device__(size_t out_index)
                      {
                          size_t coords[MAX_ND];
                          size_t tmp = out_index;
                          for (ptrdiff_t d = nd - 1; d >= 0; --d)
                          {
                              size_t dim_size = d_out_shape[d];
                              coords[d] = tmp % dim_size;
                              tmp /= dim_size;
                          }
                          size_t in_index = 0;
                          for (size_t d = 0; d < nd; ++d)
                          {
                              size_t c;
                              if (d_in_shape[d] == 1)
                              {
                                  c = 0; // Broadcasting
                              }
                              else if (d_out_shape[d] % d_in_shape[d] == 0)
                              {
                                  c = coords[d] % d_in_shape[d]; // Tiling
                              }
                              else
                              {
                                  c = coords[d]; // Direct mapping (if shapes match)
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

// // Custom reduction kernel for tomoSumTo
// template <typename T>
// __global__ void tomoSumToKernel(
//     const T *d_in, T *d_out,
//     const size_t *d_in_shape, const size_t *d_out_shape,
//     const size_t *d_in_strides, const size_t *d_out_strides,
//     size_t in_size, size_t out_size, size_t nd)
// {
//     size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (out_idx >= out_size)
//         return;

//     // Unravel out_idx to out_coords
//     size_t out_coords[MAX_ND];
//     size_t tmp = out_idx;
//     for (ptrdiff_t d = nd - 1; d >= 0; --d)
//     {
//         size_t dim_size = d_out_shape[d];
//         out_coords[d] = tmp % dim_size;
//         tmp /= dim_size;
//     }

//     // Compute sum over input elements mapping to this output index
//     T sum = T(0);
//     size_t in_coords[MAX_ND];
//     for (size_t in_idx = 0; in_idx < in_size; ++in_idx)
//     {
//         size_t unravel = in_idx;
//         bool matches = true;
//         for (ptrdiff_t d = nd - 1; d >= 0; --d)
//         {
//             size_t dim_size = d_in_shape[d];
//             in_coords[d] = unravel % dim_size;
//             unravel /= dim_size;
//             if (d_out_shape[d] != 1)
//             {
//                 size_t out_c = out_coords[d];
//                 size_t in_c = (d_in_shape[d] == 1) ? 0 : in_coords[d];
//                 if (out_c != in_c)
//                 {
//                     matches = false;
//                     break;
//                 }
//             }
//         }
//         if (matches)
//         {
//             size_t in_offset = 0;
//             for (size_t d = 0; d < nd; ++d)
//             {
//                 in_offset = in_offset + in_coords[d] * d_in_strides[d];
//             }
//             sum = sum + d_in[in_offset];
//         }
//     }
//     d_out[out_idx] = sum;
// }

// template <typename T>
// __global__ void tomoSumToKernel(
//     const T* d_in,            // Input tensor data
//     const size_t* in_shape,   // Input tensor shape
//     const size_t* in_strides, // Input tensor strides
//     T* d_out,                 // Output tensor data
//     const size_t* out_shape,  // Output tensor shape
//     const size_t* out_strides,// Output tensor strides
//     size_t out_size,          // Total number of output elements
//     size_t nd                 // Number of dimensions
// ) {
//     // Maximum supported dimensions
//     //const size_t MAX_DIMS = 10;

//     // Each block handles one output element
//     size_t out_idx = blockIdx.x;
//     if (out_idx >= out_size) return;

//     // Compute output coordinates from out_idx
//     size_t out_coords[MAX_ND];
//     size_t tmp = out_idx;
//     for (ptrdiff_t d = nd - 1; d >= 0; --d) {
//         out_coords[d] = tmp % out_shape[d];
//         tmp /= out_shape[d];
//     }

//     // Compute output offset using out_strides
//     size_t out_offset = 0;
//     for (size_t d = 0; d < nd; ++d) {
//         out_offset += out_coords[d] * out_strides[d];
//     }

//     // Compute base offset in the input tensor
//     size_t base_offset = 0;
//     for (size_t d = 0; d < nd; ++d) {
//         size_t in_c = (out_shape[d] == 1) ? 0 : out_coords[d];
//         base_offset += in_c * in_strides[d];
//     }

//     // Identify reduced dimensions
//     size_t reduced_dims[MAX_ND];
//     size_t reduced_sizes[MAX_ND];
//     size_t num_reduced = 0;
//     for (size_t d = 0; d < nd; ++d) {
//         if (out_shape[d] == 1 && in_shape[d] > 1) {
//             reduced_dims[num_reduced] = d;
//             reduced_sizes[num_reduced] = in_shape[d];
//             num_reduced++;
//         }
//     }

//     // Compute total number of elements to sum
//     size_t N = 1;
//     for (size_t j = 0; j < num_reduced; ++j) {
//         N *= reduced_sizes[j];
//     }

//     // Each thread computes a partial sum
//     T partial_sum = (T)0.0;
//     for (size_t i = threadIdx.x; i < N; i += blockDim.x) {
//         // Compute coordinates in the reduced dimensions from flattened index i
//         size_t reduced_coords[MAX_ND];
//         size_t tmp_i = i;
//         for (ptrdiff_t j = num_reduced - 1; j >= 0; --j) {
//             reduced_coords[j] = tmp_i % reduced_sizes[j];
//             tmp_i /= reduced_sizes[j];
//         }

//         // Compute input offset
//         size_t offset = base_offset;
//         for (size_t j = 0; j < num_reduced; ++j) {
//             size_t d = reduced_dims[j];
//             offset += reduced_coords[j] * in_strides[d];
//         }

//         // Accumulate into partial sum
//         partial_sum = partial_sum + d_in[offset];
//     }

//     // Use shared memory for reduction within the block
//     extern __shared__ T shared_sums[];
//     shared_sums[threadIdx.x] = partial_sum;
//     __syncthreads();

//     // Perform parallel reduction in shared memory
//     // Assumes blockDim.x is a power of two
//     for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (threadIdx.x < s) {
//             shared_sums[threadIdx.x] = shared_sums[threadIdx.x] + shared_sums[threadIdx.x + s];
//         }
//         __syncthreads();
//     }

//     // Thread 0 writes the final sum to the output
//     if (threadIdx.x == 0) {
//         d_out[out_offset] = shared_sums[0];
//     }
// }

__global__ void tomoSumToKernel(
    const __half_raw *d_in,    // Input tensor data
    const size_t *in_shape,    // Input tensor shape
    const size_t *in_strides,  // Input tensor strides
    __half_raw *d_out,         // Output tensor data
    const size_t *out_shape,   // Output tensor shape
    const size_t *out_strides, // Output tensor strides
    size_t out_size,           // Total number of output elements
    size_t nd                  // Number of dimensions
)
{
    // Maximum supported dimensions
    // const size_t MAX_DIMS = 10;

    // Each block handles one output element
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size)
        return;

    // Compute output coordinates from out_idx
    size_t out_coords[MAX_ND];
    size_t tmp = out_idx;
    for (ptrdiff_t d = nd - 1; d >= 0; --d)
    {
        out_coords[d] = tmp % out_shape[d];
        tmp /= out_shape[d];
    }

    // Compute output offset using out_strides
    size_t out_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        out_offset += out_coords[d] * out_strides[d];
    }

    // Compute base offset in the input tensor
    size_t base_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        size_t in_c = (out_shape[d] == 1) ? 0 : out_coords[d];
        base_offset += in_c * in_strides[d];
    }

    // Identify reduced dimensions
    size_t reduced_dims[MAX_ND];
    size_t reduced_sizes[MAX_ND];
    size_t num_reduced = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        if (out_shape[d] == 1 && in_shape[d] > 1)
        {
            reduced_dims[num_reduced] = d;
            reduced_sizes[num_reduced] = in_shape[d];
            num_reduced++;
        }
    }

    // Compute total number of elements to sum
    size_t N = 1;
    for (size_t j = 0; j < num_reduced; ++j)
    {
        N *= reduced_sizes[j];
    }

    // Each thread computes a partial sum
    __half_raw partial_sum = (__half_raw)0.0;
    for (size_t i = threadIdx.x; i < N; i += blockDim.x)
    {
        // Compute coordinates in the reduced dimensions from flattened index i
        size_t reduced_coords[MAX_ND];
        size_t tmp_i = i;
        for (ptrdiff_t j = num_reduced - 1; j >= 0; --j)
        {
            reduced_coords[j] = tmp_i % reduced_sizes[j];
            tmp_i /= reduced_sizes[j];
        }

        // Compute input offset
        size_t offset = base_offset;
        for (size_t j = 0; j < num_reduced; ++j)
        {
            size_t d = reduced_dims[j];
            offset += reduced_coords[j] * in_strides[d];
        }

        // Accumulate into partial sum
        partial_sum = partial_sum + d_in[offset];
    }

    // Use shared memory for reduction within the block
    extern __shared__ __half_raw shared_sumsh[];
    shared_sumsh[threadIdx.x] = partial_sum;
    __syncthreads();

    // Perform parallel reduction in shared memory
    // Assumes blockDim.x is a power of two
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            shared_sumsh[threadIdx.x] = shared_sumsh[threadIdx.x] + shared_sumsh[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final sum to the output
    if (threadIdx.x == 0)
    {
        d_out[out_offset] = shared_sumsh[0];
    }
}

__global__ void tomoSumToKernel(
    const __nv_bfloat16_raw *d_in, // Input tensor data
    const size_t *in_shape,        // Input tensor shape
    const size_t *in_strides,      // Input tensor strides
    __nv_bfloat16_raw *d_out,      // Output tensor data
    const size_t *out_shape,       // Output tensor shape
    const size_t *out_strides,     // Output tensor strides
    size_t out_size,               // Total number of output elements
    size_t nd                      // Number of dimensions
)
{
    // Maximum supported dimensions
    // const size_t MAX_DIMS = 10;

    // Each block handles one output element
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size)
        return;

    // Compute output coordinates from out_idx
    size_t out_coords[MAX_ND];
    size_t tmp = out_idx;
    for (ptrdiff_t d = nd - 1; d >= 0; --d)
    {
        out_coords[d] = tmp % out_shape[d];
        tmp /= out_shape[d];
    }

    // Compute output offset using out_strides
    size_t out_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        out_offset += out_coords[d] * out_strides[d];
    }

    // Compute base offset in the input tensor
    size_t base_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        size_t in_c = (out_shape[d] == 1) ? 0 : out_coords[d];
        base_offset += in_c * in_strides[d];
    }

    // Identify reduced dimensions
    size_t reduced_dims[MAX_ND];
    size_t reduced_sizes[MAX_ND];
    size_t num_reduced = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        if (out_shape[d] == 1 && in_shape[d] > 1)
        {
            reduced_dims[num_reduced] = d;
            reduced_sizes[num_reduced] = in_shape[d];
            num_reduced++;
        }
    }

    // Compute total number of elements to sum
    size_t N = 1;
    for (size_t j = 0; j < num_reduced; ++j)
    {
        N *= reduced_sizes[j];
    }

    // Each thread computes a partial sum
    __nv_bfloat16_raw partial_sum = (__nv_bfloat16_raw)0.0;
    for (size_t i = threadIdx.x; i < N; i += blockDim.x)
    {
        // Compute coordinates in the reduced dimensions from flattened index i
        size_t reduced_coords[MAX_ND];
        size_t tmp_i = i;
        for (ptrdiff_t j = num_reduced - 1; j >= 0; --j)
        {
            reduced_coords[j] = tmp_i % reduced_sizes[j];
            tmp_i /= reduced_sizes[j];
        }

        // Compute input offset
        size_t offset = base_offset;
        for (size_t j = 0; j < num_reduced; ++j)
        {
            size_t d = reduced_dims[j];
            offset += reduced_coords[j] * in_strides[d];
        }

        // Accumulate into partial sum
        partial_sum = partial_sum + d_in[offset];
    }

    // Use shared memory for reduction within the block
    extern __shared__ __nv_bfloat16_raw shared_sumsb[];
    shared_sumsb[threadIdx.x] = partial_sum;
    __syncthreads();

    // Perform parallel reduction in shared memory
    // Assumes blockDim.x is a power of two
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            shared_sumsb[threadIdx.x] = shared_sumsb[threadIdx.x] + shared_sumsb[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final sum to the output
    if (threadIdx.x == 0)
    {
        d_out[out_offset] = shared_sumsb[0];
    }
}

__global__ void tomoSumToKernel(
    const float *d_in,         // Input tensor data
    const size_t *in_shape,    // Input tensor shape
    const size_t *in_strides,  // Input tensor strides
    float *d_out,              // Output tensor data
    const size_t *out_shape,   // Output tensor shape
    const size_t *out_strides, // Output tensor strides
    size_t out_size,           // Total number of output elements
    size_t nd                  // Number of dimensions
)
{
    // Maximum supported dimensions
    // const size_t MAX_DIMS = 10;

    // Each block handles one output element
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size)
        return;

    // Compute output coordinates from out_idx
    size_t out_coords[MAX_ND];
    size_t tmp = out_idx;
    for (ptrdiff_t d = nd - 1; d >= 0; --d)
    {
        out_coords[d] = tmp % out_shape[d];
        tmp /= out_shape[d];
    }

    // Compute output offset using out_strides
    size_t out_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        out_offset += out_coords[d] * out_strides[d];
    }

    // Compute base offset in the input tensor
    size_t base_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        size_t in_c = (out_shape[d] == 1) ? 0 : out_coords[d];
        base_offset += in_c * in_strides[d];
    }

    // Identify reduced dimensions
    size_t reduced_dims[MAX_ND];
    size_t reduced_sizes[MAX_ND];
    size_t num_reduced = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        if (out_shape[d] == 1 && in_shape[d] > 1)
        {
            reduced_dims[num_reduced] = d;
            reduced_sizes[num_reduced] = in_shape[d];
            num_reduced++;
        }
    }

    // Compute total number of elements to sum
    size_t N = 1;
    for (size_t j = 0; j < num_reduced; ++j)
    {
        N *= reduced_sizes[j];
    }

    // Each thread computes a partial sum
    float partial_sum = (float)0.0;
    for (size_t i = threadIdx.x; i < N; i += blockDim.x)
    {
        // Compute coordinates in the reduced dimensions from flattened index i
        size_t reduced_coords[MAX_ND];
        size_t tmp_i = i;
        for (ptrdiff_t j = num_reduced - 1; j >= 0; --j)
        {
            reduced_coords[j] = tmp_i % reduced_sizes[j];
            tmp_i /= reduced_sizes[j];
        }

        // Compute input offset
        size_t offset = base_offset;
        for (size_t j = 0; j < num_reduced; ++j)
        {
            size_t d = reduced_dims[j];
            offset += reduced_coords[j] * in_strides[d];
        }

        // Accumulate into partial sum
        partial_sum = partial_sum + d_in[offset];
    }

    // Use shared memory for reduction within the block
    extern __shared__ float shared_sumsf[];
    shared_sumsf[threadIdx.x] = partial_sum;
    __syncthreads();

    // Perform parallel reduction in shared memory
    // Assumes blockDim.x is a power of two
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            shared_sumsf[threadIdx.x] = shared_sumsf[threadIdx.x] + shared_sumsf[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final sum to the output
    if (threadIdx.x == 0)
    {
        d_out[out_offset] = shared_sumsf[0];
    }
}

__global__ void tomoSumToKernel(
    const double *d_in,        // Input tensor data
    const size_t *in_shape,    // Input tensor shape
    const size_t *in_strides,  // Input tensor strides
    double *d_out,             // Output tensor data
    const size_t *out_shape,   // Output tensor shape
    const size_t *out_strides, // Output tensor strides
    size_t out_size,           // Total number of output elements
    size_t nd                  // Number of dimensions
)
{
    // Maximum supported dimensions
    // const size_t MAX_DIMS = 10;

    // Each block handles one output element
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size)
        return;

    // Compute output coordinates from out_idx
    size_t out_coords[MAX_ND];
    size_t tmp = out_idx;
    for (ptrdiff_t d = nd - 1; d >= 0; --d)
    {
        out_coords[d] = tmp % out_shape[d];
        tmp /= out_shape[d];
    }

    // Compute output offset using out_strides
    size_t out_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        out_offset += out_coords[d] * out_strides[d];
    }

    // Compute base offset in the input tensor
    size_t base_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        size_t in_c = (out_shape[d] == 1) ? 0 : out_coords[d];
        base_offset += in_c * in_strides[d];
    }

    // Identify reduced dimensions
    size_t reduced_dims[MAX_ND];
    size_t reduced_sizes[MAX_ND];
    size_t num_reduced = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        if (out_shape[d] == 1 && in_shape[d] > 1)
        {
            reduced_dims[num_reduced] = d;
            reduced_sizes[num_reduced] = in_shape[d];
            num_reduced++;
        }
    }

    // Compute total number of elements to sum
    size_t N = 1;
    for (size_t j = 0; j < num_reduced; ++j)
    {
        N *= reduced_sizes[j];
    }

    // Each thread computes a partial sum
    double partial_sum = (double)0.0;
    for (size_t i = threadIdx.x; i < N; i += blockDim.x)
    {
        // Compute coordinates in the reduced dimensions from flattened index i
        size_t reduced_coords[MAX_ND];
        size_t tmp_i = i;
        for (ptrdiff_t j = num_reduced - 1; j >= 0; --j)
        {
            reduced_coords[j] = tmp_i % reduced_sizes[j];
            tmp_i /= reduced_sizes[j];
        }

        // Compute input offset
        size_t offset = base_offset;
        for (size_t j = 0; j < num_reduced; ++j)
        {
            size_t d = reduced_dims[j];
            offset += reduced_coords[j] * in_strides[d];
        }

        // Accumulate into partial sum
        partial_sum = partial_sum + d_in[offset];
    }

    // Use shared memory for reduction within the block
    extern __shared__ double shared_sumsd[];
    shared_sumsd[threadIdx.x] = partial_sum;
    __syncthreads();

    // Perform parallel reduction in shared memory
    // Assumes blockDim.x is a power of two
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            shared_sumsd[threadIdx.x] = shared_sumsd[threadIdx.x] + shared_sumsd[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final sum to the output
    if (threadIdx.x == 0)
    {
        d_out[out_offset] = shared_sumsd[0];
    }
}

template <typename T>
cudaError_t tomoSumTo(
    const T *d_in,             // Input tensor data on device
    T *d_out,                  // Output tensor data on device
    size_t const *in_shape,    // Input tensor shape on host
    size_t in_shape_len,       // Length of in_shape
    size_t const *out_shape,   // Output tensor shape on host
    size_t out_shape_len,      // Length of out_shape
    size_t const *in_strides,  // Input tensor strides on host
    size_t in_strides_len,     // Length of in_strides
    size_t const *out_strides, // Output tensor strides on host
    size_t out_strides_len,    // Length of out_strides
    size_t in_size,            // Total size of input (unused)
    size_t out_size,           // Total size of output
    size_t nd,                 // Number of dimensions
    cudaStream_t stream        // CUDA stream for asynchronous execution
)
{
    // Maximum supported dimensions
    const size_t MAX_DIMS = 10;

    // Validate inputs
    if (nd > MAX_DIMS || in_shape_len != nd || out_shape_len != nd ||
        in_strides_len != nd || out_strides_len != nd)
    {
        return cudaErrorInvalidValue;
    }

    // Device buffers for shapes and strides
    size_t *d_in_shape, *d_out_shape, *d_in_strides, *d_out_strides;
    CHECK_CUDA(cudaMallocAsync(&d_in_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_in_strides, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_strides, nd * sizeof(size_t), stream));

    // Copy shapes and strides from host to device
    CHECK_CUDA(cudaMemcpyAsync(d_in_shape, in_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_shape, out_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_in_strides, in_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_strides, out_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    // Launch kernel
    const int threads = 256;          // Must be a power of two for reduction
    const int blocks = (int)out_size; // One block per output element
    const int shared_mem_size = (int)threads * (int)sizeof(T);
    tomoSumToKernel<<<blocks, threads, shared_mem_size, stream>>>(
        d_in, d_in_shape, d_in_strides, d_out, d_out_shape, d_out_strides, out_size, nd);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Free device memory
    CHECK_CUDA(cudaFreeAsync(d_in_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_out_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_in_strides, stream));
    CHECK_CUDA(cudaFreeAsync(d_out_strides, stream));

    return cudaSuccess;
}

#define BLOCK_SIZE 16

__global__ void tomoLinearKernelB(__nv_bfloat16_raw const *A, __nv_bfloat16_raw const *B, size_t M, size_t K, size_t N, __nv_bfloat16_raw const *bias, __nv_bfloat16_raw *C)
{
    auto row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    auto col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    auto sum = (__nv_bfloat16_raw)0.0;
    for (auto k = (size_t)0; k < K; k += BLOCK_SIZE)
    {
        __shared__ __nv_bfloat16_raw shared_a_b[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ __nv_bfloat16_raw shared_b_b[BLOCK_SIZE][BLOCK_SIZE];

        // Load Ashared
        if (row < M && k + threadIdx.x < K)
        {
            shared_a_b[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        }
        else
        {
            shared_a_b[threadIdx.y][threadIdx.x] = (__nv_bfloat16_raw)0.0;
        }

        // Load Bshared
        if (k + threadIdx.y < K && col < N)
        {
            shared_b_b[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        }
        else
        {
            shared_b_b[threadIdx.y][threadIdx.x] = (__nv_bfloat16_raw)0.0;
        }

        __syncthreads();

        for (auto m = (size_t)0; m < BLOCK_SIZE; m++)
        {
            if (k + m < K)
            {
                sum = sum + shared_a_b[threadIdx.y][m] * shared_b_b[m][threadIdx.x];
            }
        }

        __syncthreads();
    }
    if (row < M && col < N)
    {
        C[row * N + col] = sum;

        if (bias != NULL)
        {
            C[row * N + col] = C[row * N + col] + bias[row * N + col];
        }
    }
}

__global__ void tomoLinearKernelH(__half_raw const *A, __half_raw const *B, size_t M, size_t K, size_t N, __half_raw const *bias, __half_raw *C)
{
    auto row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    auto col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    auto sum = (__half_raw)0.0;
    for (auto k = (size_t)0; k < K; k += BLOCK_SIZE)
    {
        __shared__ __half_raw shared_a_h[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ __half_raw shared_b_h[BLOCK_SIZE][BLOCK_SIZE];

        // Load Ashared
        if (row < M && k + threadIdx.x < K)
        {
            shared_a_h[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        }
        else
        {
            shared_a_h[threadIdx.y][threadIdx.x] = (__half_raw)0.0;
        }

        // Load Bshared
        if (k + threadIdx.y < K && col < N)
        {
            shared_b_h[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        }
        else
        {
            shared_b_h[threadIdx.y][threadIdx.x] = (__half_raw)0.0;
        }

        __syncthreads();

        for (auto m = (size_t)0; m < BLOCK_SIZE; m++)
        {
            if (k + m < K)
            {
                sum = sum + shared_a_h[threadIdx.y][m] * shared_b_h[m][threadIdx.x];
            }
        }

        __syncthreads();
    }
    if (row < M && col < N)
    {
        C[row * N + col] = sum;

        if (bias != NULL)
        {
            C[row * N + col] = C[row * N + col] + bias[row * N + col];
        }
    }
}

__global__ void tomoLinearKernelF(float const *A, float const *B, size_t M, size_t K, size_t N, float const *bias, float *C)
{
    auto row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    auto col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    auto sum = 0.0f;
    for (auto k = (size_t)0; k < K; k += BLOCK_SIZE)
    {
        __shared__ float shared_a_f[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float shared_b_f[BLOCK_SIZE][BLOCK_SIZE];

        // Load Ashared
        if (row < M && k + threadIdx.x < K)
        {
            shared_a_f[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        }
        else
        {
            shared_a_f[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load Bshared
        if (k + threadIdx.y < K && col < N)
        {
            shared_b_f[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        }
        else
        {
            shared_b_f[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (auto m = (size_t)0; m < BLOCK_SIZE; m++)
        {
            if (k + m < K)
            {
                sum += shared_a_f[threadIdx.y][m] * shared_b_f[m][threadIdx.x];
            }
        }

        __syncthreads();
    }
    if (row < M && col < N)
    {
        C[row * N + col] = sum;

        if (bias != NULL)
        {
            C[row * N + col] += bias[row * N + col];
        }
    }
}

__global__ void tomoLinearKernelD(double const *A, double const *B, size_t M, size_t K, size_t N, double const *bias, double *C)
{
    auto row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    auto col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    auto sum = 0.0;
    for (auto k = (size_t)0; k < K; k += BLOCK_SIZE)
    {
        __shared__ double shared_a_d[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double shared_b_d[BLOCK_SIZE][BLOCK_SIZE];

        // Load Ashared
        if (row < M && k + threadIdx.x < K)
        {
            shared_a_d[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        }
        else
        {
            shared_a_d[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Load Bshared
        if (k + threadIdx.y < K && col < N)
        {
            shared_b_d[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        }
        else
        {
            shared_b_d[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (auto m = (size_t)0; m < BLOCK_SIZE; m++)
        {
            if (k + m < K)
            {
                sum += shared_a_d[threadIdx.y][m] * shared_b_d[m][threadIdx.x];
            }
        }

        __syncthreads();
    }
    if (row < M && col < N)
    {
        C[row * N + col] = sum;

        if (bias != NULL)
        {
            C[row * N + col] += bias[row * N + col];
        }
    }
}

template <typename T>
cudaError_t tomoLinear(T const *A, T const *B, size_t M, size_t K, size_t N, T const *bias, T *C, cudaStream_t stream)
{

    dim3 gridDim(((unsigned int)N + BLOCK_SIZE - 1) / BLOCK_SIZE, ((unsigned int)M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    if constexpr (std::is_same_v<T, __nv_bfloat16_raw>)
    {
        tomoLinearKernelB<<<gridDim, blockDim, 0, stream>>>(A, B, M, K, N, bias, C);
    }
    else if constexpr (std::is_same_v<T, __half_raw>)
    {
        tomoLinearKernelH<<<gridDim, blockDim, 0, stream>>>(A, B, M, K, N, bias, C);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        tomoLinearKernelF<<<gridDim, blockDim, 0, stream>>>(A, B, M, K, N, bias, C);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        tomoLinearKernelD<<<gridDim, blockDim, 0, stream>>>(A, B, M, K, N, bias, C);
    }

    return cudaGetLastError();
}

template <typename T>
struct TileDims
{
    // Default tile sizes for float, half, etc.
    static constexpr int BM = 64;
    static constexpr int BK = 32;
    static constexpr int BN = 64;
};

// Specialization for double: smaller BK to reduce shared memory usage
template <>
struct TileDims<double>
{
    static constexpr int BM = 64;
    static constexpr int BK = 16;
    static constexpr int BN = 64;
};
#define TM 4 // Sub-tile height per thread
#define TN 4 // Sub-tile width per thread

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// static __device__ inline half2 vload_half2(const __half_raw* ptr)
// {
//     // If your environment supports it, you can do direct pointer casts:
//     // return *(reinterpret_cast<const half2*>(ptr));
//     // Or an inline PTX approach. For brevity, let's just do:
//     return __halves2half2(ptr[0], ptr[1]);
// }

__global__ void tomoLinearKernelimpH(
    const __half_raw *__restrict__ A, // (M x K)
    const __half_raw *__restrict__ B, // (K x N)
    size_t M, size_t K, size_t N,
    const __half_raw *__restrict__ bias, // (optional, M x N)
    __half_raw *__restrict__ C           // (M x N)
)
{
    constexpr int BM = TileDims<__half_raw>::BM;
    constexpr int BK = TileDims<__half_raw>::BK;
    constexpr int BN = TileDims<__half_raw>::BN;

    // 2D index in the grid
    const int blockRow = blockIdx.y; // which block in dimension M
    const int blockCol = blockIdx.x; // which block in dimension N

    // 2D index in the block
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;

    // Global row/column for the block
    const int globalRow = blockRow * BM;
    const int globalCol = blockCol * BN;

    // Start offsets in shared memory
    // We'll keep two buffers for double‐buffering: As[2][BM][BK], Bs[2][BK][BN].
    __shared__ __half_raw As[2][BM][BK];
    __shared__ __half_raw Bs[2][BK][BN];

    // Each thread accumulates a sub‐tile of size TM x TN in registers
    // i.e. 4×4, or 4×8, etc.
    __half_raw sum[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
#pragma unroll
        for (int j = 0; j < TN; j++)
        {
            sum[i][j] = __half_raw(0.0f);
        }
    }

    // We’ll assume (BM * BK) and (BK * BN) are divisible by (blockDim.x * blockDim.y).
    // So each thread can copy multiple elements in a loop.

    // Compute how many tiles along K
    const int numTiles = (K + BK - 1) / BK; // assume K multiple of BK for simplicity

    // For double buffering, we keep track of the “read index” vs. the “compute index”
    int loadIdx = 0; // we load into this buffer
    int compIdx = 1; // we compute from this buffer

    // Local thread id if we want 1D
    int tid = threadRow * blockDim.x + threadCol;
    int tdim = blockDim.x * blockDim.y;

    //----------- FIRST TILE LOAD (Tile 0) -----------
    {
        // load A tile 0 into As[loadIdx], B tile 0 into Bs[loadIdx]
        // Each thread copies multiple elements from global to shared

        // A tile is (BM×BK)
        for (int idx = tid; idx < BM * BK; idx += tdim)
        {
            int a_row = idx / BK;       // local row
            int a_col = idx % BK;       // local col
            int ga = globalRow + a_row; // global row in A
            int ka = 0 + a_col;         // global col in A (tile 0 => offset=0)
            // Check boundary
            if (ga < M && ka < K)
            {
                As[loadIdx][a_row][a_col] = A[ga * K + ka];
            }
            else
            {
                As[loadIdx][a_row][a_col] = __half_raw(0.0f);
            }
        }
        // B tile is (BK×BN)
        for (int idx = tid; idx < BK * BN; idx += tdim)
        {
            int b_row = idx / BN;
            int b_col = idx % BN;
            int kb = 0 + b_row; // global row in B (tile 0 => offset=0)
            int gb = globalCol + b_col;
            if (kb < K && gb < N)
            {
                Bs[loadIdx][b_row][b_col] = B[kb * N + gb];
            }
            else
            {
                Bs[loadIdx][b_row][b_col] = __half_raw(0.0f);
            }
        }
    }
    __syncthreads();

    //----------- MAIN LOOP OVER TILES -----------
    for (int tile = 0; tile < numTiles; tile++)
    {
        // Swap loadIdx and compIdx so we compute on the tile that was loaded
        loadIdx = tile % 2;
        compIdx = (tile + 1) % 2;

        // Launch asynchronous load for the *next* tile if not the last tile
        if (tile < numTiles - 1)
        {
            int nextOffset = (tile + 1) * BK; // starting K for next tile

            // A tile in the next offset
            for (int idx = tid; idx < BM * BK; idx += tdim)
            {
                int a_row = idx / BK;
                int a_col = idx % BK;
                int ga = globalRow + a_row;
                int ka = nextOffset + a_col;
                if (ga < M && ka < K)
                {
                    As[compIdx][a_row][a_col] = A[ga * K + ka];
                }
                else
                {
                    As[compIdx][a_row][a_col] = __half_raw(0.0f);
                }
            }
            // B tile in the next offset
            for (int idx = tid; idx < BK * BN; idx += tdim)
            {
                int b_row = idx / BN;
                int b_col = idx % BN;
                int kb = nextOffset + b_row;
                int gb = globalCol + b_col;
                if (kb < K && gb < N)
                {
                    Bs[compIdx][b_row][b_col] = B[kb * N + gb];
                }
                else
                {
                    Bs[compIdx][b_row][b_col] = __half_raw(0.0f);
                }
            }
        }

        // We must __syncthreads() before we start computing on As[loadIdx], Bs[loadIdx]
        __syncthreads();

        //----- Multiply-Accumulate on the current tile "tile" from As[loadIdx], Bs[loadIdx] -----
        // Each tile is BK wide in K dimension
        // We can do partial unrolling (hand unroll the loop over BK).
        // For simplicity, we do #pragma unroll in a normal for loop:

#pragma unroll
        for (int m = 0; m < BK; m++)
        {
            // Each thread loads a tiny sub-vector from shared
            __half_raw aFrag[TM];
            __half_raw bFrag[TN];

#pragma unroll
            for (int i = 0; i < TM; i++)
            {
                int r = threadRow * TM + i; // local row in the BM dimension
                aFrag[i] = As[loadIdx][r][m];
            }
#pragma unroll
            for (int j = 0; j < TN; j++)
            {
                int c = threadCol * TN + j; // local col in the BN dimension
                bFrag[j] = Bs[loadIdx][m][c];
            }

// Now do the dot products: sum[i][j] += aFrag[i] * bFrag[j]
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
#pragma unroll
                for (int j = 0; j < TN; j++)
                {
                    sum[i][j] = __hfma(aFrag[i], bFrag[j], sum[i][j]);
                }
            }
        }
        __syncthreads(); // ensure we've finished compute before overwriting As/Bs in next iteration
    } // end for tile

//----------- Write the results to global memory -----------
// plus the bias if non-null
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int rowOut = globalRow + threadRow * TM + i;
        if (rowOut < M)
        {
#pragma unroll
            for (int j = 0; j < TN; j++)
            {
                int colOut = globalCol + threadCol * TN + j;
                if (colOut < N)
                {
                    size_t outIdx = rowOut * N + colOut;
                    __half_raw val = sum[i][j];
                    if (bias != nullptr)
                    {
                        val = __hadd(val, bias[outIdx]);
                    }
                    C[outIdx] = val;
                }
            }
        }
    }
}

__device__ inline __nv_bfloat16 bf16_fma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c)
{
    // Convert to float
    float fa = __bfloat162float(a);
    float fb = __bfloat162float(b);
    float fc = __bfloat162float(c);
    // Perform FMA in float
    float fres = fa * fb + fc;
    // Convert back
    return __float2bfloat16_rn(fres);
}

// The double‐buffered bfloat16 kernel
__global__ void tomoLinearKernelimpB(
    const __nv_bfloat16_raw *__restrict__ A, // (M x K)
    const __nv_bfloat16_raw *__restrict__ B, // (K x N)
    size_t M, size_t K, size_t N,
    const __nv_bfloat16_raw *__restrict__ bias, // (optional, M x N)
    __nv_bfloat16_raw *__restrict__ C           // (M x N)
)
{

    constexpr int BM = TileDims<__nv_bfloat16_raw>::BM;
    constexpr int BK = TileDims<__nv_bfloat16_raw>::BK;
    constexpr int BN = TileDims<__nv_bfloat16_raw>::BN;

    // 2D block index
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // 2D thread index
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;

    // Start row/col in global memory
    const int globalRow = blockRow * BM;
    const int globalCol = blockCol * BN;

    // Double‐buffered shared memory
    __shared__ __nv_bfloat16_raw As[2][BM][BK];
    __shared__ __nv_bfloat16_raw Bs[2][BK][BN];

    // Accumulator sub‐tile in registers (TM × TN)
    __nv_bfloat16_raw sum[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
#pragma unroll
        for (int j = 0; j < TN; j++)
        {
            sum[i][j] = (__nv_bfloat16_raw)0.0f;
        }
    }

    const int numTiles = (K + BK - 1) / BK;
    // 1D thread id for loading
    int tid = threadRow * blockDim.x + threadCol;
    int tdim = blockDim.x * blockDim.y;

    int loadIdx = 0;
    int compIdx = 1;

    //---------- Load first tile (tile=0) ----------
    {
        // A tile (BM×BK)
        for (int idx = tid; idx < BM * BK; idx += tdim)
        {
            int a_row = idx / BK;
            int a_col = idx % BK;
            int ga = globalRow + a_row;
            int ka = a_col;
            if (ga < M && ka < K)
            {
                As[loadIdx][a_row][a_col] = A[ga * K + ka];
            }
            else
            {
                As[loadIdx][a_row][a_col] = (__nv_bfloat16_raw)0.0f;
            }
        }
        // B tile (BK×BN)
        for (int idx = tid; idx < BK * BN; idx += tdim)
        {
            int b_row = idx / BN;
            int b_col = idx % BN;
            int kb = b_row;
            int gb = globalCol + b_col;
            if (kb < K && gb < N)
            {
                Bs[loadIdx][b_row][b_col] = B[kb * N + gb];
            }
            else
            {
                Bs[loadIdx][b_row][b_col] = (__nv_bfloat16_raw)0.0f;
            }
        }
    }
    __syncthreads();

    //---------- Main loop over K tiles ----------
    for (int tile = 0; tile < numTiles; tile++)
    {
        loadIdx = tile % 2;
        compIdx = (tile + 1) % 2;

        // Launch load for the next tile
        if (tile < numTiles - 1)
        {
            int nextOffset = (tile + 1) * BK;
            // Load As[compIdx], Bs[compIdx]
            for (int idx = tid; idx < BM * BK; idx += tdim)
            {
                int a_row = idx / BK;
                int a_col = idx % BK;
                int ga = globalRow + a_row;
                int ka = nextOffset + a_col;
                if (ga < M && ka < K)
                {
                    As[compIdx][a_row][a_col] = A[ga * K + ka];
                }
                else
                {
                    As[compIdx][a_row][a_col] = (__nv_bfloat16_raw)0.0f;
                }
            }
            for (int idx = tid; idx < BK * BN; idx += tdim)
            {
                int b_row = idx / BN;
                int b_col = idx % BN;
                int kb = nextOffset + b_row;
                int gb = globalCol + b_col;
                if (kb < K && gb < N)
                {
                    Bs[compIdx][b_row][b_col] = B[kb * N + gb];
                }
                else
                {
                    Bs[compIdx][b_row][b_col] = (__nv_bfloat16_raw)0.0f;
                }
            }
        }
        __syncthreads();

// Multiply-accumulate on tile "tile" in As[loadIdx], Bs[loadIdx]
#pragma unroll
        for (int m = 0; m < BK; m++)
        {
            __nv_bfloat16_raw aFrag[TM];
            __nv_bfloat16_raw bFrag[TN];
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
                int r = threadRow * TM + i;
                aFrag[i] = As[loadIdx][r][m];
            }
#pragma unroll
            for (int j = 0; j < TN; j++)
            {
                int c = threadCol * TN + j;
                bFrag[j] = Bs[loadIdx][m][c];
            }
// Accumulate
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
#pragma unroll
                for (int j = 0; j < TN; j++)
                {
                    sum[i][j] = bf16_fma(aFrag[i], bFrag[j], sum[i][j]);
                }
            }
        }
        __syncthreads();
    }

//----------- Write to global memory (+ optional bias) -----------
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int rowOut = globalRow + threadRow * TM + i;
        if (rowOut < M)
        {
#pragma unroll
            for (int j = 0; j < TN; j++)
            {
                int colOut = globalCol + threadCol * TN + j;
                if (colOut < N)
                {
                    size_t outIdx = rowOut * N + colOut;
                    __nv_bfloat16_raw val = sum[i][j];
                    if (bias != nullptr)
                    {
                        val = val + bias[outIdx];
                    }
                    C[outIdx] = val;
                }
            }
        }
    }
}

__device__ inline float floatFMA(float a, float b, float c)
{
    // If your GPU supports single-precision fused multiply add by default,
    // you can do:
    return __fmaf_rn(a, b, c);
    // or simply:
    // return a * b + c;
}

// Double‐buffered float kernel
__global__ void tomoLinearKernelimpF(
    const float *__restrict__ A,
    const float *__restrict__ B,
    size_t M, size_t K, size_t N,
    const float *__restrict__ bias,
    float *__restrict__ C)
{

    constexpr int BM = TileDims<float>::BM;
    constexpr int BK = TileDims<float>::BK;
    constexpr int BN = TileDims<float>::BN;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;

    const int globalRow = blockRow * BM;
    const int globalCol = blockCol * BN;

    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    float sum[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
#pragma unroll
        for (int j = 0; j < TN; j++)
        {
            sum[i][j] = 0.0f;
        }
    }

    const int numTiles = (K + BK - 1) / BK;
    int tid = threadRow * blockDim.x + threadCol;
    int tdim = blockDim.x * blockDim.y;

    int loadIdx = 0;
    int compIdx = 1;

    // Load tile 0
    {
        for (int idx = tid; idx < BM * BK; idx += tdim)
        {
            int a_row = idx / BK;
            int a_col = idx % BK;
            int ga = globalRow + a_row;
            int ka = a_col;
            As[loadIdx][a_row][a_col] = (ga < M && ka < K) ? A[ga * K + ka] : 0.0f;
        }
        for (int idx = tid; idx < BK * BN; idx += tdim)
        {
            int b_row = idx / BN;
            int b_col = idx % BN;
            int kb = b_row;
            int gb = globalCol + b_col;
            Bs[loadIdx][b_row][b_col] = (kb < K && gb < N) ? B[kb * N + gb] : 0.0f;
        }
    }
    __syncthreads();

    // main loop
    for (int tile = 0; tile < numTiles; tile++)
    {
        loadIdx = tile % 2;
        compIdx = (tile + 1) % 2;

        if (tile < numTiles - 1)
        {
            int nextOffset = (tile + 1) * BK;
            for (int idx = tid; idx < BM * BK; idx += tdim)
            {
                int a_row = idx / BK;
                int a_col = idx % BK;
                int ga = globalRow + a_row;
                int ka = nextOffset + a_col;
                As[compIdx][a_row][a_col] =
                    (ga < M && ka < K) ? A[ga * K + ka] : 0.0f;
            }
            for (int idx = tid; idx < BK * BN; idx += tdim)
            {
                int b_row = idx / BN;
                int b_col = idx % BN;
                int kb = nextOffset + b_row;
                int gb = globalCol + b_col;
                Bs[compIdx][b_row][b_col] =
                    (kb < K && gb < N) ? B[kb * N + gb] : 0.0f;
            }
        }
        __syncthreads();

// Multiply-accumulate
#pragma unroll
        for (int m = 0; m < BK; m++)
        {
            float aFrag[TM];
            float bFrag[TN];
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
                int r = threadRow * TM + i;
                aFrag[i] = As[loadIdx][r][m];
            }
#pragma unroll
            for (int j = 0; j < TN; j++)
            {
                int c = threadCol * TN + j;
                bFrag[j] = Bs[loadIdx][m][c];
            }
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
#pragma unroll
                for (int j = 0; j < TN; j++)
                {
                    sum[i][j] = floatFMA(aFrag[i], bFrag[j], sum[i][j]);
                }
            }
        }
        __syncthreads();
    }

// Write output
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int rowOut = globalRow + threadRow * TM + i;
        if (rowOut < M)
        {
#pragma unroll
            for (int j = 0; j < TN; j++)
            {
                int colOut = globalCol + threadCol * TN + j;
                if (colOut < N)
                {
                    size_t outIdx = rowOut * N + colOut;
                    float val = sum[i][j];
                    if (bias != nullptr)
                    {
                        val = val + bias[outIdx];
                    }
                    C[outIdx] = val;
                }
            }
        }
    }
}

__device__ inline double doubleFMA(double a, double b, double c)
{
    // If your hardware or compiler supports double-precision FMA by default,
    // you can do:
    return a * b + c;
    // or fma(a, b, c).
}

// Double-buffered double kernel
__global__ void tomoLinearKernelimpD(
    const double *__restrict__ A,
    const double *__restrict__ B,
    size_t M, size_t K, size_t N,
    const double *__restrict__ bias,
    double *__restrict__ C)
{
    constexpr int BM = TileDims<double>::BM;
    constexpr int BK = TileDims<double>::BK;
    constexpr int BN = TileDims<double>::BN;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;

    const int globalRow = blockRow * BM;
    const int globalCol = blockCol * BN;

    __shared__ double As[2][BM][BK];
    __shared__ double Bs[2][BK][BN];

    double sum[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
#pragma unroll
        for (int j = 0; j < TN; j++)
        {
            sum[i][j] = 0.0;
        }
    }

    const int numTiles = (K + BK - 1) / BK;
    int tid = threadRow * blockDim.x + threadCol;
    int tdim = blockDim.x * blockDim.y;

    int loadIdx = 0;
    int compIdx = 1;

    // First tile load
    {
        for (int idx = tid; idx < BM * BK; idx += tdim)
        {
            int a_row = idx / BK;
            int a_col = idx % BK;
            int ga = globalRow + a_row;
            int ka = a_col;
            As[loadIdx][a_row][a_col] =
                (ga < M && ka < K) ? A[ga * K + ka] : 0.0;
        }
        for (int idx = tid; idx < BK * BN; idx += tdim)
        {
            int b_row = idx / BN;
            int b_col = idx % BN;
            int kb = b_row;
            int gb = globalCol + b_col;
            Bs[loadIdx][b_row][b_col] =
                (kb < K && gb < N) ? B[kb * N + gb] : 0.0;
        }
    }
    __syncthreads();

    // Main loop
    for (int tile = 0; tile < numTiles; tile++)
    {
        loadIdx = tile % 2;
        compIdx = (tile + 1) % 2;

        // Next tile prefetch
        if (tile < numTiles - 1)
        {
            int nextOffset = (tile + 1) * BK;
            for (int idx = tid; idx < BM * BK; idx += tdim)
            {
                int a_row = idx / BK;
                int a_col = idx % BK;
                int ga = globalRow + a_row;
                int ka = nextOffset + a_col;
                As[compIdx][a_row][a_col] =
                    (ga < M && ka < K) ? A[ga * K + ka] : 0.0;
            }
            for (int idx = tid; idx < BK * BN; idx += tdim)
            {
                int b_row = idx / BN;
                int b_col = idx % BN;
                int kb = nextOffset + b_row;
                int gb = globalCol + b_col;
                Bs[compIdx][b_row][b_col] =
                    (kb < K && gb < N) ? B[kb * N + gb] : 0.0;
            }
        }
        __syncthreads();

// Multiply-accumulate
#pragma unroll
        for (int m = 0; m < BK; m++)
        {
            double aFrag[TM];
            double bFrag[TN];
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
                int r = threadRow * TM + i;
                aFrag[i] = As[loadIdx][r][m];
            }
#pragma unroll
            for (int j = 0; j < TN; j++)
            {
                int c = threadCol * TN + j;
                bFrag[j] = Bs[loadIdx][m][c];
            }
#pragma unroll
            for (int i = 0; i < TM; i++)
            {
#pragma unroll
                for (int j = 0; j < TN; j++)
                {
                    sum[i][j] = doubleFMA(aFrag[i], bFrag[j], sum[i][j]);
                }
            }
        }
        __syncthreads();
    }

// Write out
#pragma unroll
    for (int i = 0; i < TM; i++)
    {
        int rowOut = globalRow + threadRow * TM + i;
        if (rowOut < M)
        {
#pragma unroll
            for (int j = 0; j < TN; j++)
            {
                int colOut = globalCol + threadCol * TN + j;
                if (colOut < N)
                {
                    size_t outIdx = rowOut * N + colOut;
                    double val = sum[i][j];
                    if (bias != nullptr)
                    {
                        val += bias[outIdx];
                    }
                    C[outIdx] = val;
                }
            }
        }
    }
}

template <typename T>
cudaError_t tomoLinearImp(T const *A, T const *B, size_t M, size_t K, size_t N, T const *bias, T *C, cudaStream_t stream)
{
    constexpr int BM = TileDims<T>::BM;
    // constexpr int BK = TileDims<T>::BK;
    constexpr int BN = TileDims<T>::BN;

    dim3 blockDim(BN / TN, BM / TM); // e.g., (16, 16) with BN=128, BM=128, TN=8, TM=8
    dim3 gridDim((unsigned int)CEIL_DIV(N, BN), (unsigned int)CEIL_DIV(M, BM));

    if constexpr (std::is_same_v<T, __half_raw>)
    {
        tomoLinearKernelimpH<<<gridDim, blockDim, 0, stream>>>(
            A, B, M, K, N, bias, C);
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16_raw>)
    {
        tomoLinearKernelimpB<<<gridDim, blockDim, 0, stream>>>(
            A, B, M, K, N, bias, C);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        tomoLinearKernelimpF<<<gridDim, blockDim, 0, stream>>>(
            A, B, M, K, N, bias, C);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        tomoLinearKernelimpD<<<gridDim, blockDim, 0, stream>>>(
            A, B, M, K, N, bias, C);
    }

    return cudaGetLastError();
}

__global__ void tomoTransposeKernelH(const __half_raw *A, size_t M, size_t N, __half_raw *C)
{
    __shared__ __half_raw tile_h[BLOCK_SIZE][BLOCK_SIZE + 1];

    // (row, col) in the original A
    size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // 1) Read input => shared memory
    if (row < M && col < N)
        tile_h[threadIdx.y][threadIdx.x] = A[row * N + col];
    else
        tile_h[threadIdx.y][threadIdx.x] = (__half_raw)0.0;

    __syncthreads();

    // (row, col) in transposed output (swap blockIdx.x, blockIdx.y)
    size_t new_row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    size_t new_col = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    // 2) Write from shared memory => output
    if (new_row < N && new_col < M)
        C[new_row * M + new_col] = tile_h[threadIdx.x][threadIdx.y];
}

__global__ void tomoTransposeKernelB(const __nv_bfloat16_raw *A, size_t M, size_t N, __nv_bfloat16_raw *C)
{
    __shared__ __nv_bfloat16_raw tile_b[BLOCK_SIZE][BLOCK_SIZE + 1];

    size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < M && col < N)
        tile_b[threadIdx.y][threadIdx.x] = A[row * N + col];
    else
        tile_b[threadIdx.y][threadIdx.x] = (__nv_bfloat16_raw)0.0;

    __syncthreads();

    size_t new_row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    size_t new_col = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (new_row < N && new_col < M)
        C[new_row * M + new_col] = tile_b[threadIdx.x][threadIdx.y];
}

__global__ void tomoTransposeKernelF(const float *A, size_t M, size_t N, float *C)
{
    __shared__ float tile_f[BLOCK_SIZE][BLOCK_SIZE + 1];

    size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < M && col < N)
        tile_f[threadIdx.y][threadIdx.x] = A[row * N + col];
    else
        tile_f[threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();

    size_t new_row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    size_t new_col = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (new_row < N && new_col < M)
        C[new_row * M + new_col] = tile_f[threadIdx.x][threadIdx.y];
}

__global__ void tomoTransposeKernelD(const double *A, size_t M, size_t N, double *C)
{
    __shared__ double tile_d[BLOCK_SIZE][BLOCK_SIZE + 1];

    size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (row < M && col < N)
        tile_d[threadIdx.y][threadIdx.x] = A[row * N + col];
    else
        tile_d[threadIdx.y][threadIdx.x] = 0.0; // double literal

    __syncthreads();

    size_t new_row = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    size_t new_col = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    if (new_row < N && new_col < M)
        C[new_row * M + new_col] = tile_d[threadIdx.x][threadIdx.y];
}

// __global__ void tomoTransposeKernelH(__half_raw const *A, size_t M, size_t N, __half_raw *C)
// {
//     // Shared memory to hold a tile of the input matrix
//     __shared__ __half_raw tile_h[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflicts

//     // Input coordinates (reading from A)
//     auto x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
//     auto y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

//     // Load data into shared memory (coalesced read from A)
//     if (y < M && x < N)
//     {
//         tile_h[threadIdx.y][threadIdx.x] = A[y * N + x];
//     }
//     else
//     {
//         tile_h[threadIdx.y][threadIdx.x] = (__half_raw)0.0; // Padding for out-of-bounds
//     }

//     __syncthreads();

//     // Output coordinates (writing to C)
//     auto tx = blockIdx.y * BLOCK_SIZE + threadIdx.x; // Swapped block indices
//     auto ty = blockIdx.x * BLOCK_SIZE + threadIdx.y;

//     // Write transposed data to global memory (coalesced write to C)
//     if (ty < N && tx < M)
//     {
//         C[tx * N + ty] = tile_h[threadIdx.x][threadIdx.y]; // Note swapped indices
//     }
// }

template <typename T>
cudaError_t tomoTranspose(T const *A, size_t M, size_t N, T *C, cudaStream_t stream)
{

    // dim3 gridDim(((unsigned int)N + BLOCK_SIZE - 1) / BLOCK_SIZE, ((unsigned int)M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(((unsigned int)N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 ((unsigned int)M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if constexpr (std::is_same_v<T, __nv_bfloat16_raw>)
    {
        tomoTransposeKernelB<<<gridDim, blockDim, 0, stream>>>(A, M, N, C);
    }
    else if constexpr (std::is_same_v<T, __half_raw>)
    {
        tomoTransposeKernelH<<<gridDim, blockDim, 0, stream>>>(A, M, N, C);
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        tomoTransposeKernelF<<<gridDim, blockDim, 0, stream>>>(A, M, N, C);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        tomoTransposeKernelD<<<gridDim, blockDim, 0, stream>>>(A, M, N, C);
    }

    return cudaGetLastError();
}

template <typename T>
__global__ void tomoMaxToKernel(
    T const *d_in, T *d_out,
    size_t const *d_in_shape, size_t const *d_out_shape,
    size_t const *d_in_strides, size_t const *d_out_strides,
    size_t in_size, size_t out_size, size_t nd)
{
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_size)
        return;

    // Unravel out_idx to out_coords
    size_t out_coords[MAX_ND];
    size_t tmp = out_idx;
    for (ptrdiff_t d = nd - 1; d >= 0; --d)
    {
        size_t dim_size = d_out_shape[d];
        out_coords[d] = tmp % dim_size;
        tmp /= dim_size;
    }

    // Initialize max_val to the smallest possible value for type T
    T max_val = std::numeric_limits<T>::lowest();
    size_t in_coords[MAX_ND];
    for (size_t in_idx = 0; in_idx < in_size; ++in_idx)
    {
        size_t unravel = in_idx;
        bool matches = true;
        for (ptrdiff_t d = nd - 1; d >= 0; --d)
        {
            size_t dim_size = d_in_shape[d];
            in_coords[d] = unravel % dim_size;
            unravel /= dim_size;
            if (d_out_shape[d] != 1)
            {
                size_t out_c = out_coords[d];
                size_t in_c = (d_in_shape[d] == 1) ? 0 : in_coords[d];
                if (out_c != in_c)
                {
                    matches = false;
                    break;
                }
            }
        }
        if (matches)
        {
            size_t in_offset = 0;
            for (size_t d = 0; d < nd; ++d)
            {
                in_offset += in_coords[d] * d_in_strides[d];
            }
            max_val = std::max(max_val, d_in[in_offset]);
        }
    }
    d_out[out_idx] = max_val;
}

template <typename T>
cudaError_t tomoMaxTo(
    const T *d_in, T *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size, size_t out_size, size_t nd,
    cudaStream_t stream)
{
    // Validate inputs
    if (out_strides_len != nd)
        return cudaErrorInvalidValue;

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
    tomoMaxToKernel<<<blocks, threads, 0, stream>>>(
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
__global__ void tomoMinToKernel(
    T const *d_in, T *d_out,
    size_t const *d_in_shape, size_t const *d_out_shape,
    size_t const *d_in_strides, size_t const *d_out_strides,
    size_t in_size, size_t out_size, size_t nd)
{
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_size)
        return;

    // Unravel out_idx to out_coords
    size_t out_coords[MAX_ND];
    size_t tmp = out_idx;
    for (ptrdiff_t d = nd - 1; d >= 0; --d)
    {
        size_t dim_size = d_out_shape[d];
        out_coords[d] = tmp % dim_size;
        tmp /= dim_size;
    }

    // Initialize min_val to the largest possible value for type T
    T min_val = std::numeric_limits<T>::max();
    size_t in_coords[MAX_ND];
    for (size_t in_idx = 0; in_idx < in_size; ++in_idx)
    {
        size_t unravel = in_idx;
        bool matches = true;
        for (ptrdiff_t d = nd - 1; d >= 0; --d)
        {
            size_t dim_size = d_in_shape[d];
            in_coords[d] = unravel % dim_size;
            unravel /= dim_size;
            if (d_out_shape[d] != 1)
            {
                size_t out_c = out_coords[d];
                size_t in_c = (d_in_shape[d] == 1) ? 0 : in_coords[d];
                if (out_c != in_c)
                {
                    matches = false;
                    break;
                }
            }
        }
        if (matches)
        {
            size_t in_offset = 0;
            for (size_t d = 0; d < nd; ++d)
            {
                in_offset += in_coords[d] * d_in_strides[d];
            }
            min_val = std::min(min_val, d_in[in_offset]);
        }
    }
    d_out[out_idx] = min_val;
}

template <typename T>
cudaError_t tomoMinTo(
    const T *d_in, T *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t in_size, size_t out_size, size_t nd,
    cudaStream_t stream)
{
    // Validate inputs
    if (out_strides_len != nd)
        return cudaErrorInvalidValue;

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
    tomoMinToKernel<<<blocks, threads, 0, stream>>>(
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
__global__ void tomoTensordotKernel(
    const T *d_a, const T *d_b, T *d_out,
    const size_t *d_a_shape, const size_t *d_b_shape, const size_t *d_out_shape,
    const size_t *d_a_strides, const size_t *d_b_strides, const size_t *d_out_strides,
    size_t a_nd, size_t b_nd, size_t out_nd,
    size_t out_size, size_t num_contracted,
    const size_t *d_contracted_axes_a, const size_t *d_contracted_axes_b)
{
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_size)
        return;

    // Unravel output index
    size_t out_coords[MAX_ND];
    size_t tmp = out_idx;
    for (size_t d = out_nd; d-- > 0;)
    {
        out_coords[d] = tmp % d_out_shape[d];
        tmp /= d_out_shape[d];
    }

    // Initialize coordinates for A and B
    size_t a_coords[MAX_ND] = {0};
    size_t b_coords[MAX_ND] = {0};

    // Map output coordinates to non-contracted axes
    size_t a_non_contracted_idx = 0;
    size_t b_non_contracted_idx = 0;

    for (size_t d = 0; d < a_nd; ++d)
    {
        bool is_contracted = false;
        for (size_t p = 0; p < num_contracted; ++p)
        {
            if (d == d_contracted_axes_a[p])
            {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted)
        {
            a_coords[d] = out_coords[a_non_contracted_idx++];
        }
    }

    for (size_t d = 0; d < b_nd; ++d)
    {
        bool is_contracted = false;
        for (size_t p = 0; p < num_contracted; ++p)
        {
            if (d == d_contracted_axes_b[p])
            {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted)
        {
            b_coords[d] = out_coords[a_non_contracted_idx + b_non_contracted_idx++];
        }
    }

    // Contracted dimensions loop
    size_t loop_size = 1;
    size_t contracted_dims[MAX_ND];
    for (size_t p = 0; p < num_contracted; ++p)
    {
        contracted_dims[p] = d_a_shape[d_contracted_axes_a[p]];
        loop_size *= contracted_dims[p];
    }

    T sum = (T)0.0f;
    for (size_t k = 0; k < loop_size; ++k)
    {
        size_t temp_k = k;

        // Set contracted indices for A and B
        for (size_t p = 0; p < num_contracted; ++p)
        {
            const size_t dim_idx = num_contracted - p - 1;
            const size_t dim_size = contracted_dims[dim_idx];
            const size_t idx = temp_k % dim_size;
            temp_k /= dim_size;

            a_coords[d_contracted_axes_a[dim_idx]] = idx;
            b_coords[d_contracted_axes_b[dim_idx]] = idx;
        }

        // Calculate offsets
        size_t a_offset = 0, b_offset = 0;
        for (size_t d = 0; d < a_nd; ++d)
            a_offset += a_coords[d] * d_a_strides[d];
        for (size_t d = 0; d < b_nd; ++d)
            b_offset += b_coords[d] * d_b_strides[d];

        sum = sum + d_a[a_offset] * d_b[b_offset];
    }

    d_out[out_idx] = sum;
}

// Main tensordot function
template <typename T>
cudaError_t tomoTensordot(
    const T *d_a, const T *d_b, T *d_out,
    const size_t *a_shape, size_t a_shape_len,
    const size_t *b_shape, size_t b_shape_len,
    const size_t *out_shape, size_t out_shape_len,
    const size_t *a_strides, size_t a_strides_len,
    const size_t *b_strides, size_t b_strides_len,
    const size_t *out_strides, size_t out_strides_len,
    const size_t *contracted_axes_a, size_t contracted_axes_a_len,
    const size_t *contracted_axes_b, size_t contracted_axes_b_len,
    size_t a_nd, size_t b_nd, size_t out_nd,
    size_t out_size, size_t num_contracted,
    cudaStream_t stream)
{
    // Validate inputs
    if (a_shape_len != a_nd || b_shape_len != b_nd || out_shape_len != out_nd ||
        a_strides_len != a_nd || b_strides_len != b_nd || out_strides_len != out_nd ||
        contracted_axes_a_len != num_contracted || contracted_axes_b_len != num_contracted)
    {
        return cudaErrorInvalidValue;
    }

    // Device buffers
    size_t *d_a_shape, *d_b_shape, *d_out_shape;
    size_t *d_a_strides, *d_b_strides, *d_out_strides;
    size_t *d_contracted_axes_a, *d_contracted_axes_b;

    CHECK_CUDA(cudaMallocAsync(&d_a_shape, a_nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_b_shape, b_nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_shape, out_nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_a_strides, a_nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_b_strides, b_nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_strides, out_nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_contracted_axes_a, num_contracted * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_contracted_axes_b, num_contracted * sizeof(size_t), stream));

    // Copy data to device
    CHECK_CUDA(cudaMemcpyAsync(d_a_shape, a_shape, a_nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_b_shape, b_shape, b_nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_shape, out_shape, out_nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_a_strides, a_strides, a_nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_b_strides, b_strides, b_nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_strides, out_strides, out_nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_contracted_axes_a, contracted_axes_a, num_contracted * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_contracted_axes_b, contracted_axes_b, num_contracted * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    // Launch kernel
    const int threads = 256;
    const int blocks = ((int)out_size + threads - 1) / threads;
    tomoTensordotKernel<T><<<blocks, threads, 0, stream>>>(
        d_a, d_b, d_out,
        d_a_shape, d_b_shape, d_out_shape,
        d_a_strides, d_b_strides, d_out_strides,
        a_nd, b_nd, out_nd,
        out_size, num_contracted,
        d_contracted_axes_a, d_contracted_axes_b);

    CHECK_CUDA(cudaGetLastError());

    // Free device memory
    CHECK_CUDA(cudaFreeAsync(d_a_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_b_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_out_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_a_strides, stream));
    CHECK_CUDA(cudaFreeAsync(d_b_strides, stream));
    CHECK_CUDA(cudaFreeAsync(d_out_strides, stream));
    CHECK_CUDA(cudaFreeAsync(d_contracted_axes_a, stream));
    CHECK_CUDA(cudaFreeAsync(d_contracted_axes_b, stream));

    return cudaSuccess;
}

template <typename T>
__global__ void tomoTransposeExKernel(
    T const *d_in,
    T *d_out,
    size_t const *d_in_shape,
    size_t const *d_out_shape,
    size_t const *d_in_strides,
    size_t const *d_out_strides, // Included for API consistency, not used
    size_t const *d_perm,
    size_t const nd,
    size_t const out_size)
{
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_size)
        return;

    // Unravel out_idx to out_coord
    size_t out_coord[MAX_ND];
    size_t tmp = out_idx;
    for (size_t d = nd; d-- > 0;)
    {
        out_coord[d] = tmp % d_out_shape[d];
        tmp /= d_out_shape[d];
    }

    // Compute input_coord using permutation
    size_t input_coord[MAX_ND];
    for (size_t i = 0; i < nd; ++i)
    {
        size_t p = d_perm[i];
        input_coord[p] = out_coord[i];
    }

    // Compute in_idx using input strides
    size_t in_idx = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        in_idx += input_coord[d] * d_in_strides[d];
    }

    // Copy element
    d_out[out_idx] = d_in[in_idx];
}

template <typename T>
cudaError_t tomoTransposeEx(
    T const *d_in,
    T *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *out_shape, size_t const out_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const *out_strides, size_t const out_strides_len,
    size_t const *perm, size_t const perm_len,
    size_t const nd,
    size_t const in_size,
    size_t const out_size,
    cudaStream_t const stream)
{
    // Validate dimensions
    if (nd > MAX_ND)
        return cudaErrorInvalidValue;
    if (in_shape_len != nd || out_shape_len != nd || in_strides_len != nd ||
        out_strides_len != nd || perm_len != nd)
        return cudaErrorInvalidValue;

    // Validate permutation
    bool seen[MAX_ND] = {false};
    for (size_t i = 0; i < nd; ++i)
    {
        size_t p = perm[i];
        if (p >= nd || seen[p])
            return cudaErrorInvalidValue;
        seen[p] = true;
    }

    // Validate output shape
    for (size_t i = 0; i < nd; ++i)
    {
        if (out_shape[i] != in_shape[perm[i]])
            return cudaErrorInvalidValue;
    }

    // Allocate device memory
    size_t *d_in_shape, *d_out_shape, *d_in_strides, *d_out_strides, *d_perm;
    CHECK_CUDA(cudaMallocAsync(&d_in_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_in_strides, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_strides, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_perm, nd * sizeof(size_t), stream));

    // Copy data to device
    CHECK_CUDA(cudaMemcpyAsync(d_in_shape, in_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_shape, out_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_in_strides, in_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_strides, out_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_perm, perm, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    // Launch kernel
    const int threads = 256;
    const int blocks = ((int)out_size + threads - 1) / threads;
    tomoTransposeExKernel<T><<<blocks, threads, 0, stream>>>(
        d_in, d_out, d_in_shape, d_out_shape, d_in_strides, d_out_strides, d_perm, nd, out_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    // Free device memory
    cudaFreeAsync(d_in_shape, stream);
    cudaFreeAsync(d_out_shape, stream);
    cudaFreeAsync(d_in_strides, stream);
    cudaFreeAsync(d_out_strides, stream);
    cudaFreeAsync(d_perm, stream);

    return cudaSuccess;
}

void computeRollaxisPerm(size_t nd, size_t axis, size_t start, size_t *perm)
{
    size_t temp_perm[MAX_ND];
    size_t idx = 0;
    // Collect all axes except the one to roll
    for (size_t i = 0; i < nd; ++i)
    {
        if (i != axis)
        {
            temp_perm[idx++] = i;
        }
    }
    // Adjust start if it exceeds the number of remaining axes
    if (start > idx)
    {
        start = idx;
    }
    // Build the permutation: axes before start, axis, axes after start
    for (size_t i = 0; i < start; ++i)
    {
        perm[i] = temp_perm[i];
    }
    perm[start] = axis;
    for (size_t i = start; i < nd - 1; ++i)
    {
        perm[i + 1] = temp_perm[i];
    }
}

template <typename T>
cudaError_t tomoRollaxis(
    T const *d_in,               // Input tensor on device
    T *d_out,                    // Output tensor on device
    size_t const *in_shape,      // Input shape array
    size_t const in_shape_len,   // Length of in_shape
    size_t const *in_strides,    // Input strides array
    size_t const in_strides_len, // Length of in_strides
    size_t const axis,           // Axis to roll
    size_t const start,          // Target position
    size_t const nd,             // Number of dimensions
    size_t const in_size,        // Total input elements
    size_t const out_size,       // Total output elements (should equal in_size)
    cudaStream_t const stream    // CUDA stream for async execution
)
{
    // Validate inputs
    if (nd > MAX_ND || in_shape_len != nd || in_strides_len != nd ||
        axis >= nd || start > nd || in_size != out_size)
    {
        return cudaErrorInvalidValue;
    }

    // Compute permutation
    size_t perm[MAX_ND];
    computeRollaxisPerm(nd, axis, start, perm);

    // Compute output shape
    size_t out_shape[MAX_ND];
    for (size_t i = 0; i < nd; ++i)
    {
        out_shape[i] = in_shape[perm[i]];
    }

    // Allocate device memory for arrays
    size_t *d_in_shape, *d_out_shape, *d_in_strides, *d_perm;
    CHECK_CUDA(cudaMallocAsync(&d_in_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_in_strides, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_perm, nd * sizeof(size_t), stream));

    // Copy data to device
    CHECK_CUDA(cudaMemcpyAsync(d_in_shape, in_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_shape, out_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_in_strides, in_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_perm, perm, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    // Launch kernel (assuming tomoTransposeExKernel is defined)
    const int threads = 256;
    const int blocks = (static_cast<int>(out_size) + threads - 1) / threads;
    tomoTransposeExKernel<T><<<blocks, threads, 0, stream>>>(
        d_in, d_out, d_in_shape, d_out_shape, d_in_strides, nullptr, d_perm, nd, out_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    // Free device memory
    cudaFreeAsync(d_in_shape, stream);
    cudaFreeAsync(d_out_shape, stream);
    cudaFreeAsync(d_in_strides, stream);
    cudaFreeAsync(d_perm, stream);
    return cudaSuccess;
}

template <typename T>
__global__ void tomoSwapaxesExKernel(
    T const *d_in, T *d_out,
    size_t const *d_in_shape, size_t const *d_out_shape,
    size_t const *d_in_strides, size_t const *d_out_strides,
    size_t axis1, size_t axis2, size_t nd, size_t out_size)
{
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_size)
        return;

    // Compute multi-dimensional output coordinates
    size_t out_coord[MAX_ND];
    size_t tmp = out_idx;
    for (size_t d = nd; d-- > 0;)
    {
        out_coord[d] = tmp % d_out_shape[d];
        tmp /= d_out_shape[d];
    }

    // Map to input coordinates by swapping axis1 and axis2
    size_t in_coord[MAX_ND];
    for (size_t d = 0; d < nd; ++d)
    {
        if (d == axis1)
        {
            in_coord[d] = out_coord[axis2];
        }
        else if (d == axis2)
        {
            in_coord[d] = out_coord[axis1];
        }
        else
        {
            in_coord[d] = out_coord[d];
        }
    }

    // Compute input index using strides
    size_t in_idx = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        in_idx += in_coord[d] * d_in_strides[d];
    }

    // Copy element from input to output
    d_out[out_idx] = d_in[in_idx];
}

template <typename T>
cudaError_t tomoSwapaxes(
    T const *d_in,              // Input tensor on device
    T *d_out,                   // Output tensor on device
    size_t const *in_shape,     // Input shape array
    size_t in_shape_len,        // Length of in_shape
    size_t const *out_shape,    // Output shape array
    size_t out_shape_len,       // Length of out_shape
    size_t const *in_strides,   // Input strides array
    size_t in_strides_len,      // Length of in_strides
    size_t const *out_strides,  // Output strides array
    size_t out_strides_len,     // Length of out_strides
    size_t axis1, size_t axis2, // Axes to swap
    size_t nd,                  // Number of dimensions
    size_t in_size,             // Total input elements
    size_t out_size,            // Total output elements
    cudaStream_t stream)        // CUDA stream for async execution
{
    // Input validation
    if (nd > MAX_ND || in_shape_len != nd || out_shape_len != nd ||
        in_strides_len != nd || out_strides_len != nd ||
        axis1 >= nd || axis2 >= nd || in_size != out_size)
    {
        return cudaErrorInvalidValue;
    }

    // Verify output shape matches swapped input shape
    for (size_t d = 0; d < nd; ++d)
    {
        size_t expected = (d == axis1) ? in_shape[axis2] : (d == axis2) ? in_shape[axis1]
                                                                        : in_shape[d];
        if (out_shape[d] != expected)
        {
            return cudaErrorInvalidValue;
        }
    }

    // Allocate device memory for shapes and strides
    size_t *d_in_shape, *d_out_shape, *d_in_strides, *d_out_strides;
    CHECK_CUDA(cudaMallocAsync(&d_in_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_in_strides, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_strides, nd * sizeof(size_t), stream));

    // Copy shape and stride data to device
    CHECK_CUDA(cudaMemcpyAsync(d_in_shape, in_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_shape, out_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_in_strides, in_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_strides, out_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream));

    // Launch kernel
    const int threads = 256;
    const int blocks = (int)(out_size + threads - 1) / threads;
    tomoSwapaxesExKernel<T><<<blocks, threads, 0, stream>>>(
        d_in, d_out, d_in_shape, d_out_shape, d_in_strides, d_out_strides,
        axis1, axis2, nd, out_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    // Free device memory
    cudaFreeAsync(d_in_shape, stream);
    cudaFreeAsync(d_out_shape, stream);
    cudaFreeAsync(d_in_strides, stream);
    cudaFreeAsync(d_out_strides, stream);
    return cudaSuccess;
}

//------------------------------------------------------------------------------
// tomoIm2colKernel
//------------------------------------------------------------------------------
// Converts an image with shape (n, c, h, w) into a column tensor of shape
// (n, c, kh, kw, out_h, out_w). The convolution parameters are given by kernel
// size (kh, kw), stride (sy, sx), pad (ph, pw) and dilation (dy, dx).
//------------------------------------------------------------------------------
// tomoIm2colKernel (same logic, one thread per element of d_col)
//------------------------------------------------------------------------------
template <typename T>
__global__ void tomoIm2colKernel(
    T const *d_img, T *d_col,
    size_t const n, size_t const c, size_t const h, size_t const w,
    size_t const kh, size_t const kw,
    size_t const out_h, size_t const out_w,
    size_t const sy, size_t const sx,
    size_t const ph, size_t const pw,
    size_t const dy, size_t const dx)
{
    // Each thread processes one element in the "col" space
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * c * kh * kw * out_h * out_w;
    if (idx >= total)
        return;

    // Unravel idx into 6 coordinates: [n_idx, c_idx, k_y, k_x, out_y, out_x]
    size_t dims[6] = {n, c, kh, kw, out_h, out_w};
    size_t coords[6];
    {
        size_t tmp = idx;
        for (ptrdiff_t d = 5; d >= 0; --d)
        {
            coords[d] = tmp % dims[d];
            tmp /= dims[d];
        }
    }
    size_t n_idx = coords[0];
    size_t c_idx = coords[1];
    size_t k_y = coords[2];
    size_t k_x = coords[3];
    size_t out_y = coords[4];
    size_t out_x = coords[5];

    // Compute the corresponding input coordinate
    int in_y = static_cast<int>(k_y * dy + out_y * sy) - static_cast<int>(ph);
    int in_x = static_cast<int>(k_x * dx + out_x * sx) - static_cast<int>(pw);

    T value = (T)0;
    if (in_y >= 0 && in_y < static_cast<int>(h) &&
        in_x >= 0 && in_x < static_cast<int>(w))
    {
        // input index
        size_t img_idx = n_idx * (c * h * w) + c_idx * (h * w) + (size_t)in_y * w + (size_t)in_x;
        value = d_img[img_idx];
    }

    // Write to d_col
    d_col[idx] = value;
}

//------------------------------------------------------------------------------
// tomoCol2imKernel (FIXED with atomicAdd or at least one-thread-per-col-element)
//------------------------------------------------------------------------------
// We do: one thread per element in the column tensor, then use atomicAdd on d_img.
template <typename T>
__global__ void tomoCol2imKernel(
    T const *d_col,
    T *d_img,
    size_t const n, size_t const c,
    size_t const h, size_t const w,
    size_t const kh, size_t const kw,
    size_t const out_h, size_t const out_w,
    size_t const sy, size_t const sx,
    size_t const ph, size_t const pw,
    size_t const dx, size_t const dy)
{
    // Each thread processes one element in the "col" space
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * c * kh * kw * out_h * out_w;
    if (idx >= total)
        return;

    // Unravel idx into (n, c, kh, kw, out_h, out_w)
    size_t dims[6] = {n, c, kh, kw, out_h, out_w};
    size_t coords[6];
    {
        size_t tmp = idx;
        for (ptrdiff_t d = 5; d >= 0; --d)
        {
            coords[d] = tmp % dims[d];
            tmp /= dims[d];
        }
    }
    size_t n_idx = coords[0];
    size_t c_idx = coords[1];
    size_t k_y = coords[2];
    size_t k_x = coords[3];
    size_t out_y = coords[4];
    size_t out_x = coords[5];

    T val = d_col[idx]; // contribution from this col element

    // Compute the corresponding input coordinate
    int in_y = static_cast<int>(k_y * dy + out_y * sy) - static_cast<int>(ph);
    int in_x = static_cast<int>(k_x * dx + out_x * sx) - static_cast<int>(pw);

    // If inside the image, use atomicAdd to sum partial contributions
    if (in_y >= 0 && in_y < static_cast<int>(h) &&
        in_x >= 0 && in_x < static_cast<int>(w))
    {
        size_t img_idx = n_idx * (c * h * w) + c_idx * (h * w) + (size_t)in_y * w + (size_t)in_x;

        // Use atomicAdd. Implementation differs for half/bfloat16 vs. float/double
        if constexpr (std::is_same_v<T, __half_raw>)
        {
            atomicAdd(reinterpret_cast<__half *>(&d_img[img_idx]),
                      static_cast<__half>(val));
        }
        else if constexpr (std::is_same_v<T, __nv_bfloat16_raw>)
        {
            atomicAdd(reinterpret_cast<__nv_bfloat16 *>(&d_img[img_idx]),
                      static_cast<__nv_bfloat16>(val));
        }
        else
        {
            atomicAdd(&d_img[img_idx], val);
        }
    }
}

//------------------------------------------------------------------------------
// Host wrappers with updated usage
//------------------------------------------------------------------------------
template <typename T>
cudaError_t tomoIm2col(
    T const *d_img, T *d_col,
    size_t const n, size_t const c,
    size_t const h, size_t const w,
    size_t const kh, size_t const kw,
    size_t const sy, size_t const sx,
    size_t const ph, size_t const pw,
    size_t const dy, size_t const dx,
    cudaStream_t stream)
{
    size_t const out_h = (h + 2 * ph - (kh - 1) * dy - 1) / sy + 1; // more general formula
    size_t const out_w = (w + 2 * pw - (kw - 1) * dx - 1) / sx + 1;
    size_t const total = n * c * kh * kw * out_h * out_w;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    tomoIm2colKernel<T><<<blocks, threads, 0, stream>>>(
        d_img, d_col,
        n, c, h, w,
        kh, kw, out_h, out_w,
        sy, sx, ph, pw,
        dy, dx);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return err;

    return cudaSuccess;
}

template <typename T>
cudaError_t tomoCol2im(
    T const *d_col, T *d_img,
    size_t const n, size_t const c,
    size_t const h, size_t const w,
    size_t const kh, size_t const kw,
    size_t const sy, size_t const sx,
    size_t const ph, size_t const pw,
    size_t const dx, size_t const dy,
    cudaStream_t stream)
{
    // We can zero the output first because we'll do atomicAdd:
    size_t const img_total = n * c * h * w;
    cudaError_t err = cudaMemsetAsync(d_img, 0, img_total * sizeof(T), stream);
    if (err != cudaSuccess)
        return err;

    // compute out_h/out_w in the same general form
    size_t const out_h = (h + 2 * ph - (kh - 1) * dy - 1) / sy + 1;
    size_t const out_w = (w + 2 * pw - (kw - 1) * dx - 1) / sx + 1;

    size_t const col_total = n * c * kh * kw * out_h * out_w;
    int threads = 256;
    int blocks = (int)((col_total + threads - 1) / threads);

    tomoCol2imKernel<T><<<blocks, threads, 0, stream>>>(
        d_col, d_img,
        n, c, h, w,
        kh, kw, out_h, out_w,
        sy, sx, ph, pw,
        dx, dy);
    err = cudaGetLastError();
    return err;
}

#include <cuda_runtime.h>
#include <type_traits> // for std::is_same_v

//------------------------------------------------------------------------------
// 1D im2col kernel
//------------------------------------------------------------------------------
template <typename T>
__global__ void tomoIm2col1dKernel(
    T const *__restrict__ d_in, // [N, C, L]
    T *__restrict__ d_col,      // [N, C, K, outL]
    size_t n,                   // batch size
    size_t c,                   // channels
    size_t l,                   // input length
    size_t k,                   // kernel size
    size_t outL,                // output length
    size_t stride,
    size_t pad,
    size_t dilation)
{
    // linear index into the total of (n * c * k * outL)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * c * k * outL;
    if (idx >= total)
        return;

    // Unravel idx into 4D coords: (n_idx, c_idx, k_idx, out_idx)
    // dims = [n, c, k, outL]
    size_t dims[4] = {n, c, k, outL};
    size_t coords[4];
    {
        size_t tmp = idx;
        for (int d = 3; d >= 0; --d)
        {
            coords[d] = tmp % dims[d];
            tmp /= dims[d];
        }
    }
    size_t n_idx = coords[0];
    size_t c_idx = coords[1];
    size_t k_idx = coords[2];
    size_t out_idx = coords[3];

    // Compute the corresponding input index:
    // in_idx = out_idx*stride - pad + k_idx*dilation
    int in_idx = static_cast<int>(out_idx * stride) - static_cast<int>(pad) + static_cast<int>(k_idx * dilation);

    // read value from input if in range
    T value = T(0);
    if (in_idx >= 0 && in_idx < static_cast<int>(l))
    {
        size_t in_linear = n_idx * (c * l) + c_idx * l + (size_t)in_idx;
        value = d_in[in_linear];
    }

    // write into d_col
    d_col[idx] = value;
}

//------------------------------------------------------------------------------
// 1D col2im kernel
//------------------------------------------------------------------------------
template <typename T>
__global__ void tomoCol2im1dKernel(
    T const *__restrict__ d_col, // [N, C, K, outL]
    T *__restrict__ d_in,        // [N, C, L]
    size_t n,
    size_t c,
    size_t l,
    size_t k,
    size_t outL,
    size_t stride,
    size_t pad,
    size_t dilation)
{
    // linear index into [N, C, K, outL]
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * c * k * outL;
    if (idx >= total)
        return;

    // unravel idx -> (n_idx, c_idx, k_idx, out_idx)
    size_t dims[4] = {n, c, k, outL};
    size_t coords[4];
    {
        size_t tmp = idx;
        for (int d = 3; d >= 0; --d)
        {
            coords[d] = tmp % dims[d];
            tmp /= dims[d];
        }
    }
    size_t n_idx = coords[0];
    size_t c_idx = coords[1];
    size_t k_idx = coords[2];
    size_t out_idx = coords[3];

    T val = d_col[idx];

    // compute the input index
    int in_idx = static_cast<int>(out_idx * stride) - static_cast<int>(pad) + static_cast<int>(k_idx * dilation);

    if (in_idx >= 0 && in_idx < static_cast<int>(l))
    {
        size_t in_linear = n_idx * (c * l) + c_idx * l + (size_t)in_idx;

        // If you also want half/bfloat16 support with atomicAdd:
        //   if constexpr (std::is_same_v<T, __half_raw>) { ... }
        //   else if constexpr (std::is_same_v<T, __nv_bfloat16_raw>) { ... }
        //   else { atomicAdd(...); }

        if constexpr (std::is_same_v<T, __half_raw>)
        {
            atomicAdd(reinterpret_cast<__half *>(&d_in[in_linear]),
                      static_cast<__half>(val));
        }
        else if constexpr (std::is_same_v<T, __nv_bfloat16_raw>)
        {
            atomicAdd(reinterpret_cast<__nv_bfloat16 *>(&d_in[in_linear]),
                      static_cast<__nv_bfloat16>(val));
        }
        else
        {
            atomicAdd(&d_in[in_linear], val);
        }
    }
}

template <typename T>
cudaError_t tomoIm2col1d(
    T const *d_in,
    T *d_col,
    size_t n, // batch
    size_t c, // channels
    size_t l, // input length
    size_t k, // kernel size
    size_t stride,
    size_t pad,
    size_t dilation,
    cudaStream_t stream)
{
    // compute outL
    //   outL = floor( (l + 2*pad - (k-1)*dilation - 1) / stride ) + 1
    size_t outL = (l + 2 * pad - (k - 1) * dilation - 1) / stride + 1;

    // total threads = n*c*k*outL
    size_t total = n * c * k * outL;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    tomoIm2col1dKernel<T><<<blocks, threads, 0, stream>>>(
        d_in, d_col,
        n, c, l, k, outL,
        stride, pad, dilation);

    return cudaGetLastError();
}

template <typename T>
cudaError_t tomoCol2im1d(
    T const *d_col,
    T *d_in,
    size_t n, // batch
    size_t c, // channels
    size_t l, // input length
    size_t k, // kernel size
    size_t stride,
    size_t pad,
    size_t dilation,
    cudaStream_t stream)
{
    // Optionally zero out d_in first if you want to sum partial results:
    cudaError_t err = cudaMemsetAsync(d_in, 0, n * c * l * sizeof(T), stream);
    if (err != cudaSuccess)
        return err;

    // compute outL same as above
    size_t outL = (l + 2 * pad - (k - 1) * dilation - 1) / stride + 1;

    size_t total = n * c * k * outL;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    tomoCol2im1dKernel<T><<<blocks, threads, 0, stream>>>(
        d_col, d_in,
        n, c, l, k, outL,
        stride, pad, dilation);

    return cudaGetLastError();
}

// Forward declaration so we can partially specialize.
template <typename T>
__device__ inline T deviceInfinity(bool negative);

// Specialize for float
template <>
__device__ inline float deviceInfinity<float>(bool negative)
{
    return negative ? -std::numeric_limits<float>::infinity()
                    : std::numeric_limits<float>::infinity();
}

// Specialize for double
template <>
__device__ inline double deviceInfinity<double>(bool negative)
{
    return negative ? -std::numeric_limits<double>::infinity()
                    : std::numeric_limits<double>::infinity();
}

// Specialize for __half_raw
// We do a simple conversion from float∞ to half∞.
template <>
__device__ inline __half_raw deviceInfinity<__half_raw>(bool negative)
{
    float inf = negative ? -std::numeric_limits<float>::infinity()
                         : std::numeric_limits<float>::infinity();
    // This cast depends on your environment; you may need __float2half_rn(inf).
    // For “raw” half, do a reinterpret if you already have an operator.
    // If no operator is available, define a custom conversion.
    // For demonstration, assume direct C-style cast is valid:
    return (__half_raw)inf;
}

// Specialize for __nv_bfloat16_raw
template <>
__device__ inline __nv_bfloat16_raw deviceInfinity<__nv_bfloat16_raw>(bool negative)
{
    float inf = negative ? -std::numeric_limits<float>::infinity()
                         : std::numeric_limits<float>::infinity();
    // Similarly for bfloat16.
    return (__nv_bfloat16_raw)inf;
}

template <typename T>
__global__ void tomoArgmaxKernel(
    const T *d_in, // Input data
    const size_t *in_shape,
    const size_t *in_strides,
    size_t *d_out, // Output indices
    const size_t *out_shape,
    const size_t *out_strides,
    size_t out_size,
    size_t nd)
{
    // Each block handles one output element
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size)
        return;

    // Compute output coords from out_idx
    size_t out_coords[MAX_ND];
    {
        size_t tmp = out_idx;
        for (ptrdiff_t d = nd - 1; d >= 0; --d)
        {
            out_coords[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
    }

    // Compute output offset
    size_t out_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        out_offset += out_coords[d] * out_strides[d];
    }

    // Compute base offset in the input
    size_t base_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        size_t in_c = (out_shape[d] == 1) ? 0 : out_coords[d];
        base_offset += in_c * in_strides[d];
    }

    // Identify reduced dimensions
    size_t reduced_dims[MAX_ND];
    size_t reduced_sizes[MAX_ND];
    size_t num_reduced = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        // If out_shape[d] == 1 but in_shape[d] > 1 => dimension is reduced
        if (out_shape[d] == 1 && in_shape[d] > 1)
        {
            reduced_dims[num_reduced] = d;
            reduced_sizes[num_reduced] = in_shape[d];
            num_reduced++;
        }
    }

    // Total # of elements in reduced dimensions
    size_t N = 1;
    for (size_t j = 0; j < num_reduced; ++j)
    {
        N *= reduced_sizes[j];
    }

    // Each thread finds a local maximum over its subset
    T local_max = deviceInfinity<T>(/*negative=*/true); // negative infinity
    size_t local_argmax = 0;

    for (size_t i = threadIdx.x; i < N; i += blockDim.x)
    {
        // Unravel i into coords along the reduced dims
        size_t reduced_coords[MAX_ND];
        {
            size_t tmp_i = i;
            for (ptrdiff_t j = num_reduced - 1; j >= 0; --j)
            {
                reduced_coords[j] = tmp_i % reduced_sizes[j];
                tmp_i /= reduced_sizes[j];
            }
        }

        // Compute input offset
        size_t offset = base_offset;
        for (size_t j = 0; j < num_reduced; ++j)
        {
            size_t d = reduced_dims[j];
            offset += reduced_coords[j] * in_strides[d];
        }

        // Update local max and argmax
        T val = d_in[offset];
        if (val > local_max)
        {
            local_max = val;
            local_argmax = i;
        }
    }

    // Use shared memory for block-level reduction
    extern __shared__ char shared_mem[];
    // We place an array of T, then an array of size_t
    T *shared_max = reinterpret_cast<T *>(shared_mem);
    size_t *shared_argmax = reinterpret_cast<size_t *>(shared_max + blockDim.x);

    shared_max[threadIdx.x] = local_max;
    shared_argmax[threadIdx.x] = local_argmax;
    __syncthreads();

    // Parallel reduction by half
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            if (shared_max[threadIdx.x + s] > shared_max[threadIdx.x])
            {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + s];
                shared_argmax[threadIdx.x] = shared_argmax[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes out final argmax
    if (threadIdx.x == 0)
    {
        d_out[out_offset] = shared_argmax[0];
    }
}

template <typename T>
cudaError_t tomoArgmax(
    const T *d_in,
    size_t *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t out_size,
    size_t nd,
    cudaStream_t stream)
{
    // Validate shape/stride array lengths
    if (in_shape_len != nd || out_shape_len != nd ||
        in_strides_len != nd || out_strides_len != nd)
    {
        return cudaErrorInvalidValue;
    }

    // Allocate device buffers for shapes & strides
    size_t *d_in_shape, *d_out_shape;
    size_t *d_in_strides, *d_out_strides;
    CHECK_CUDA(cudaMallocAsync(&d_in_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_in_strides, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_strides, nd * sizeof(size_t), stream));

    // Copy to device
    CHECK_CUDA(cudaMemcpyAsync(d_in_shape, in_shape, nd * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_shape, out_shape, nd * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_in_strides, in_strides, nd * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_strides, out_strides, nd * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));

    // Launch kernel
    const int threads = 256; // must be power-of-two for that simple reduction
    const int blocks = static_cast<int>(out_size);
    // shared memory:
    //  each thread has a T plus a size_t
    const int smem_sz = threads * (int(sizeof(T)) + int(sizeof(size_t)));

    tomoArgmaxKernel<T><<<blocks, threads, smem_sz, stream>>>(
        d_in, d_in_shape, d_in_strides,
        d_out, d_out_shape, d_out_strides,
        out_size, nd);
    CHECK_CUDA(cudaGetLastError());

    // Cleanup
    CHECK_CUDA(cudaFreeAsync(d_in_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_out_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_in_strides, stream));
    CHECK_CUDA(cudaFreeAsync(d_out_strides, stream));

    return cudaSuccess;
}

template <typename T>
__global__ void tomoArgminKernel(
    const T *d_in,
    const size_t *in_shape,
    const size_t *in_strides,
    size_t *d_out,
    const size_t *out_shape,
    const size_t *out_strides,
    size_t out_size,
    size_t nd)
{
    size_t out_idx = blockIdx.x;
    if (out_idx >= out_size)
        return;

    // Compute output coords
    size_t out_coords[MAX_ND];
    {
        size_t tmp = out_idx;
        for (ptrdiff_t d = nd - 1; d >= 0; --d)
        {
            out_coords[d] = tmp % out_shape[d];
            tmp /= out_shape[d];
        }
    }

    // Output offset
    size_t out_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        out_offset += out_coords[d] * out_strides[d];
    }

    // Base offset in input
    size_t base_offset = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        size_t in_c = (out_shape[d] == 1) ? 0 : out_coords[d];
        base_offset += in_c * in_strides[d];
    }

    // Reduced dimensions
    size_t reduced_dims[MAX_ND];
    size_t reduced_sizes[MAX_ND];
    size_t num_reduced = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        if (out_shape[d] == 1 && in_shape[d] > 1)
        {
            reduced_dims[num_reduced] = d;
            reduced_sizes[num_reduced] = in_shape[d];
            num_reduced++;
        }
    }

    size_t N = 1;
    for (size_t j = 0; j < num_reduced; ++j)
    {
        N *= reduced_sizes[j];
    }

    // Initialize local min to +∞
    T local_min = deviceInfinity<T>(/*negative=*/false);
    size_t local_argmin = 0;

    for (size_t i = threadIdx.x; i < N; i += blockDim.x)
    {
        // Unravel i -> reduced_coords
        size_t reduced_coords[MAX_ND];
        {
            size_t tmp_i = i;
            for (ptrdiff_t j = num_reduced - 1; j >= 0; --j)
            {
                reduced_coords[j] = tmp_i % reduced_sizes[j];
                tmp_i /= reduced_sizes[j];
            }
        }

        // Input offset
        size_t offset = base_offset;
        for (size_t j = 0; j < num_reduced; ++j)
        {
            size_t d = reduced_dims[j];
            offset += reduced_coords[j] * in_strides[d];
        }

        // Compare & update local min
        T val = d_in[offset];
        if (val < local_min)
        {
            local_min = val;
            local_argmin = i;
        }
    }

    // Shared memory reduction
    extern __shared__ char shared_mem[];
    T *shared_min = reinterpret_cast<T *>(shared_mem);
    size_t *shared_argmin = reinterpret_cast<size_t *>(shared_min + blockDim.x);

    shared_min[threadIdx.x] = local_min;
    shared_argmin[threadIdx.x] = local_argmin;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            if (shared_min[threadIdx.x + s] < shared_min[threadIdx.x])
            {
                shared_min[threadIdx.x] = shared_min[threadIdx.x + s];
                shared_argmin[threadIdx.x] = shared_argmin[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // Write final argmin
    if (threadIdx.x == 0)
    {
        d_out[out_offset] = shared_argmin[0];
    }
}

template <typename T>
cudaError_t tomoArgmin(
    const T *d_in,
    size_t *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t out_size,
    size_t nd,
    cudaStream_t stream)
{
    if (in_shape_len != nd || out_shape_len != nd ||
        in_strides_len != nd || out_strides_len != nd)
    {
        return cudaErrorInvalidValue;
    }

    size_t *d_in_shape, *d_out_shape;
    size_t *d_in_strides, *d_out_strides;
    CHECK_CUDA(cudaMallocAsync(&d_in_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_shape, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_in_strides, nd * sizeof(size_t), stream));
    CHECK_CUDA(cudaMallocAsync(&d_out_strides, nd * sizeof(size_t), stream));

    CHECK_CUDA(cudaMemcpyAsync(d_in_shape, in_shape, nd * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_shape, out_shape, nd * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_in_strides, in_strides, nd * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_out_strides, out_strides, nd * sizeof(size_t),
                               cudaMemcpyHostToDevice, stream));

    const int threads = 256;
    const int blocks = static_cast<int>(out_size);
    const int smem_sz = threads * (int(sizeof(T)) + int(sizeof(size_t)));

    tomoArgminKernel<T><<<blocks, threads, smem_sz, stream>>>(
        d_in, d_in_shape, d_in_strides,
        d_out, d_out_shape, d_out_strides,
        out_size, nd);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaFreeAsync(d_in_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_out_shape, stream));
    CHECK_CUDA(cudaFreeAsync(d_in_strides, stream));
    CHECK_CUDA(cudaFreeAsync(d_out_strides, stream));
    return cudaSuccess;
}

template <typename T>
__global__ void tomoMaxPool2dForwardKernel(
    const T *__restrict__ input, // [N, C, H, W]
    T *__restrict__ output,      // [N, C, outH, outW]
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW)
{
    // Each thread corresponds to one element in output: [N, C, outH, outW]
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * C * outH * outW;
    if (idx >= total)
        return;

    // unravel idx -> (n, c, oh, ow)
    size_t ow = idx % outW;
    size_t tmp = idx / outW;
    size_t oh = tmp % outH;
    tmp /= outH;
    size_t c = tmp % C;
    size_t n = tmp / C;

    // compute the “start/end” in input image
    size_t in_start_h = oh * strideH - padH;
    size_t in_start_w = ow * strideW - padW;
    size_t in_end_h = in_start_h + kernelH;
    size_t in_end_w = in_start_w + kernelW;

    // T max_val = static_cast<T>(-FLT_MAX); // or -FLT_MAX for float
    T max_val = deviceInfinity<T>(true); // or -FLT_MAX for float
    for (size_t ih = in_start_h; ih < in_end_h; ih++)
    {
        for (size_t iw = in_start_w; iw < in_end_w; iw++)
        {
            if (ih < H && iw < W)
            {
                size_t in_index = (n * C + c) * (H * W) + ih * W + iw;
                T val = input[in_index];
                if (val > max_val)
                {
                    max_val = val;
                }
            }
        }
    }
    output[idx] = max_val;
}

template <typename T>
cudaError_t tomoMaxPool2dForward(
    const T *input,
    T *output,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW,
    cudaStream_t stream)
{
    size_t total = N * C * outH * outW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    tomoMaxPool2dForwardKernel<T><<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H, W,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        padH, padW);

    return cudaGetLastError();
}

template <typename T>
__global__ void tomoMaxPool2dBackwardKernel(
    const T *__restrict__ input,   // [N, C, H, W] (to find max indexes)
    const T *__restrict__ gradOut, // [N, C, outH, outW]
    T *gradIn,                     // [N, C, H, W] (to accumulate)
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW)
{
    // one thread per (n,c,oh,ow)
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * C * outH * outW;
    if (idx >= total)
        return;

    size_t ow = idx % outW;
    size_t tmp = idx / outW;
    size_t oh = tmp % outH;
    tmp /= outH;
    size_t c = tmp % C;
    size_t n = tmp / C;

    // figure out region in input
    size_t in_start_h = oh * strideH - padH;
    size_t in_start_w = ow * strideW - padW;
    size_t in_end_h = in_start_h + kernelH;
    size_t in_end_w = in_start_w + kernelW;

    // find which (ih, iw) had the max
    T max_val = deviceInfinity<T>(true);
    ptrdiff_t max_h = -1, max_w = -1;
    for (size_t ih = in_start_h; ih < in_end_h; ih++)
    {
        for (size_t iw = in_start_w; iw < in_end_w; iw++)
        {
            if (ih < H && iw < W)
            {
                size_t in_index = (n * C + c) * (H * W) + ih * W + iw;
                T val = input[in_index];
                if (val > max_val)
                {
                    max_val = val;
                    max_h = (ptrdiff_t)ih;
                    max_w = (ptrdiff_t)iw;
                }
            }
        }
    }

    // Add the upstream gradient to that max location
    T grad_val = gradOut[idx];
    if (max_h >= 0 && max_w >= 0)
    {
        size_t in_index = (n * C + c) * (H * W) + max_h * W + max_w;

        if constexpr (std::is_same_v<T, __half_raw>)
        {
            atomicAdd(reinterpret_cast<__half *>(&gradIn[in_index]),
                      static_cast<__half>(grad_val));
        }
        else if constexpr (std::is_same_v<T, __nv_bfloat16_raw>)
        {
            atomicAdd(reinterpret_cast<__nv_bfloat16 *>(&gradIn[in_index]),
                      static_cast<__nv_bfloat16>(grad_val));
        }
        else
        {
            atomicAdd(&gradIn[in_index], grad_val);
        }
    }
}

template <typename T>
cudaError_t tomoMaxPool2dBackward(
    const T *input,
    const T *gradOut,
    T *gradIn,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW,
    cudaStream_t stream)
{
    // zero gradIn first
    cudaError_t err = cudaMemsetAsync(gradIn, 0, N * C * H * W * sizeof(T), stream);
    if (err != cudaSuccess)
        return err;

    size_t total = (size_t)N * C * outH * outW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    tomoMaxPool2dBackwardKernel<T><<<blocks, threads, 0, stream>>>(
        input, gradOut, gradIn,
        N, C, H, W,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        padH, padW);

    return cudaGetLastError();
}

template <typename T>
__global__ void tomoAvgPool2dForwardKernel(
    const T *__restrict__ input, // [N, C, H, W]
    T *__restrict__ output,      // [N, C, outH, outW]
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * C * outH * outW;
    if (idx >= total)
        return;

    size_t ow = idx % outW;
    size_t tmp = idx / outW;
    size_t oh = tmp % outH;
    tmp /= outH;
    size_t c = tmp % C;
    size_t n = tmp / C;

    size_t in_start_h = oh * strideH - padH;
    size_t in_start_w = ow * strideW - padW;
    size_t in_end_h = in_start_h + kernelH;
    size_t in_end_w = in_start_w + kernelW;

    T sum_val = static_cast<T>(0.0);
    int count = 0;
    for (size_t ih = in_start_h; ih < in_end_h; ih++)
    {
        for (size_t iw = in_start_w; iw < in_end_w; iw++)
        {
            if (ih < H && iw < W)
            {
                size_t in_index = (n * C + c) * (H * W) + ih * W + iw;
                sum_val = sum_val + input[in_index];
                count++;
            }
        }
    }
    if (count > 0)
    {
        sum_val = sum_val / static_cast<T>(count);
    }
    output[idx] = sum_val;
}

template <typename T>
cudaError_t tomoAvgPool2dForward(
    const T *input,
    T *output,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW,
    cudaStream_t stream)
{
    size_t total = (size_t)N * C * outH * outW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    tomoAvgPool2dForwardKernel<T><<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H, W,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        padH, padW);

    return cudaGetLastError();
}

template <typename T>
__global__ void tomoAvgPool2dBackwardKernel(
    const T *__restrict__ gradOut, // [N, C, outH, outW]
    T *gradIn,                     // [N, C, H, W]
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * C * outH * outW;
    if (idx >= total)
        return;

    size_t ow = idx % outW;
    size_t tmp = idx / outW;
    size_t oh = tmp % outH;
    tmp /= outH;
    int c = tmp % C;
    int n = tmp / C;

    T gradVal = gradOut[idx];

    // figure out region in input
    size_t in_start_h = oh * strideH - padH;
    size_t in_start_w = ow * strideW - padW;
    size_t in_end_h = in_start_h + kernelH;
    size_t in_end_w = in_start_w + kernelW;

    // find how many valid elements are in that region
    size_t count = 0;
    for (size_t ih = in_start_h; ih < in_end_h; ih++)
    {
        for (size_t iw = in_start_w; iw < in_end_w; iw++)
        {
            if (ih < H && iw < W)
            {
                count++;
            }
        }
    }

    if (count == 0)
        return;
    T factor = gradVal / static_cast<T>(count);

    // distribute
    for (size_t ih = in_start_h; ih < in_end_h; ih++)
    {
        for (size_t iw = in_start_w; iw < in_end_w; iw++)
        {
            if (ih < H && iw < W)
            {
                size_t in_index = (n * C + c) * (H * W) + ih * W + iw;
                // atomicAdd(&gradIn[in_index], factor);

                if constexpr (std::is_same_v<T, __half_raw>)
                {
                    atomicAdd(reinterpret_cast<__half *>(&gradIn[in_index]),
                              static_cast<__half>(factor));
                }
                else if constexpr (std::is_same_v<T, __nv_bfloat16_raw>)
                {
                    atomicAdd(reinterpret_cast<__nv_bfloat16 *>(&gradIn[in_index]),
                              static_cast<__nv_bfloat16>(factor));
                }
                else
                {
                    atomicAdd(&gradIn[in_index], factor);
                }
            }
        }
    }
}

template <typename T>
cudaError_t tomoAvgPool2dBackward(
    const T *gradOut,
    T *gradIn,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW,
    cudaStream_t stream)
{
    // zero gradIn
    cudaError_t err = cudaMemsetAsync(gradIn, 0, (size_t)N * C * H * W * sizeof(T), stream);
    if (err != cudaSuccess)
        return err;

    size_t total = N * C * outH * outW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    tomoAvgPool2dBackwardKernel<T><<<blocks, threads, 0, stream>>>(
        gradOut, gradIn,
        N, C, H, W,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        padH, padW);

    return cudaGetLastError();
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
    return tomoBroadcastTo<__half_raw>(
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
    return tomoBroadcastTo<__nv_bfloat16_raw>(
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
    return tomoBroadcastTo<float>(
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
    return tomoBroadcastTo<double>(
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
    return tomoSumTo<__half_raw>(
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
    return tomoSumTo<__nv_bfloat16_raw>(
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
    return tomoSumTo<float>(
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
    return tomoSumTo<double>(
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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearH(
    __half_raw const *A, __half_raw const *B, size_t M, size_t K, size_t N, __half_raw const *bias, __half_raw *C,
    cudaStream_t stream)
{
    return tomoLinear<__half_raw>(
        A, B,
        M, K,
        N, bias,
        C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearB(
    __nv_bfloat16_raw const *A, __nv_bfloat16_raw const *B, size_t M, size_t K, size_t N, __nv_bfloat16_raw const *bias, __nv_bfloat16_raw *C,
    cudaStream_t stream)
{
    return tomoLinear<__nv_bfloat16_raw>(
        A, B,
        M, K,
        N, bias,
        C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearF(
    float const *A, float const *B, size_t M, size_t K, size_t N, float const *bias, float *C,
    cudaStream_t stream)
{
    return tomoLinear<float>(
        A, B,
        M, K,
        N, bias,
        C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearD(
    double const *A, double const *B, size_t M, size_t K, size_t N, double const *bias, double *C,
    cudaStream_t stream)
{
    return tomoLinear<double>(
        A, B,
        M, K,
        N, bias,
        C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearImpH(
    __half_raw const *A, __half_raw const *B, size_t M, size_t K, size_t N,
    __half_raw const *bias, __half_raw *C, cudaStream_t stream)
{
    return tomoLinearImp<__half_raw>(A, B, M, K, N, bias, C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearImpB(
    __nv_bfloat16_raw const *A, __nv_bfloat16_raw const *B, size_t M, size_t K, size_t N,
    __nv_bfloat16_raw const *bias, __nv_bfloat16_raw *C, cudaStream_t stream)
{
    return tomoLinearImp<__nv_bfloat16_raw>(A, B, M, K, N, bias, C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearImpF(
    float const *A, float const *B, size_t M, size_t K, size_t N,
    float const *bias, float *C, cudaStream_t stream)
{
    return tomoLinearImp<float>(A, B, M, K, N, bias, C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoLinearImpD(
    double const *A, double const *B, size_t M, size_t K, size_t N,
    double const *bias, double *C, cudaStream_t stream)
{
    return tomoLinearImp<double>(A, B, M, K, N, bias, C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeH(__half_raw const *A, size_t M, size_t N, __half_raw *C, cudaStream_t stream)
{
    return tomoTranspose<__half_raw>(
        A,
        M,
        N,
        C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeB(__nv_bfloat16_raw const *A, size_t M, size_t N, __nv_bfloat16_raw *C, cudaStream_t stream)
{
    return tomoTranspose<__nv_bfloat16_raw>(
        A,
        M,
        N,
        C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeF(float const *A, size_t M, size_t N, float *C, cudaStream_t stream)
{
    return tomoTranspose<float>(
        A,
        M,
        N,
        C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeD(double const *A, size_t M, size_t N, double *C, cudaStream_t stream)
{
    return tomoTranspose<double>(
        A,
        M,
        N,
        C, stream);
}

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
    cudaStream_t stream)
{
    return tomoMaxTo<__half_raw>(
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
    cudaStream_t stream)
{
    return tomoMaxTo<__nv_bfloat16_raw>(
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
    cudaStream_t stream)
{
    return tomoMaxTo<float>(
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
    cudaStream_t stream)
{
    return tomoMaxTo<double>(
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
    cudaStream_t stream)
{
    return tomoMinTo<__half_raw>(
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
    cudaStream_t stream)
{
    return tomoMinTo<__nv_bfloat16_raw>(
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
    cudaStream_t stream)
{
    return tomoMinTo<float>(
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
    cudaStream_t stream)
{
    return tomoMinTo<double>(
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
    cudaStream_t stream)
{
    return tomoTensordot<__half_raw>(
        d_a, d_b, d_out,
        a_shape, a_shape_len,
        b_shape, b_shape_len,
        out_shape, out_shape_len,
        a_strides, a_strides_len,
        b_strides, b_strides_len,
        out_strides, out_strides_len,
        contracted_axes_a, contracted_axes_a_len,
        contracted_axes_b, contracted_axes_b_len,
        a_nd, b_nd, out_nd,
        out_size, num_contracted,
        stream);
}

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
    cudaStream_t stream)
{
    return tomoTensordot<__nv_bfloat16_raw>(
        d_a, d_b, d_out,
        a_shape, a_shape_len,
        b_shape, b_shape_len,
        out_shape, out_shape_len,
        a_strides, a_strides_len,
        b_strides, b_strides_len,
        out_strides, out_strides_len,
        contracted_axes_a, contracted_axes_a_len,
        contracted_axes_b, contracted_axes_b_len,
        a_nd, b_nd, out_nd,
        out_size, num_contracted,
        stream);
}

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
    cudaStream_t stream)
{
    return tomoTensordot<float>(
        d_a, d_b, d_out,
        a_shape, a_shape_len,
        b_shape, b_shape_len,
        out_shape, out_shape_len,
        a_strides, a_strides_len,
        b_strides, b_strides_len,
        out_strides, out_strides_len,
        contracted_axes_a, contracted_axes_a_len,
        contracted_axes_b, contracted_axes_b_len,
        a_nd, b_nd, out_nd,
        out_size, num_contracted,
        stream);
}

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
    cudaStream_t stream)
{
    return tomoTensordot<double>(
        d_a, d_b, d_out,
        a_shape, a_shape_len,
        b_shape, b_shape_len,
        out_shape, out_shape_len,
        a_strides, a_strides_len,
        b_strides, b_strides_len,
        out_strides, out_strides_len,
        contracted_axes_a, contracted_axes_a_len,
        contracted_axes_b, contracted_axes_b_len,
        a_nd, b_nd, out_nd,
        out_size, num_contracted,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeExH(
    __half_raw const *d_in, __half_raw *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *out_shape, size_t const out_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const *out_strides, size_t const out_strides_len,
    size_t const *perm, size_t const perm_len,
    size_t const nd,
    size_t const in_size, size_t const out_size,
    cudaStream_t const stream)
{
    return tomoTransposeEx<__half_raw>(
        d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len,
        in_strides, in_strides_len, out_strides, out_strides_len,
        perm, perm_len, nd, in_size, out_size, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeExB(
    __nv_bfloat16_raw const *d_in, __nv_bfloat16_raw *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *out_shape, size_t const out_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const *out_strides, size_t const out_strides_len,
    size_t const *perm, size_t const perm_len,
    size_t const nd,
    size_t const in_size, size_t const out_size,
    cudaStream_t const stream)
{
    return tomoTransposeEx<__nv_bfloat16_raw>(
        d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len,
        in_strides, in_strides_len, out_strides, out_strides_len,
        perm, perm_len, nd, in_size, out_size, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeExF(
    float const *d_in, float *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *out_shape, size_t const out_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const *out_strides, size_t const out_strides_len,
    size_t const *perm, size_t const perm_len,
    size_t const nd,
    size_t const in_size, size_t const out_size,
    cudaStream_t const stream)
{
    return tomoTransposeEx<float>(
        d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len,
        in_strides, in_strides_len, out_strides, out_strides_len,
        perm, perm_len, nd, in_size, out_size, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeExD(
    double const *d_in, double *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *out_shape, size_t const out_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const *out_strides, size_t const out_strides_len,
    size_t const *perm, size_t const perm_len,
    size_t const nd,
    size_t const in_size, size_t const out_size,
    cudaStream_t const stream)
{
    return tomoTransposeEx<double>(
        d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len,
        in_strides, in_strides_len, out_strides, out_strides_len,
        perm, perm_len, nd, in_size, out_size, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoRollaxisH(
    __half_raw const *d_in, __half_raw *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const axis, size_t const start,
    size_t const nd, size_t const in_size, size_t const out_size,
    cudaStream_t const stream)
{
    return tomoRollaxis<__half_raw>(
        d_in, d_out, in_shape, in_shape_len, in_strides, in_strides_len,
        axis, start, nd, in_size, out_size, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoRollaxisB(
    __nv_bfloat16_raw const *d_in, __nv_bfloat16_raw *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const axis, size_t const start,
    size_t const nd, size_t const in_size, size_t const out_size,
    cudaStream_t const stream)
{
    return tomoRollaxis<__nv_bfloat16_raw>(
        d_in, d_out, in_shape, in_shape_len, in_strides, in_strides_len,
        axis, start, nd, in_size, out_size, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoRollaxisF(
    float const *d_in, float *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const axis, size_t const start,
    size_t const nd, size_t const in_size, size_t const out_size,
    cudaStream_t const stream)
{
    return tomoRollaxis<float>(
        d_in, d_out, in_shape, in_shape_len, in_strides, in_strides_len,
        axis, start, nd, in_size, out_size, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoRollaxisD(
    double const *d_in, double *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const axis, size_t const start,
    size_t const nd, size_t const in_size, size_t const out_size,
    cudaStream_t const stream)
{
    return tomoRollaxis<double>(
        d_in, d_out, in_shape, in_shape_len, in_strides, in_strides_len,
        axis, start, nd, in_size, out_size, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwapaxesH(
    __half_raw const *d_in, __half_raw *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t axis1, size_t axis2,
    size_t nd, size_t in_size, size_t out_size,
    cudaStream_t stream)
{
    return tomoSwapaxes<__half_raw>(
        d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len,
        in_strides, in_strides_len, out_strides, out_strides_len,
        axis1, axis2, nd, in_size, out_size, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwapaxesB(
    __nv_bfloat16_raw const *d_in, __nv_bfloat16_raw *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t axis1, size_t axis2,
    size_t nd, size_t in_size, size_t out_size,
    cudaStream_t stream)
{
    return tomoSwapaxes<__nv_bfloat16_raw>(
        d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len,
        in_strides, in_strides_len, out_strides, out_strides_len,
        axis1, axis2, nd, in_size, out_size, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwapaxesF(
    float const *d_in, float *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t axis1, size_t axis2,
    size_t nd, size_t in_size, size_t out_size,
    cudaStream_t stream)
{
    return tomoSwapaxes<float>(
        d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len,
        in_strides, in_strides_len, out_strides, out_strides_len,
        axis1, axis2, nd, in_size, out_size, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoSwapaxesD(
    double const *d_in, double *d_out,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t axis1, size_t axis2,
    size_t nd, size_t in_size, size_t out_size,
    cudaStream_t stream)
{
    return tomoSwapaxes<double>(
        d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len,
        in_strides, in_strides_len, out_strides, out_strides_len,
        axis1, axis2, nd, in_size, out_size, stream);
}

// __half version for im2col
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2colH(
    __half_raw const *d_img, __half_raw *d_col,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dy, size_t dx,
    cudaStream_t stream)
{
    return tomoIm2col<__half_raw>(
        d_img, d_col,
        n, c, h, w,
        kh, kw,
        sy, sx,
        ph, pw,
        dy, dx,
        stream);
}

// __nv_bfloat16 version for im2col
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2colB(
    __nv_bfloat16_raw const *d_img, __nv_bfloat16_raw *d_col,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dy, size_t dx,
    cudaStream_t stream)
{
    return tomoIm2col<__nv_bfloat16_raw>(
        d_img, d_col,
        n, c, h, w,
        kh, kw,
        sy, sx,
        ph, pw,
        dy, dx,
        stream);
}

// float version for im2col
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2colF(
    float const *d_img, float *d_col,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dy, size_t dx,
    cudaStream_t stream)
{
    return tomoIm2col<float>(
        d_img, d_col,
        n, c, h, w,
        kh, kw,
        sy, sx,
        ph, pw,
        dy, dx,
        stream);
}

// double version for im2col
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2colD(
    double const *d_img, double *d_col,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dy, size_t dx,
    cudaStream_t stream)
{
    return tomoIm2col<double>(
        d_img, d_col,
        n, c, h, w,
        kh, kw,
        sy, sx,
        ph, pw,
        dy, dx,
        stream);
}

// __half version for col2im
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2imH(
    __half_raw const *d_col, __half_raw *d_img,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dx, size_t dy,
    cudaStream_t stream)
{
    return tomoCol2im<__half_raw>(
        d_col, d_img,
        n, c, h, w,
        kh, kw,
        sy, sx,
        ph, pw,
        dx, dy,
        stream);
}

// __nv_bfloat16 version for col2im
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2imB(
    __nv_bfloat16_raw const *d_col, __nv_bfloat16_raw *d_img,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dx, size_t dy,
    cudaStream_t stream)
{
    return tomoCol2im<__nv_bfloat16_raw>(
        d_col, d_img,
        n, c, h, w,
        kh, kw,
        sy, sx,
        ph, pw,
        dx, dy,
        stream);
}

// float version for col2im
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2imF(
    float const *d_col, float *d_img,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dx, size_t dy,
    cudaStream_t stream)
{
    return tomoCol2im<float>(
        d_col, d_img,
        n, c, h, w,
        kh, kw,
        sy, sx,
        ph, pw,
        dx, dy,
        stream);
}

// double version for col2im
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2imD(
    double const *d_col, double *d_img,
    size_t n, size_t c, size_t h, size_t w,
    size_t kh, size_t kw,
    size_t sy, size_t sx,
    size_t ph, size_t pw,
    size_t dx, size_t dy,
    cudaStream_t stream)
{
    return tomoCol2im<double>(
        d_col, d_img,
        n, c, h, w,
        kh, kw,
        sy, sx,
        ph, pw,
        dx, dy,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2col1dH(
    __half_raw const *d_in,
    __half_raw *d_col,
    size_t n, size_t c, size_t l,
    size_t k,
    size_t stride,
    size_t pad,
    size_t dilation,
    cudaStream_t stream)
{
    return tomoIm2col1d<__half_raw>(
        d_in, d_col,
        n, c, l, k,
        stride, pad, dilation,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2col1dB(
    __nv_bfloat16_raw const *d_in,
    __nv_bfloat16_raw *d_col,
    size_t n, size_t c, size_t l,
    size_t k,
    size_t stride,
    size_t pad,
    size_t dilation,
    cudaStream_t stream)
{
    return tomoIm2col1d<__nv_bfloat16_raw>(
        d_in, d_col,
        n, c, l, k,
        stride, pad, dilation,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2col1dF(
    float const *d_in,
    float *d_col,
    size_t n, size_t c, size_t l,
    size_t k,
    size_t stride,
    size_t pad,
    size_t dilation,
    cudaStream_t stream)
{
    return tomoIm2col1d<float>(
        d_in, d_col,
        n, c, l, k,
        stride, pad, dilation,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoIm2col1dD(
    double const *d_in,
    double *d_col,
    size_t n, size_t c, size_t l,
    size_t k,
    size_t stride,
    size_t pad,
    size_t dilation,
    cudaStream_t stream)
{
    return tomoIm2col1d<double>(
        d_in, d_col,
        n, c, l, k,
        stride, pad, dilation,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2im1dH(
    __half_raw const *d_col,
    __half_raw *d_in,
    size_t n, size_t c, size_t l,
    size_t k,
    size_t stride,
    size_t pad,
    size_t dilation,
    cudaStream_t stream)
{
    return tomoCol2im1d<__half_raw>(
        d_col, d_in,
        n, c, l, k,
        stride, pad, dilation,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2im1dB(
    __nv_bfloat16_raw const *d_col,
    __nv_bfloat16_raw *d_in,
    size_t n, size_t c, size_t l,
    size_t k,
    size_t stride,
    size_t pad,
    size_t dilation,
    cudaStream_t stream)
{
    return tomoCol2im1d<__nv_bfloat16_raw>(
        d_col, d_in,
        n, c, l, k,
        stride, pad, dilation,
        stream);
}
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2im1dF(
    float const *d_col,
    float *d_in,
    size_t n, size_t c, size_t l,
    size_t k,
    size_t stride,
    size_t pad,
    size_t dilation,
    cudaStream_t stream)
{
    return tomoCol2im1d<float>(
        d_col, d_in,
        n, c, l, k,
        stride, pad, dilation,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoCol2im1dD(
    double const *d_col,
    double *d_in,
    size_t n, size_t c, size_t l,
    size_t k,
    size_t stride,
    size_t pad,
    size_t dilation,
    cudaStream_t stream)
{
    return tomoCol2im1d<double>(
        d_col, d_in,
        n, c, l, k,
        stride, pad, dilation,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgmaxH(
    const __half_raw *d_in,
    size_t *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t const *out_strides,
    size_t out_strides_len,
    size_t out_size,
    size_t nd,
    cudaStream_t stream)
{
    return tomoArgmax<__half_raw>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        out_size, nd, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgmaxB(
    const __nv_bfloat16_raw *d_in,
    size_t *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t const *out_strides,
    size_t out_strides_len,
    size_t out_size,
    size_t nd,
    cudaStream_t stream)
{
    return tomoArgmax<__nv_bfloat16_raw>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        out_size, nd, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgmaxF(
    const float *d_in,
    size_t *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t const *out_strides,
    size_t out_strides_len,
    size_t out_size,
    size_t nd,
    cudaStream_t stream)
{
    return tomoArgmax<float>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        out_size, nd, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgmaxD(
    const double *d_in,
    size_t *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t const *out_strides,
    size_t out_strides_len,
    size_t out_size,
    size_t nd,
    cudaStream_t stream)
{
    return tomoArgmax<double>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        out_size, nd, stream);
}

// Argmin wrappers

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgminH(
    const __half_raw *d_in,
    size_t *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t const *out_strides,
    size_t out_strides_len,
    size_t out_size,
    size_t nd,
    cudaStream_t stream)
{
    return tomoArgmin<__half_raw>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        out_size, nd, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgminB(
    const __nv_bfloat16_raw *d_in,
    size_t *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t const *out_strides,
    size_t out_strides_len,
    size_t out_size,
    size_t nd,
    cudaStream_t stream)
{
    return tomoArgmin<__nv_bfloat16_raw>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        out_size, nd, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgminF(
    const float *d_in,
    size_t *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t const *out_strides,
    size_t out_strides_len,
    size_t out_size,
    size_t nd,
    cudaStream_t stream)
{
    return tomoArgmin<float>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        out_size, nd, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoArgminD(
    const double *d_in,
    size_t *d_out,
    size_t const *in_shape,
    size_t in_shape_len,
    size_t const *out_shape,
    size_t out_shape_len,
    size_t const *in_strides,
    size_t in_strides_len,
    size_t const *out_strides,
    size_t out_strides_len,
    size_t out_size,
    size_t nd,
    cudaStream_t stream)
{
    return tomoArgmin<double>(
        d_in, d_out,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        out_size, nd, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxPool2dForwardH(
    __half_raw const *d_in, __half_raw *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoMaxPool2dForward<__half_raw>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxPool2dForwardB(
    __nv_bfloat16_raw const *d_in, __nv_bfloat16_raw *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoMaxPool2dForward<__nv_bfloat16_raw>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxPool2dForwardF(
    float const *d_in, float *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoMaxPool2dForward<float>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxPool2dForwardD(
    double const *d_in, double *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoMaxPool2dForward<double>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxPool2dBackwardH(
    const __half_raw *input,
    const __half_raw *gradOut,
    __half_raw *gradIn,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW,
    cudaStream_t stream)
{
    return tomoMaxPool2dBackward<__half_raw>(
        input, gradOut, gradIn,
        N, C, H, W,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        padH, padW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxPool2dBackwardB(
    const __nv_bfloat16_raw *input,
    const __nv_bfloat16_raw *gradOut,
    __nv_bfloat16_raw *gradIn,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW,
    cudaStream_t stream)
{
    return tomoMaxPool2dBackward<__nv_bfloat16_raw>(
        input, gradOut, gradIn,
        N, C, H, W,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        padH, padW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxPool2dBackwardF(
    const float *input,
    const float *gradOut,
    float *gradIn,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW,
    cudaStream_t stream)
{
    return tomoMaxPool2dBackward<float>(
        input, gradOut, gradIn,
        N, C, H, W,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        padH, padW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoMaxPool2dBackwardD(
    const double *input,
    const double *gradOut,
    double *gradIn,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kernelH, size_t kernelW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW,
    cudaStream_t stream)
{
    return tomoMaxPool2dBackward<double>(
        input, gradOut, gradIn,
        N, C, H, W,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        padH, padW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAvgPool2dForwardH(
    __half_raw const *d_in, __half_raw *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoAvgPool2dForward<__half_raw>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAvgPool2dForwardB(
    __nv_bfloat16_raw const *d_in, __nv_bfloat16_raw *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoAvgPool2dForward<__nv_bfloat16_raw>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAvgPool2dForwardF(
    float const *d_in, float *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoAvgPool2dForward<float>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAvgPool2dForwardD(
    double const *d_in, double *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoAvgPool2dForward<double>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAvgPool2dBackwardH(
    __half_raw const *d_in, __half_raw *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoAvgPool2dBackward<__half_raw>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAvgPool2dBackwardB(
    __nv_bfloat16_raw const *d_in, __nv_bfloat16_raw *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoAvgPool2dBackward<__nv_bfloat16_raw>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAvgPool2dBackwardF(
    float const *d_in, float *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoAvgPool2dBackward<float>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoAvgPool2dBackwardD(
    double const *d_in, double *d_out,
    size_t N, size_t C, size_t H, size_t W,
    size_t outH, size_t outW,
    size_t kH, size_t kW,
    size_t sH, size_t sW,
    size_t pH, size_t pW,
    cudaStream_t stream)
{
    return tomoAvgPool2dBackward<double>(
        d_in, d_out,
        N, C, H, W,
        outH, outW,
        kH, kW,
        sH, sW,
        pH, pW,
        stream);
}