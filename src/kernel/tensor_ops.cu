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

    // Compute sum over input elements mapping to this output index
    T sum = T(0);
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

__global__ void tomoTransposeKernelH(__half_raw const*A, size_t M, size_t N, __half_raw *C)
{
    // Shared memory to hold a tile of the input matrix
    __shared__ __half_raw tile_h[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflicts

    // Input coordinates (reading from A)
    auto x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    auto y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Load data into shared memory (coalesced read from A)
    if (y < M && x < N)
    {
        tile_h[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    else
    {
        tile_h[threadIdx.y][threadIdx.x] = (__half_raw)0.0; // Padding for out-of-bounds
    }

    __syncthreads();

    // Output coordinates (writing to C)
    auto tx = blockIdx.y * BLOCK_SIZE + threadIdx.x; // Swapped block indices
    auto ty = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    // Write transposed data to global memory (coalesced write to C)
    if (ty < N && tx < M)
    {
        C[tx * N + ty] = tile_h[threadIdx.x][threadIdx.y]; // Note swapped indices
    }
}

__global__ void tomoTransposeKernelB(__nv_bfloat16_raw const*A, size_t M, size_t N, __nv_bfloat16_raw *C)
{
    // Shared memory to hold a tile of the input matrix
    __shared__ __nv_bfloat16_raw tile_b[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflicts

    // Input coordinates (reading from A)
    auto x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    auto y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Load data into shared memory (coalesced read from A)
    if (y < M && x < N)
    {
        tile_b[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    else
    {
        tile_b[threadIdx.y][threadIdx.x] = (__nv_bfloat16_raw)0.0; // Padding for out-of-bounds
    }

    __syncthreads();

    // Output coordinates (writing to C)
    auto tx = blockIdx.y * BLOCK_SIZE + threadIdx.x; // Swapped block indices
    auto ty = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    // Write transposed data to global memory (coalesced write to C)
    if (ty < N && tx < M)
    {
        C[tx * N + ty] = tile_b[threadIdx.x][threadIdx.y]; // Note swapped indices
    }
}

__global__ void tomoTransposeKernelF(float const*A, size_t M, size_t N, float *C)
{
    // Shared memory to hold a tile of the input matrix
    __shared__ float tile_f[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflicts

    // Input coordinates (reading from A)
    auto x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    auto y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Load data into shared memory (coalesced read from A)
    if (y < M && x < N)
    {
        tile_f[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    else
    {
        tile_f[threadIdx.y][threadIdx.x] = 0.0f; // Padding for out-of-bounds
    }

    __syncthreads();

    // Output coordinates (writing to C)
    auto tx = blockIdx.y * BLOCK_SIZE + threadIdx.x; // Swapped block indices
    auto ty = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    // Write transposed data to global memory (coalesced write to C)
    if (ty < N && tx < M)
    {
        C[tx * N + ty] = tile_f[threadIdx.x][threadIdx.y]; // Note swapped indices
    }
}

__global__ void tomoTransposeKernelD(double const*A, size_t M, size_t N, double *C)
{
    // Shared memory to hold a tile of the input matrix
    __shared__ double tile_d[BLOCK_SIZE][BLOCK_SIZE + 1]; // +1 to avoid bank conflicts

    // Input coordinates (reading from A)
    auto x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    auto y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Load data into shared memory (coalesced read from A)
    if (y < M && x < N)
    {
        tile_d[threadIdx.y][threadIdx.x] = A[y * N + x];
    }
    else
    {
        tile_d[threadIdx.y][threadIdx.x] = 0.0; // Padding for out-of-bounds
    }

    __syncthreads();

    // Output coordinates (writing to C)
    auto tx = blockIdx.y * BLOCK_SIZE + threadIdx.x; // Swapped block indices
    auto ty = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    // Write transposed data to global memory (coalesced write to C)
    if (ty < N && tx < M)
    {
        C[tx * N + ty] = tile_d[threadIdx.x][threadIdx.y]; // Note swapped indices
    }
}

template <typename T>
cudaError_t tomoTranspose(T const*A, size_t M, size_t N, T *C, cudaStream_t stream)
{

    dim3 gridDim(((unsigned int)N + BLOCK_SIZE - 1) / BLOCK_SIZE, ((unsigned int)M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

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

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeH(__half_raw const*A, size_t M, size_t N, __half_raw *C, cudaStream_t stream)
{
    return tomoTranspose<__half_raw>(
        A,
        M,
        N,
        C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeB(__nv_bfloat16_raw const*A, size_t M, size_t N, __nv_bfloat16_raw *C, cudaStream_t stream)
{
    return tomoTranspose<__nv_bfloat16_raw>(
        A,
        M,
        N,
        C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeF(float const*A, size_t M, size_t N, float *C, cudaStream_t stream)
{
    return tomoTranspose<float>(
        A,
        M,
        N,
        C, stream);
}

TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoTransposeD(double const*A, size_t M, size_t N, double *C, cudaStream_t stream)
{
    return tomoTranspose<double>(
        A,
        M,
        N,
        C, stream);
}
