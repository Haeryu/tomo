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

__global__ void tomoTransposeKernelH(__half_raw const *A, size_t M, size_t N, __half_raw *C)
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

__global__ void tomoTransposeKernelB(__nv_bfloat16_raw const *A, size_t M, size_t N, __nv_bfloat16_raw *C)
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

__global__ void tomoTransposeKernelF(float const *A, size_t M, size_t N, float *C)
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

__global__ void tomoTransposeKernelD(double const *A, size_t M, size_t N, double *C)
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
cudaError_t tomoTranspose(T const *A, size_t M, size_t N, T *C, cudaStream_t stream)
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
cudaError_t tomoMaxToErr(const T *d_in,
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
        tomoMaxTo<T>(d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len, in_strides, in_strides_len, out_strides, out_strides_len, in_size, out_size, nd, stream);
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
cudaError_t tomoMinToErr(const T *d_in,
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
        tomoMinTo<T>(d_in, d_out, in_shape, in_shape_len, out_shape, out_shape_len, in_strides, in_strides_len, out_strides, out_strides_len, in_size, out_size, nd, stream);
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

// Error-handling wrapper
template <typename T>
cudaError_t tomoTensordotErr(
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
    try
    {
        return tomoTensordot<T>(
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

template <typename T>
cudaError_t tomoTransposeExErr(
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
    try
    {
        return tomoTransposeEx<T>(
            d_in, d_out,
            in_shape, in_shape_len,
            out_shape, out_shape_len,
            in_strides, in_strides_len,
            out_strides, out_strides_len,
            perm, perm_len,
            nd, in_size, out_size,
            stream);
    }
    catch (const thrust::system_error &e)
    {
        if (e.code().category() == thrust::cuda_category())
        {
            return static_cast<cudaError_t>(e.code().value());
        }
        return cudaErrorUnknown;
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }
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
cudaError_t tomoRollaxisErr(
    T const *d_in, T *d_out,
    size_t const *in_shape, size_t const in_shape_len,
    size_t const *in_strides, size_t const in_strides_len,
    size_t const axis, size_t const start,
    size_t const nd, size_t const in_size, size_t const out_size,
    cudaStream_t const stream)
{
    try
    {
        return tomoRollaxis<T>(
            d_in, d_out, in_shape, in_shape_len, in_strides, in_strides_len,
            axis, start, nd, in_size, out_size, stream);
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }
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
    // Total output elements = n * c * kh * kw * out_h * out_w
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * c * kh * kw * out_h * out_w;
    if (idx >= total)
        return;

    // Unravel idx into 6 coordinates: [n_idx, c_idx, k_y, k_x, out_y, out_x]
    size_t dims[6] = {n, c, kh, kw, out_h, out_w};
    size_t coords[6];
    size_t tmp = idx;
    for (ptrdiff_t d = 5; d >= 0; --d)
    {
        coords[d] = tmp % dims[d];
        tmp /= dims[d];
    }
    size_t n_idx = coords[0];
    size_t c_idx = coords[1];
    size_t k_y = coords[2];
    size_t k_x = coords[3];
    size_t out_y = coords[4];
    size_t out_x = coords[5];

    // Compute corresponding input coordinates
    int in_y = static_cast<int>(k_y * dy + out_y * sy) - static_cast<int>(ph);
    int in_x = static_cast<int>(k_x * dx + out_x * sx) - static_cast<int>(pw);
    T value = (T)0.0;
    if (in_y >= 0 && in_y < static_cast<int>(h) &&
        in_x >= 0 && in_x < static_cast<int>(w))
    {
        size_t img_idx = n_idx * (c * h * w) +
                         c_idx * (h * w) +
                         in_y * w +
                         in_x;
        value = d_img[img_idx];
    }
    d_col[idx] = value;
}

//------------------------------------------------------------------------------
// tomoCol2imKernel
//------------------------------------------------------------------------------
// Reconstructs an image of shape (n, c, h, w) by summing contributions from
// a column tensor with shape (n, c, kh, kw, out_h, out_w) produced by im2col.
template <typename T>
__global__ void tomoCol2imKernel(
    T const *d_col, T *d_img,
    size_t const n, size_t const c, size_t const h, size_t const w,
    size_t const kh, size_t const kw,
    size_t const out_h, size_t const out_w,
    size_t const sy, size_t const sx,
    size_t const ph, size_t const pw,
    size_t const dx, size_t const dy)
{
    // Each thread computes one pixel in the output image.
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n * c * h * w;
    if (idx >= total)
        return;

    // Unravel idx into image coordinates: [n_idx, c_idx, y, x]
    size_t n_idx = idx / (c * h * w);
    size_t rem = idx % (c * h * w);
    size_t c_idx = rem / (h * w);
    size_t rem2 = rem % (h * w);
    size_t y = rem2 / w;
    size_t x = rem2 % w;

    T sum = (T)0.0f;
    // Loop over kernel positions that may contribute to (y, x)
    for (size_t ky = 0; ky < kh; ++ky)
    {
        int out_y = static_cast<int>(y) + static_cast<int>(ph) - static_cast<int>(ky * dy);
        if (out_y < 0 || out_y >= static_cast<int>(out_h * sy))
            continue;
        if (out_y % sy != 0)
            continue;
        out_y /= sy;
        for (size_t kx = 0; kx < kw; ++kx)
        {
            int out_x = static_cast<int>(x) + static_cast<int>(pw) - static_cast<int>(kx * dx);
            if (out_x < 0 || out_x >= static_cast<int>(out_w * sx))
                continue;
            if (out_x % sx != 0)
                continue;
            out_x /= sx;
            // Compute index into d_col assuming layout: (n, c, kh, kw, out_h, out_w)
            size_t col_idx = (((((n_idx * c + c_idx) * kh + ky) * kw + kx) * out_h + out_y) * out_w + out_x);
            sum = sum + d_col[col_idx];
        }
    }
    d_img[idx] = sum;
}

//------------------------------------------------------------------------------
// Host wrappers following your tomoXXXX style
//------------------------------------------------------------------------------

// tomoIm2col
template <typename T>
cudaError_t tomoIm2col(
    T const *d_img, T *d_col,
    size_t const n, size_t const c, size_t const h, size_t const w,
    size_t const kh, size_t const kw,
    size_t const sy, size_t const sx,
    size_t const ph, size_t const pw,
    size_t const dy, size_t const dx,
    cudaStream_t stream)
{
    size_t const out_h = (h + 2 * ph - kh) / sy + 1;
    size_t const out_w = (w + 2 * pw - kw) / sx + 1;
    size_t const total = n * c * kh * kw * out_h * out_w;
    const int threads = 256;
    const int blocks = (int)(total + threads - 1) / threads;
    tomoIm2colKernel<T><<<blocks, threads, 0, stream>>>(
        d_img, d_col,
        n, c, h, w,
        kh, kw, out_h, out_w,
        sy, sx, ph, pw,
        dy, dx);
    CHECK_CUDA(cudaGetLastError());
    return cudaSuccess;
}

// tomoCol2im
template <typename T>
cudaError_t tomoCol2im(
    T const *d_col, T *d_img,
    size_t const n, size_t const c, size_t const h, size_t const w,
    size_t const kh, size_t const kw,
    size_t const sy, size_t const sx,
    size_t const ph, size_t const pw,
    size_t const dx, size_t const dy,
    cudaStream_t stream)
{
    size_t const out_h = (h + 2 * ph - kh) / sy + 1;
    size_t const out_w = (w + 2 * pw - kw) / sx + 1;
    size_t const total = n * c * h * w;
    const int threads = 256;
    const int blocks = (int)(total + threads - 1) / threads;
    tomoCol2imKernel<T><<<blocks, threads, 0, stream>>>(
        d_col, d_img,
        n, c, h, w,
        kh, kw, out_h, out_w,
        sy, sx, ph, pw,
        dx, dy);
    CHECK_CUDA(cudaGetLastError());
    return cudaSuccess;
}

//------------------------------------------------------------------------------
// Error-wrapped versions (like tomoMinToErr)
//------------------------------------------------------------------------------

template <typename T>
cudaError_t tomoIm2colErr(
    T const *d_img, T *d_col,
    size_t const n, size_t const c, size_t const h, size_t const w,
    size_t const kh, size_t const kw,
    size_t const sy, size_t const sx,
    size_t const ph, size_t const pw,
    size_t const dy, size_t const dx,
    cudaStream_t stream)
{
    try
    {
        tomoIm2col<T>(d_img, d_col,
                      n, c, h, w,
                      kh, kw,
                      sy, sx,
                      ph, pw,
                      dy, dx, stream);
    }
    catch (const thrust::system_error &e)
    {
        if (e.code().category() == thrust::cuda_category())
            return static_cast<cudaError_t>(e.code().value());
        else
            return cudaErrorUnknown;
    }
    catch (...)
    {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

template <typename T>
cudaError_t tomoCol2imErr(
    T const *d_col, T *d_img,
    size_t const n, size_t const c, size_t const h, size_t const w,
    size_t const kh, size_t const kw,
    size_t const sy, size_t const sx,
    size_t const ph, size_t const pw,
    size_t const dx, size_t const dy,
    cudaStream_t stream)
{
    try
    {
        tomoCol2im<T>(d_col, d_img,
                      n, c, h, w,
                      kh, kw,
                      sy, sx,
                      ph, pw,
                      dx, dy, stream);
    }
    catch (const thrust::system_error &e)
    {
        if (e.code().category() == thrust::cuda_category())
            return static_cast<cudaError_t>(e.code().value());
        else
            return cudaErrorUnknown;
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
    return tomoMaxToErr<__half_raw>(
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
    return tomoMaxToErr<__nv_bfloat16_raw>(
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
    return tomoMaxToErr<float>(
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
    return tomoMaxToErr<double>(
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
    return tomoMinToErr<__half_raw>(
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
    return tomoMinToErr<__nv_bfloat16_raw>(
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
    return tomoMinToErr<float>(
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
    return tomoMinToErr<double>(
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
    return tomoTensordotErr<__half_raw>(
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
    return tomoTensordotErr<__nv_bfloat16_raw>(
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
    return tomoTensordotErr<float>(
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
    return tomoTensordotErr<double>(
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
    return tomoTransposeExErr<__half_raw>(
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
    return tomoTransposeExErr<__nv_bfloat16_raw>(
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
    return tomoTransposeExErr<float>(
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
    return tomoTransposeExErr<double>(
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
    return tomoRollaxisErr<__half_raw>(
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
    return tomoRollaxisErr<__nv_bfloat16_raw>(
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
    return tomoRollaxisErr<float>(
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
    return tomoRollaxisErr<double>(
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
