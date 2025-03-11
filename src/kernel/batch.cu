#define TOMO_OPS_EXPORTS
#include "tomo_dll.h"
#include "batch.h"

#include "float_op.cuh"

template <typename T>
__global__ void tomoGetItemKernel(
    T const *x, T *y,
    size_t const *in_shape, size_t const *out_shape,
    size_t const *in_strides, size_t const *out_strides,
    size_t const *starts, size_t const *steps,
    size_t nd, size_t out_size)
{
    size_t out_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_size)
    {
        return;
    }

    // Unravel out_idx to out_coords
    size_t out_coords[32]; // max_rank = 4
    size_t tmp = out_idx;
    for (ptrdiff_t d = (ptrdiff_t)nd - 1; d >= 0; --d)
    {
        out_coords[d] = tmp % out_shape[d];
        tmp /= out_shape[d];
    }

    // Compute corresponding input coordinates
    size_t in_coords[32];
    for (size_t d = 0; d < nd; ++d)
    {
        in_coords[d] = starts[d] + out_coords[d] * steps[d];
    }

    // Compute input index
    size_t in_idx = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        in_idx += in_coords[d] * in_strides[d];
    }

    // Copy value
    y[out_idx] = x[in_idx];
}

template <typename T>
__global__ void tomoGetItemGradKernel(
    T const *gy, T *gx,
    size_t const *in_shape, size_t const *out_shape,
    size_t const *in_strides, size_t const *out_strides,
    size_t const *starts, size_t const *steps,
    size_t nd, size_t out_size)
{
    size_t out_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_size)
    {
        return;
    }

    // Unravel out_idx to out_coords
    size_t out_coords[32];
    size_t tmp = out_idx;
    for (ptrdiff_t d = (ptrdiff_t)nd - 1; d >= 0; --d)
    {
        out_coords[d] = tmp % out_shape[d];
        tmp /= out_shape[d];
    }

    // Compute corresponding input coordinates
    size_t in_coords[32];
    for (size_t d = 0; d < nd; ++d)
    {
        in_coords[d] = starts[d] + out_coords[d] * steps[d];
    }

    // Compute input index
    size_t in_idx = 0;
    for (size_t d = 0; d < nd; ++d)
    {
        in_idx += in_coords[d] * in_strides[d];
    }

    // Add gradient (no overlap in standard slicing)
    gx[in_idx] = gx[in_idx] + gy[out_idx];
}

template <typename T>
cudaError_t tomoGetItem(
    T const *x, T *y,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream)
{

    size_t *d_in_shape, *d_out_shape, *d_in_strides, *d_out_strides, *d_starts, *d_steps;
    cudaError_t err;

    err = cudaMallocAsync(&d_in_shape, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMallocAsync(&d_out_shape, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMallocAsync(&d_in_strides, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMallocAsync(&d_out_strides, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMallocAsync(&d_starts, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMallocAsync(&d_steps, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }

    err = cudaMemcpyAsync(d_in_shape, in_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMemcpyAsync(d_out_shape, out_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMemcpyAsync(d_in_strides, in_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMemcpyAsync(d_out_strides, out_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMemcpyAsync(d_starts, starts, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMemcpyAsync(d_steps, steps, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }

    const int threads = 256;
    const int blocks = (out_size + threads - 1) / threads;
    tomoGetItemKernel<T><<<blocks, threads, 0, stream>>>(
        x, y, d_in_shape, d_out_shape, d_in_strides, d_out_strides, d_starts, d_steps, nd, out_size);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return err;
    }

    cudaFreeAsync(d_in_shape, stream);
    cudaFreeAsync(d_out_shape, stream);
    cudaFreeAsync(d_in_strides, stream);
    cudaFreeAsync(d_out_strides, stream);
    cudaFreeAsync(d_starts, stream);
    cudaFreeAsync(d_steps, stream);

    return cudaSuccess;
}

template <typename T>
cudaError_t tomoGetItemGrad(
    T const *gy, T *gx,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream)
{

    size_t *d_in_shape, *d_out_shape, *d_in_strides, *d_out_strides, *d_starts, *d_steps;
    cudaError_t err;

    err = cudaMallocAsync(&d_in_shape, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMallocAsync(&d_out_shape, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMallocAsync(&d_in_strides, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMallocAsync(&d_out_strides, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMallocAsync(&d_starts, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMallocAsync(&d_steps, nd * sizeof(size_t), stream);
    if (err != cudaSuccess)
    {
        return err;
    }

    err = cudaMemcpyAsync(d_in_shape, in_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMemcpyAsync(d_out_shape, out_shape, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMemcpyAsync(d_in_strides, in_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMemcpyAsync(d_out_strides, out_strides, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMemcpyAsync(d_starts, starts, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }
    err = cudaMemcpyAsync(d_steps, steps, nd * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
    {
        return err;
    }

    int const threads = 256;
    int const blocks = (out_size + threads - 1) / threads;
    tomoGetItemGradKernel<T><<<blocks, threads, 0, stream>>>(
        gy, gx, d_in_shape, d_out_shape, d_in_strides, d_out_strides, d_starts, d_steps, nd, out_size);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        return err;
    }

    cudaFreeAsync(d_in_shape, stream);
    cudaFreeAsync(d_out_shape, stream);
    cudaFreeAsync(d_in_strides, stream);
    cudaFreeAsync(d_out_strides, stream);
    cudaFreeAsync(d_starts, stream);
    cudaFreeAsync(d_steps, stream);

    return cudaSuccess;
}

// ----- GetItem Wrappers -----
// Half
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemH(
    __half_raw const *x,
    __half_raw *y,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream)
{
    return tomoGetItem<__half_raw>(
        x, y,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        starts, starts_len,
        steps, steps_len,
        nd, out_size,
        stream);
}

// Bfloat16
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemB(
    __nv_bfloat16_raw const *x,
    __nv_bfloat16_raw *y,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream)
{
    return tomoGetItem<__nv_bfloat16_raw>(
        x, y,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        starts, starts_len,
        steps, steps_len,
        nd, out_size,
        stream);
}

// Float
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemF(
    float const *x,
    float *y,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream)
{
    return tomoGetItem<float>(
        x, y,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        starts, starts_len,
        steps, steps_len,
        nd, out_size,
        stream);
}

// Double
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemD(
    double const *x,
    double *y,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream)
{
    return tomoGetItem<double>(
        x, y,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        starts, starts_len,
        steps, steps_len,
        nd, out_size,
        stream);
}

// ----- GetItemGrad Wrappers -----
// Half
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemGradH(
    __half_raw const *gy,
    __half_raw *gx,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream)
{
    return tomoGetItemGrad<__half_raw>(
        gy, gx,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        starts, starts_len,
        steps, steps_len,
        nd, out_size,
        stream);
}

// Bfloat16
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemGradB(
    __nv_bfloat16_raw const *gy,
    __nv_bfloat16_raw *gx,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream)
{
    return tomoGetItemGrad<__nv_bfloat16_raw>(
        gy, gx,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        starts, starts_len,
        steps, steps_len,
        nd, out_size,
        stream);
}

// Float
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemGradF(
    float const *gy,
    float *gx,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream)
{
    return tomoGetItemGrad<float>(
        gy, gx,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        starts, starts_len,
        steps, steps_len,
        nd, out_size,
        stream);
}

// Double
TOMO_EXTERN_C TOMO_OPS_API cudaError_t tomoGetItemGradD(
    double const *gy,
    double *gx,
    size_t const *in_shape, size_t in_shape_len,
    size_t const *out_shape, size_t out_shape_len,
    size_t const *in_strides, size_t in_strides_len,
    size_t const *out_strides, size_t out_strides_len,
    size_t const *starts, size_t starts_len,
    size_t const *steps, size_t steps_len,
    size_t nd, size_t out_size,
    cudaStream_t stream)
{
    return tomoGetItemGrad<double>(
        gy, gx,
        in_shape, in_shape_len,
        out_shape, out_shape_len,
        in_strides, in_strides_len,
        out_strides, out_strides_len,
        starts, starts_len,
        steps, steps_len,
        nd, out_size,
        stream);
}
