const tm = @import("tomo_lib");
const std = @import("std");

// TODO: make Tensor operation small piece use Expression template -> fuse
pub fn main() !void {
    // const allocator = tm.allocator.cuda_pinned_allocator;

    var stream = try tm.stream.Stream.create();
    defer stream.destroy();

    var cuda_context = try tm.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    // const F = tm.BF16;
    const F = f32;

    const batch = 2;

    const row1 = 3;
    const col1 = 4;

    const row2 = 4;
    const col2 = 3;

    var device_tensor1 = try tm.tensor.GPUTensor(F).initAsync(&.{ batch, row1, col1 }, &stream);
    defer device_tensor1.deinitAsync(&stream);

    try device_tensor1.fill(1.0, &stream);

    var device_tensor2 = try tm.tensor.GPUTensor(F).initAsync(&.{ batch, row2, col2 }, &stream);
    defer device_tensor2.deinitAsync(&stream);
    try device_tensor2.fill(2.0, &stream);

    try device_tensor1.product(&device_tensor2, &stream);

    const Ep = tm.tensor.matmul_epilogue.Epilogue(void, void);

    var dot = try device_tensor1.matmul(
        false,
        &device_tensor2,
        false,
        null,
        false,
        tm.c.CUBLAS_COMPUTE_32F,
        1.0,
        0.0,
        Ep,
        Ep.Config{},
        &stream,
        &cuda_context,
        F,
    );
    defer dot.deinitAsync(&stream);

    var e_dot = dot.getErasedPtr();

    var host = try e_dot.asOriginal().toHost(tm.allocator.cuda_pinned_allocator, &stream);
    defer host.deinit(tm.allocator.cuda_pinned_allocator);

    try stream.sync();

    std.debug.print("{d}", .{host});
}
