const tm = @import("tomo_lib");
const std = @import("std");

pub fn main() !void {
    const allocator = tm.allocator.cuda_pinned_allocator;

    var stream = try tm.stream.Stream.create();
    defer {
        stream.sync() catch {};
        stream.destroy();
    }

    var cuda_context = try tm.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    //const F = tm.BF16;
    const F = f32;

    const row1 = 2;
    const col1 = 3;

    const row2 = 3;
    const col2 = 4;

    var device_tensor1 = tm.tensor.GPUTensor(F, 2){};
    try device_tensor1.initAsync(.{ row1, col1 }, &stream);
    defer device_tensor1.deinitAsync(&stream);

    var host_tensor1 = try tm.tensor.CPUTensor(F, 2).init(allocator, .{ row1, col1 });
    defer host_tensor1.deinit(allocator);

    var device_tensor2 = tm.tensor.GPUTensor(F, 2){};
    try device_tensor2.initAsync(.{ row2, col2 }, &stream);
    defer device_tensor2.deinitAsync(&stream);

    var host_tensor2 = try tm.tensor.CPUTensor(F, 2).init(allocator, .{ row2, col2 });
    defer host_tensor2.deinit(allocator);

    var device_tensor_res = tm.tensor.GPUTensor(F, 2){};
    try device_tensor_res.initAsync(.{ col2, row1 }, &stream);
    defer device_tensor_res.deinitAsync(&stream);

    var host_tensor_res = try tm.tensor.CPUTensor(F, 2).init(allocator, .{ col2, row1 });
    defer host_tensor_res.deinit(allocator);

    for (0..row1) |i| {
        for (0..col1) |j| {
            host_tensor1.at(.{ i, j }).* = @floatFromInt(i * col1 + j);
        }
    }
    for (0..row2) |i| {
        for (0..col2) |j| {
            host_tensor2.at(.{ i, j }).* = @floatFromInt(i * col2 + j);
        }
    }

    try stream.sync();

    try device_tensor1.writeFromHostAsync(host_tensor1.data, 0, &stream);
    try device_tensor2.writeFromHostAsync(host_tensor2.data, 0, &stream);

    try device_tensor1.matmul(
        false,
        &device_tensor2,
        false,
        null,
        null,
        false,
        false,
        tm.c.CUBLAS_COMPUTE_32F,
        &stream,
        &cuda_context,
        &device_tensor_res,
    );

    try stream.sync();

    try host_tensor_res.writeFromDevice(
        device_tensor_res.ptr.?,
        device_tensor_res.getLen(),
        0,
        &stream,
    );

    std.debug.print("{d}", .{host_tensor1});
    std.debug.print("{d}", .{host_tensor2});
    std.debug.print("{d}", .{host_tensor_res});
}
