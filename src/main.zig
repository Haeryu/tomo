const tm = @import("tomo_lib");
const std = @import("std");

pub fn main() !void {
    // var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer _ = gpa.deinit();

    // allocator = gpa.allocator();
    const allocator = tm.allocator.cuda_pinned_allocator;

    var stream = try tm.stream.Stream.create();
    defer {
        stream.sync() catch {};
        stream.destroy();
    }

    var cuda_context = try tm.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    const F = f32;

    var device_tensor = tm.tensor.GPUTensor(F, 2){};
    try device_tensor.initAsync(.{ 30, 40 }, &stream);
    defer device_tensor.deinitAsync(&stream);

    var device_tensor_t_res = tm.tensor.GPUTensor(F, 2){};
    try device_tensor_t_res.initAsync(.{ 40, 30 }, &stream);
    defer device_tensor_t_res.deinitAsync(&stream);

    var device_tensor_t_doubled_res = tm.tensor.GPUTensor(F, 2){};
    try device_tensor_t_doubled_res.initAsync(.{ 40, 30 }, &stream);
    defer device_tensor_t_doubled_res.deinitAsync(&stream);

    var device_tensor_matmul_t_res = tm.tensor.GPUTensor(F, 2){};
    try device_tensor_matmul_t_res.initAsync(.{ 30, 30 }, &stream);
    defer device_tensor_matmul_t_res.deinitAsync(&stream);

    var device_tensor_matmul_res = tm.tensor.GPUTensor(F, 2){};
    try device_tensor_matmul_res.initAsync(.{ 30, 30 }, &stream);
    defer device_tensor_matmul_res.deinitAsync(&stream);

    var host_tensor = try tm.tensor.CPUTensor(F, 2).init(allocator, .{ 30, 40 });
    defer host_tensor.deinit(allocator);

    var host_tensor_t_res = try tm.tensor.CPUTensor(F, 2).init(allocator, .{ 40, 30 });
    defer host_tensor_t_res.deinit(allocator);

    var host_tensor_matmul_res = try tm.tensor.CPUTensor(F, 2).init(allocator, .{ 30, 30 });
    defer host_tensor_matmul_res.deinit(allocator);

    for (0..30) |i| {
        for (0..40) |j| {
            host_tensor.at(.{ i, j }).* = @floatFromInt(i * 4 + j);
        }
    }

    try stream.sync();

    try device_tensor.writeFromHostAsync(host_tensor.data, 0, &stream);

    try device_tensor.transpose(&cuda_context, &stream, &device_tensor_t_res);

    try device_tensor_t_res.mul(&device_tensor_t_res, &cuda_context, &stream, &device_tensor_t_doubled_res);

    try device_tensor.matmulTransposed(&device_tensor_t_res, &cuda_context, &stream, &device_tensor_matmul_t_res);

    try device_tensor_matmul_t_res.transpose(&cuda_context, &stream, &device_tensor_matmul_res);

    try device_tensor_matmul_res.sin(&cuda_context, &stream);
    //try device_tensor_matmul_res.cos(&cuda_context, &stream);
    // try device_tensor_matmul_res.tan(&cuda_context, &stream);
    //try device_tensor_matmul_res.inv(&cuda_context, &stream);
    //  try device_tensor_matmul_res.relu(&cuda_context, &stream);
    // try device_tensor_matmul_res.addAssign(&device_tensor_matmul_t_res, &cuda_context, &stream);
    // try device_tensor_matmul_res.subAssign(&device_tensor_matmul_t_res, &cuda_context, &stream);
    // try device_tensor_matmul_res.mulAssign(&device_tensor_matmul_t_res, &cuda_context, &stream);
    // try device_tensor_matmul_res.divAssign(&device_tensor_matmul_t_res, &cuda_context, &stream);
    // try device_tensor_matmul_res.scale(10.0, &cuda_context, &stream);
    // try device_tensor_matmul_res.sqrt(&cuda_context, &stream);
    // try device_tensor_matmul_res.square(&cuda_context, &stream);
    // try device_tensor_matmul_res.softmax(&cuda_context, &stream);

    // var sum: f32 = undefined;
    // try device_tensor_matmul_res.l1Norm(&cuda_context, &stream, &sum);

    try device_tensor_matmul_res.pow(2, &cuda_context, &stream);
    try device_tensor_matmul_res.powf(0.5, &cuda_context, &stream);

    try host_tensor_t_res.writeFromDevice(device_tensor_t_res.ptr.?, device_tensor_t_res.getLen(), 0, &stream);
    try host_tensor_matmul_res.writeFromDevice(device_tensor_matmul_res.ptr.?, device_tensor_matmul_res.getLen(), 0, &stream);

    try stream.sync();

    var new = try host_tensor_matmul_res.reshape(allocator, 1, .{30 * 30});
    defer new.deinit(allocator);

    // try host_tensor.print("d", stdout);
    // try host_tensor_t_res.print("d", stdout);
    // try host_tensor_matmul_t_res.print("d", stdout);

    // std.debug.print("{d}\n", .{host_tensor});
    // std.debug.print("{d}\n", .{host_tensor_t_res});
    std.debug.print("{d}\n", .{new});
    // std.debug.print("{d}\n", .{sum});
}
