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

    const F = tm.BF16;
    // const F = f16;

    var device_tensor = tm.tensor.GPUTensor(F, 2){};
    try device_tensor.initAsync(.{ 30, 40 }, &stream);
    defer device_tensor.deinitAsync(&stream);

    var host_tensor = try tm.tensor.CPUTensor(F, 2).init(allocator, .{ 30, 40 });
    defer host_tensor.deinit(allocator);

    for (0..30) |i| {
        for (0..40) |j| {
            // host_tensor.at(.{ i, j }).* = @floatFromInt(i * 4 + j);
            host_tensor.at(.{ i, j }).* = F.fromF64(@floatFromInt(i * 4 + j));
        }
    }

    try stream.sync();

    try device_tensor.writeFromHostAsync(host_tensor.data, 0, &stream);

    try device_tensor.sin(&stream);
    try device_tensor.cos(&stream);
    try device_tensor.relu(&stream);

    try stream.sync();

    try host_tensor.writeFromDevice(device_tensor.ptr.?, device_tensor.getLen(), 0, &stream);
    try stream.sync();

    var res = try host_tensor.reshape(allocator, 1, .{30 * 40});
    defer res.deinit(allocator);

    std.debug.print("{d}", .{res});
}
