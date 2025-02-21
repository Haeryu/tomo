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

    const row = 300;
    const col = 400;

    var device_tensor = tm.tensor.GPUTensor(F, 2){};
    try device_tensor.initAsync(.{ row, col }, &stream);
    defer device_tensor.deinitAsync(&stream);

    var host_tensor = try tm.tensor.CPUTensor(F, 2).init(allocator, .{ row, col });
    defer host_tensor.deinit(allocator);

    for (0..row) |i| {
        for (0..col) |j| {
            // host_tensor.at(.{ i, j }).* = @floatFromInt(i * 4 + j);
            host_tensor.at(.{ i, j }).* = F.fromF64(@floatFromInt(i * 4 + j));
        }
    }

    try stream.sync();

    try device_tensor.writeFromHostAsync(host_tensor.data, 0, &stream);

    try device_tensor.sin(&stream);
    try device_tensor.cos(&stream);
    try device_tensor.relu(&stream);

    try device_tensor.softmax(&stream);

    var sum: F = undefined;
    try device_tensor.sumReduce(&stream, &sum);

    // try stream.sync();

    try host_tensor.writeFromDevice(device_tensor.ptr.?, device_tensor.getLen(), 0, &stream);
    try stream.sync();

    var res = try host_tensor.reshape(allocator, 1, .{row * col});
    defer res.deinit(allocator);

    std.debug.print("{d}\n", .{res});
    std.debug.print("{d}\n", .{sum});
}
