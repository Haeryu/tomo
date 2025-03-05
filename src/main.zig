const tm = @import("tomo_lib");
const std = @import("std");

// TODO: make Tensor operation small piece use Expression template -> fuse
pub fn main() !void {
    // const allocator = tm.allocator.cuda_pinned_allocator;

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var stream = try tm.stream.Stream.create();
    defer stream.destroy();

    var cuda_context = try tm.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    // const F = tm.BF16;
    const F = f32;

    const row1 = 3;
    const col1 = 4;

    const row2 = 6;
    const col2 = 8;

    var device_tensor1 = try tm.tensor.GPUTensor(F).initAsync(&.{ row1, col1 }, &stream);
    defer device_tensor1.deinitAsync(&stream);
    try device_tensor1.fillHeUniform(&cuda_context, &stream);
    //try device_tensor1.writeFromHostAsync(&.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }, 0, &stream);

    var broad = try device_tensor1.broadCastTo(&.{ row2, col2 }, &stream);
    defer broad.deinitAsync(&stream);

    var sum = try device_tensor1.sum(allocator, &.{1}, false, &stream);
    defer sum.deinitAsync(&stream);

    var host = try device_tensor1.toHost(allocator, &stream);
    defer host.deinit(allocator);

    var broad_host = try broad.toHost(allocator, &stream);
    defer broad_host.deinit(allocator);

    var sum_host = try sum.toHost(allocator, &stream);
    defer sum_host.deinit(allocator);

    try stream.sync();

    std.debug.print("{d}", .{host});
    std.debug.print("{d}", .{broad_host});
    std.debug.print("{d}", .{sum_host});
}
