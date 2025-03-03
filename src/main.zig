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

    const row1 = 3;
    const col1 = 4;

    const row2 = 12;
    const col2 = 12;

    var device_tensor1 = try tm.tensor.GPUTensor(F).initAsync(&.{ row1, col1 }, &stream);
    defer device_tensor1.deinitAsync(&stream);
    try device_tensor1.fillHeNormal(&cuda_context, &stream);

    var broad = try device_tensor1.broadCastTo(&.{ row2, col2 }, &stream);
    defer broad.deinitAsync(&stream);

    var sum = try device_tensor1.sum(tm.allocator.cuda_pinned_allocator, &.{1}, false, &stream);
    defer sum.deinitAsync(&stream);

    var host = try device_tensor1.toHost(tm.allocator.cuda_pinned_allocator, &stream);
    defer host.deinit(tm.allocator.cuda_pinned_allocator);

    var broad_host = try broad.toHost(tm.allocator.cuda_pinned_allocator, &stream);
    defer broad_host.deinit(tm.allocator.cuda_pinned_allocator);

    var sum_host = try sum.toHost(tm.allocator.cuda_pinned_allocator, &stream);
    defer sum_host.deinit(tm.allocator.cuda_pinned_allocator);

    try stream.sync();

    std.debug.print("{d}", .{host});
    std.debug.print("{d}", .{broad_host});
    std.debug.print("{d}", .{sum_host});
}
