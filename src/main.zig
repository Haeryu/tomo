const tm = @import("tomo_lib");

const std = @import("std");

pub fn main() !void {
    // Initialize the allocator.
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create a CUDA stream.
    var stream: tm.stream.Stream = try tm.stream.Stream.create();
    defer stream.destroy();

    // Initialize CUDA context.
    var cuda_context: tm.cuda_context.CudaContext = try tm.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    // Use f32 for this test.
    const F = f32;

    // ============================================================
    // Tensordot Test: Contract A's axis 2 with B's axis 0
    // ============================================================
    // Create tensor A with shape (2, 3, 4)
    const a_shape = &[_]usize{ 2, 3, 4 };
    var tensor_a = try tm.tensor.GPUTensor(F).initAsync(a_shape, &stream);
    defer tensor_a.deinitAsync(&stream);

    // Create tensor B with shape (4, 5, 6)
    const b_shape = &[_]usize{ 4, 5, 6 };
    var tensor_b = try tm.tensor.GPUTensor(F).initAsync(b_shape, &stream);
    defer tensor_b.deinitAsync(&stream);

    // Fill tensor A with sequential values starting at 1.
    // const numA = 2 * 3 * 4;
    var a_values: [24]F = undefined;
    for (&a_values, 0..) |*val, idx| {
        val.* = @floatFromInt(idx + 1);
    }
    try tensor_a.writeFromHostAsync(&a_values, 0, &stream);

    // Fill tensor B with sequential values starting at 1.
    //const numB = 4 * 5 * 6;
    var b_values: [120]F = undefined;
    for (&b_values, 0..) |*val, idx| {
        val.* = @floatFromInt(idx + 1);
    }
    try tensor_b.writeFromHostAsync(&b_values, 0, &stream);

    // Define contraction axes: contract A's axis 2 with B's axis 0.
    const axes_a = &[_]usize{2};
    const axes_b = &[_]usize{0};

    // tensordot: Expected output shape is (2, 3, 5, 6)
    var tensordot_tensor = try tensor_a.tensordot(&tensor_b, allocator, axes_a, axes_b, &stream);
    defer tensordot_tensor.deinitAsync(&stream);

    // Transfer tensordot result to the host for inspection.
    var host_tensordot = try tensordot_tensor.toHost(allocator, &stream);
    defer host_tensordot.deinit(allocator);
    std.debug.print("Tensordot output (A shape: {any}, B shape: {any}, contracted A axis2 with B axis0):\n{any}\n", .{ a_shape, b_shape, host_tensordot });

    try stream.sync();
    std.debug.print("Tensordot test complete.\n", .{});
}
