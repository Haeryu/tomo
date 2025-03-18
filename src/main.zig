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
    var a_values: [24]F = undefined;
    for (&a_values, 0..) |*val, idx| {
        val.* = @floatFromInt(idx + 1);
    }
    try tensor_a.writeFromHostAsync(&a_values, 0, &stream);

    // Fill tensor B with sequential values starting at 1.
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

    // ============================================================
    // SumTo Test: Sum over axis 1 to reduce shape from (2, 3, 4) to (2, 1, 4)
    // ============================================================
    // Create tensor A with shape (2, 3, 4) for sumTo test
    var tensor_a_sumto = try tm.tensor.GPUTensor(F).initAsync(a_shape, &stream);
    defer tensor_a_sumto.deinitAsync(&stream);

    // Fill with the same values as before
    try tensor_a_sumto.writeFromHostAsync(&a_values, 0, &stream);

    // Define target shape (2, 1, 4) to sum over axis 1
    const target_shape = &[_]usize{ 2, 1, 4 };

    // Perform sumTo operation
    var sumto_tensor = try tensor_a_sumto.sumTo(allocator, target_shape, &stream);
    defer sumto_tensor.deinitAsync(&stream);

    // Transfer sumTo result to the host
    var host_sumto = try sumto_tensor.toHost(allocator, &stream);
    defer host_sumto.deinit(allocator);

    // Compute expected output on host
    var expected_sumto: [2 * 1 * 4]F = undefined;
    for (0..2) |i| {
        for (0..4) |k| {
            var sum: F = 0;
            for (0..3) |j| {
                sum += a_values[i * 12 + j * 4 + k];
            }
            expected_sumto[i * 4 + k] = sum;
        }
    }

    // Compare the results
    for (0..8) |idx| {
        if (host_sumto.data[idx] != expected_sumto[idx]) {
            std.debug.print("SumTo mismatch at index {}: expected {}, got {}\n", .{ idx, expected_sumto[idx], host_sumto.data[idx] });
            return error.TestFailed;
        }
    }
    std.debug.print("SumTo test passed.\n", .{});

    // Synchronize the stream to ensure all operations are complete
    try stream.sync();
    std.debug.print("All tests complete.\n", .{});
}
