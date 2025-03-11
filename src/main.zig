const tm = @import("tomo_lib");
const std = @import("std");

pub fn main() !void {
    // Initialize the allocator
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    // Create a CUDA stream
    var stream = try tm.stream.Stream.create();
    defer stream.destroy();

    // Initialize CUDA context
    var cuda_context = try tm.cuda_context.CudaContext.init();
    defer cuda_context.deinit();

    // Define the floating-point type
    const F = f32;

    // Define the shape of the original tensor: 3 rows, 4 columns
    const rows = 3;
    const cols = 4;

    // Create a 3x4 GPUTensor and initialize it with values 1 to 12
    var device_tensor = try tm.tensor.GPUTensor(F).initAsync(&.{ rows, cols }, &stream);
    defer device_tensor.deinitAsync(&stream);

    // Write test values to the tensor
    const values = [_]F{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    try device_tensor.writeFromHostAsync(&values, 0, &stream);

    // Define slices to extract a subtensor
    // Rows 1 to 2 (0-based index), Columns 2 to 3 (0-based index)
    const slices = [_]tm.tensor.GPUTensor(F).Slice{
        .{ .start = 1, .stop = 3, .step = 1 }, // Rows: 1 to 2
        .{ .start = 2, .stop = 4, .step = 1 }, // Columns: 2 to 3
    };

    // Extract the subtensor using getItem
    var subtensor = try device_tensor.getItem(allocator, &slices, &stream);
    defer subtensor.deinitAsync(&stream);

    // Transfer the subtensor to the host for printing
    var host_subtensor = try subtensor.toHost(allocator, &stream);
    defer host_subtensor.deinit(allocator);

    // Create a gradient tensor (gy) with the same shape as the subtensor, filled with ones
    var gy = try tm.tensor.GPUTensor(F).initAsync(subtensor.base.getShapeConst(), &stream);
    defer gy.deinitAsync(&stream);
    try gy.fill(1.0, &stream);

    // Compute the gradient of the original tensor using getItemGrad
    var gx = try device_tensor.getItemGrad(allocator, &slices, &gy, &stream);
    defer gx.deinitAsync(&stream);

    // Transfer the gradient to the host for printing
    var host_gx = try gx.toHost(allocator, &stream);
    defer host_gx.deinit(allocator);

    // Synchronize the stream to ensure all operations are complete
    try stream.sync();

    // Print the results
    std.debug.print("Subtensor:\n{d}\n", .{host_subtensor});
    std.debug.print("Gradient gx:\n{d}\n", .{host_gx});
}
