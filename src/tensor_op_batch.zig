const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const Bf16 = @import("bf16.zig").BF16;

const is_debugging = @import("builtin").mode == .Debug;

pub fn TensorOpBatch(comptime T: type) type {
    return struct {
        const Self = GPUTensor(T);

        pub fn getItem(self: *const Self, allocator: std.mem.Allocator, slices: []const Self.Slice, stream: *const Stream) !Self {
            if (slices.len != self.base.getShapeConst().len) return error.InvalidSlices;

            const out_shape = try Self.Slice.computeOutputShapeAlloc(allocator, self.base.getShapeConst(), slices);
            defer allocator.free(out_shape);

            var items = try Self.initAsync(out_shape, stream);
            errdefer items.deinitAsync(stream);

            const starts = try allocator.alloc(usize, self.base.getShapeConst().len);
            defer allocator.free(starts);

            const steps = try allocator.alloc(usize, self.base.getShapeConst().len);
            defer allocator.free(steps);

            for (slices, 0..) |s, d| {
                const start = s.start orelse 0;
                const dim = self.base.getShapeConst()[d];
                const adjusted_start = if (start < 0) start + @as(isize, @intCast(dim)) else start;
                starts[d] = @intCast(std.math.clamp(adjusted_start, @as(isize, 0), @as(isize, @intCast(dim))));
                steps[d] = @intCast(s.step orelse 1);
            }

            const out_size = items.calcLen();

            switch (T) {
                Bf16 => try err.checkCuda(c.tomoGetItemB(
                    @ptrCast(self.ptr.?),
                    @ptrCast(items.ptr.?),
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    out_shape.ptr,
                    out_shape.len,
                    self.base.getStrides().ptr,
                    self.base.getStrides().len,
                    items.base.getStrides().ptr,
                    items.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    out_shape.len,
                    out_size,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoGetItemH(
                    @ptrCast(self.ptr.?),
                    @ptrCast(items.ptr.?),
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    out_shape.ptr,
                    out_shape.len,
                    self.base.getStrides().ptr,
                    self.base.getStrides().len,
                    items.base.getStrides().ptr,
                    items.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    out_shape.len,
                    out_size,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoGetItemF(
                    self.ptr.?,
                    items.ptr.?,
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    out_shape.ptr,
                    out_shape.len,
                    self.base.getStrides().ptr,
                    self.base.getStrides().len,
                    items.base.getStrides().ptr,
                    items.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    out_shape.len,
                    out_size,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoGetItemD(
                    self.ptr.?,
                    items.ptr.?,
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    out_shape.ptr,
                    out_shape.len,
                    self.base.getStrides().ptr,
                    self.base.getStrides().len,
                    items.base.getStrides().ptr,
                    items.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    out_shape.len,
                    out_size,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try items.hasNaN(stream)) {
                return error.HasNan;
            }

            return items.move();
        }

        pub fn setItem(self: *Self, allocator: std.mem.Allocator, slices: []const Self.Slice, src: *const Self, stream: *const Stream) !void {
            // Validate that the number of slices matches the tensor's dimensions
            if (slices.len != self.base.getShapeConst().len) return error.InvalidSlices;
            if (src.base.getShapeConst().len != slices.len) return error.InvalidSourceShape;

            // Compute the expected shape of the source tensor based on the slices
            const expected_shape = try Self.Slice.computeOutputShapeAlloc(allocator, self.base.getShapeConst(), slices);
            defer allocator.free(expected_shape);

            // Validate that the source tensor's shape matches the expected shape
            for (expected_shape, 0..) |dim, i| {
                if (dim != src.base.getShapeConst()[i]) return error.ShapeMismatch;
            }

            // Allocate arrays for starts and steps
            const starts = try allocator.alloc(usize, self.base.getShapeConst().len);
            defer allocator.free(starts);

            const steps = try allocator.alloc(usize, self.base.getShapeConst().len);
            defer allocator.free(steps);

            // Compute starts and steps for each dimension
            for (slices, 0..) |s, d| {
                const start = s.start orelse 0;
                const dim = self.base.getShapeConst()[d];
                const adjusted_start = if (start < 0) start + @as(isize, @intCast(dim)) else start;
                starts[d] = @intCast(std.math.clamp(adjusted_start, @as(isize, 0), @as(isize, @intCast(dim))));
                steps[d] = @intCast(s.step orelse 1);
            }

            // Calculate the total number of elements in the source tensor
            const src_size = src.calcLen();

            // Call the appropriate CUDA function based on the tensor's data type
            switch (T) {
                Bf16 => try err.checkCuda(c.tomoSetItemB(
                    @ptrCast(src.ptr.?),
                    @ptrCast(self.ptr.?),
                    src.base.getShapeConst().ptr,
                    src.base.getShapeConst().len,
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    src.base.getStrides().ptr,
                    src.base.getStrides().len,
                    self.base.getStrides().ptr,
                    self.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    slices.len,
                    src_size,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoSetItemH(
                    @ptrCast(src.ptr.?),
                    @ptrCast(self.ptr.?),
                    src.base.getShapeConst().ptr,
                    src.base.getShapeConst().len,
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    src.base.getStrides().ptr,
                    src.base.getStrides().len,
                    self.base.getStrides().ptr,
                    self.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    slices.len,
                    src_size,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoSetItemF(
                    src.ptr.?,
                    self.ptr.?,
                    src.base.getShapeConst().ptr,
                    src.base.getShapeConst().len,
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    src.base.getStrides().ptr,
                    src.base.getStrides().len,
                    self.base.getStrides().ptr,
                    self.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    slices.len,
                    src_size,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoSetItemD(
                    src.ptr.?,
                    self.ptr.?,
                    src.base.getShapeConst().ptr,
                    src.base.getShapeConst().len,
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    src.base.getStrides().ptr,
                    src.base.getStrides().len,
                    self.base.getStrides().ptr,
                    self.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    slices.len,
                    src_size,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try self.hasNaN(stream)) {
                return error.HasNan;
            }
        }

        pub fn getItemGrad(self: *const Self, allocator: std.mem.Allocator, slices: []const Self.Slice, gy: *const Self, stream: *const Stream) !Self {
            if (slices.len != self.base.getShapeConst().len) return error.InvalidSlices;

            // Allocate gx with the same shape as self
            var gx = try Self.initAsync(self.base.getShapeConst(), stream);
            errdefer gx.deinitAsync(stream);

            // Zero initialize gx
            try err.checkCuda(c.cudaMemsetAsync(gx.ptr.?, 0, gx.calcLen() * @sizeOf(T), stream.stream));

            // Compute starts and steps (same as getItem)
            const starts = try allocator.alloc(usize, self.base.getShapeConst().len);
            defer allocator.free(starts);

            const steps = try allocator.alloc(usize, self.base.getShapeConst().len);
            defer allocator.free(steps);

            for (slices, 0..) |s, d| {
                const start = s.start orelse 0;
                const dim = self.base.getShapeConst()[d];
                const adjusted_start = if (start < 0) start + @as(isize, @intCast(dim)) else start;
                starts[d] = @intCast(std.math.clamp(adjusted_start, @as(isize, 0), @as(isize, @intCast(dim))));
                steps[d] = @intCast(s.step orelse 1);
            }

            const out_size = gy.calcLen();

            // Launch gradient kernel
            switch (T) {
                Bf16 => try err.checkCuda(c.tomoGetItemGradB(
                    @ptrCast(gy.ptr.?),
                    @ptrCast(gx.ptr.?),
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    gy.base.getShapeConst().ptr,
                    gy.base.getShapeConst().len,
                    gx.base.getStrides().ptr,
                    gx.base.getStrides().len,
                    gy.base.getStrides().ptr,
                    gy.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    gy.base.getShapeConst().len,
                    out_size,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoGetItemGradH(
                    @ptrCast(gy.ptr.?),
                    @ptrCast(gx.ptr.?),
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    gy.base.getShapeConst().ptr,
                    gy.base.getShapeConst().len,
                    gx.base.getStrides().ptr,
                    gx.base.getStrides().len,
                    gy.base.getStrides().ptr,
                    gy.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    gy.base.getShapeConst().len,
                    out_size,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoGetItemGradF(
                    gy.ptr.?,
                    gx.ptr.?,
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    gy.base.getShapeConst().ptr,
                    gy.base.getShapeConst().len,
                    gx.base.getStrides().ptr,
                    gx.base.getStrides().len,
                    gy.base.getStrides().ptr,
                    gy.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    gy.base.getShapeConst().len,
                    out_size,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoGetItemGradD(
                    gy.ptr.?,
                    gx.ptr.?,
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    gy.base.getShapeConst().ptr,
                    gy.base.getShapeConst().len,
                    gx.base.getStrides().ptr,
                    gx.base.getStrides().len,
                    gy.base.getStrides().ptr,
                    gy.base.getStrides().len,
                    starts.ptr,
                    starts.len,
                    steps.ptr,
                    steps.len,
                    gy.base.getShapeConst().len,
                    out_size,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try gx.hasNaN(stream)) {
                return error.HasNan;
            }

            return gx.move();
        }

        pub fn toOneHot(indices: *const GPUTensor(usize), comptime F: type, num_classes: usize, stream: *const Stream) !GPUTensor(F) {
            if (indices.base.getShapeConst().len != 1) return error.InvalidIndicesRank;

            const batch_size = indices.base.getShapeConst()[0];

            var one_hot = try GPUTensor(F).initAsync(&[_]usize{ batch_size, num_classes }, stream);
            errdefer one_hot.deinitAsync(stream);

            switch (F) {
                Bf16 => try err.checkCuda(c.tomoOneHotB(indices.ptr.?, @ptrCast(one_hot.ptr.?), batch_size, num_classes, stream.stream)),
                f16 => try err.checkCuda(c.tomoOneHotH(indices.ptr.?, @ptrCast(one_hot.ptr.?), batch_size, num_classes, stream.stream)),
                f32 => try err.checkCuda(c.tomoOneHotF(indices.ptr.?, one_hot.ptr.?, batch_size, num_classes, stream.stream)),
                f64 => try err.checkCuda(c.tomoOneHotD(indices.ptr.?, one_hot.ptr.?, batch_size, num_classes, stream.stream)),
                else => unreachable,
            }

            return one_hot.move();
        }

        fn computeTensordotShape(allocator: std.mem.Allocator, a_shape: []const usize, b_shape: []const usize, axes_a: []const usize, axes_b: []const usize) ![]usize {
            var out_shape = std.ArrayList(usize).init(allocator);

            // Add non-contracted axes from a
            for (a_shape, 0..) |dim, i| {
                if (!std.mem.containsAtLeast(usize, axes_a, 1, &.{i})) {
                    try out_shape.append(dim);
                }
            }

            // Add non-contracted axes from b
            for (b_shape, 0..) |dim, i| {
                if (!std.mem.containsAtLeast(usize, axes_b, 1, &.{i})) {
                    try out_shape.append(dim);
                }
            }

            return out_shape.toOwnedSlice();
        }

        pub fn tensordot(
            self: *const Self,
            other: *const Self,
            allocator: std.mem.Allocator,
            axes_a: []const usize,
            axes_b: []const usize,
            stream: *const Stream,
        ) !Self {
            // Validate contraction dimensions
            if (axes_a.len != axes_b.len) return error.ContractionRankMismatch;
            for (axes_a, axes_b) |a_dim, b_dim| {
                if (self.base.getShapeConst()[a_dim] != other.base.getShapeConst()[b_dim])
                    return error.ContractionDimensionMismatch;
            }

            // Compute output shape
            const out_shape = try computeTensordotShape(allocator, self.base.getShapeConst(), other.base.getShapeConst(), axes_a, axes_b);
            defer allocator.free(out_shape);

            // Create output tensor
            var out_tensor = try Self.initAsync(out_shape, stream);
            errdefer out_tensor.deinitAsync(stream);

            // Get stride information
            const a_strides = self.base.getStrides();
            const b_strides = other.base.getStrides();

            switch (T) {
                Bf16 => try err.checkCuda(c.tomoTensordotB(
                    @ptrCast(self.ptr.?),
                    @ptrCast(other.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    other.base.getShapeConst().ptr,
                    other.base.getShapeConst().len,
                    out_shape.ptr,
                    out_shape.len,
                    a_strides.ptr,
                    a_strides.len,
                    b_strides.ptr,
                    b_strides.len,
                    out_tensor.base.getStrides().ptr,
                    out_tensor.base.getStrides().len,
                    axes_a.ptr,
                    axes_a.len,
                    axes_b.ptr,
                    axes_b.len,
                    self.base.rank,
                    other.base.rank,
                    out_shape.len,
                    out_tensor.calcLen(),
                    axes_a.len,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoTensordotH(
                    @ptrCast(self.ptr.?),
                    @ptrCast(other.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    other.base.getShapeConst().ptr,
                    other.base.getShapeConst().len,
                    out_shape.ptr,
                    out_shape.len,
                    a_strides.ptr,
                    a_strides.len,
                    b_strides.ptr,
                    b_strides.len,
                    out_tensor.base.getStrides().ptr,
                    out_tensor.base.getStrides().len,
                    axes_a.ptr,
                    axes_a.len,
                    axes_b.ptr,
                    axes_b.len,
                    self.base.rank,
                    other.base.rank,
                    out_shape.len,
                    out_tensor.calcLen(),
                    axes_a.len,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoTensordotF(
                    self.ptr.?,
                    other.ptr.?,
                    out_tensor.ptr.?,
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    other.base.getShapeConst().ptr,
                    other.base.getShapeConst().len,
                    out_shape.ptr,
                    out_shape.len,
                    a_strides.ptr,
                    a_strides.len,
                    b_strides.ptr,
                    b_strides.len,
                    out_tensor.base.getStrides().ptr,
                    out_tensor.base.getStrides().len,
                    axes_a.ptr,
                    axes_a.len,
                    axes_b.ptr,
                    axes_b.len,
                    self.base.rank,
                    other.base.rank,
                    out_shape.len,
                    out_tensor.calcLen(),
                    axes_a.len,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoTensordotD(
                    self.ptr.?,
                    other.ptr.?,
                    out_tensor.ptr.?,
                    self.base.getShapeConst().ptr,
                    self.base.getShapeConst().len,
                    other.base.getShapeConst().ptr,
                    other.base.getShapeConst().len,
                    out_shape.ptr,
                    out_shape.len,
                    a_strides.ptr,
                    a_strides.len,
                    b_strides.ptr,
                    b_strides.len,
                    out_tensor.base.getStrides().ptr,
                    out_tensor.base.getStrides().len,
                    axes_a.ptr,
                    axes_a.len,
                    axes_b.ptr,
                    axes_b.len,
                    self.base.rank,
                    other.base.rank,
                    out_shape.len,
                    out_tensor.calcLen(),
                    axes_a.len,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try out_tensor.hasNaN(stream)) {
                return error.HasNan;
            }

            return out_tensor.move();
        }

        pub fn tensordotImp(
            self: *const Self,
            other: *const Self,
            allocator: std.mem.Allocator,
            axes_a: []const usize,
            axes_b: []const usize,
            stream: *const Stream,
        ) !Self {
            const a_shape = self.base.getShapeConst();
            const b_shape = other.base.getShapeConst();

            // Validate contraction dimensions
            if (axes_a.len != axes_b.len) return error.ContractionRankMismatch;
            for (axes_a, axes_b) |a_dim, b_dim| {
                if (a_shape[a_dim] != b_shape[b_dim]) return error.ContractionDimensionMismatch;
            }

            // Check for simple matrix multiplication case
            if (a_shape.len == 2 and b_shape.len == 2 and
                axes_a.len == 1 and axes_b.len == 1 and
                axes_a[0] == 1 and axes_b[0] == 0)
            {
                var result = try self.linearImp(other, null, stream);
                errdefer result.deinitAsync(stream);

                return result.move();
            }

            // General case: reshape and transpose to use linearImp

            // Step 1: Identify non-contracted axes
            var a_non_contracted = std.ArrayList(usize).init(allocator);
            defer a_non_contracted.deinit();
            for (0..a_shape.len) |i| {
                if (!std.mem.containsAtLeast(usize, axes_a, 1, &.{i})) {
                    try a_non_contracted.append(i);
                }
            }

            var b_non_contracted = std.ArrayList(usize).init(allocator);
            defer b_non_contracted.deinit();
            for (0..b_shape.len) |i| {
                if (!std.mem.containsAtLeast(usize, axes_b, 1, &.{i})) {
                    try b_non_contracted.append(i);
                }
            }

            // Step 2: Compute sizes for reshaping
            var a_non_contracted_size: usize = 1;
            for (a_non_contracted.items) |i| a_non_contracted_size *= a_shape[i];
            var contracted_size: usize = 1;
            for (axes_a) |i| contracted_size *= a_shape[i];
            var b_non_contracted_size: usize = 1;
            for (b_non_contracted.items) |i| b_non_contracted_size *= b_shape[i];

            // Step 3: Transpose and make contiguous, then reshape for a
            var a_perm = try allocator.alloc(usize, a_shape.len);
            defer allocator.free(a_perm);
            for (a_non_contracted.items, 0..) |ax, i| a_perm[i] = ax;
            for (axes_a, 0..) |ax, i| a_perm[a_non_contracted.items.len + i] = ax;
            var a_transposed = try self.transposeEx(allocator, a_perm, stream);
            defer a_transposed.deinitAsync(stream);

            // Reshape a_transposed to [a_non_contracted_size, contracted_size]
            var a_reshaped_shape = try allocator.alloc(usize, 2);
            defer allocator.free(a_reshaped_shape);

            a_reshaped_shape[0] = a_non_contracted_size;
            a_reshaped_shape[1] = contracted_size;
            var a_reshaped = &a_transposed;
            try a_reshaped.reshape(a_reshaped_shape);

            // Step 4: Transpose and make contiguous, then reshape for b
            var b_perm = try allocator.alloc(usize, b_shape.len);
            defer allocator.free(b_perm);
            // Align b's contraction axes order with a's
            for (axes_b, 0..) |ax, i| b_perm[i] = ax;
            for (b_non_contracted.items, 0..) |ax, i| b_perm[axes_b.len + i] = ax;
            var b_transposed = try other.transposeEx(allocator, b_perm, stream);
            defer b_transposed.deinitAsync(stream);

            // Reshape b_transposed to [contracted_size, b_non_contracted_size]
            var b_reshaped_shape = try allocator.alloc(usize, 2);
            defer allocator.free(b_reshaped_shape);
            b_reshaped_shape[0] = contracted_size;
            b_reshaped_shape[1] = b_non_contracted_size;
            var b_reshaped = b_transposed;
            try b_reshaped.reshape(b_reshaped_shape);

            // Step 5: Perform matrix multiplication
            var matmul_result = try a_reshaped.linearImp(&b_reshaped, null, stream);
            errdefer matmul_result.deinitAsync(stream);
            // No defer here; ownership transferred to result

            // Step 6: Reshape result
            var out_shape = try allocator.alloc(usize, a_non_contracted.items.len + b_non_contracted.items.len);
            defer allocator.free(out_shape);
            for (a_non_contracted.items, 0..) |ax, i| out_shape[i] = a_shape[ax];
            for (b_non_contracted.items, 0..) |ax, i| out_shape[a_non_contracted.items.len + i] = b_shape[ax];
            var result = &matmul_result;
            try result.reshape(out_shape);

            if (is_debugging and try result.hasNaN(stream)) {
                return error.HasNan;
            }

            return result.move();
        }
        pub fn transposeEx(
            self: *const Self,
            allocator: std.mem.Allocator,
            perm: []const usize,
            stream: *const Stream,
        ) !Self {
            const in_shape = self.base.getShapeConst();
            const in_strides = self.base.getStrides();

            // Compute output shape and strides
            const out_shape = try allocator.alloc(usize, in_shape.len);
            defer allocator.free(out_shape);

            const out_strides = try allocator.alloc(usize, in_shape.len);
            defer allocator.free(out_strides);

            for (perm, 0..) |p, i| {
                out_shape[i] = in_shape[p];
                out_strides[i] = in_strides[p];
            }

            var out_tensor: Self = try .initAsync(out_shape, stream);
            errdefer out_tensor.deinitAsync(stream);

            switch (T) {
                Bf16 => try err.checkCuda(c.tomoTransposeExB(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    in_shape.ptr,
                    in_shape.len,
                    out_shape.ptr,
                    out_shape.len,
                    in_strides.ptr,
                    in_strides.len,
                    out_strides.ptr,
                    out_strides.len,
                    perm.ptr,
                    perm.len,
                    in_shape.len,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoTransposeExH(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    in_shape.ptr,
                    in_shape.len,
                    out_shape.ptr,
                    out_shape.len,
                    in_strides.ptr,
                    in_strides.len,
                    out_strides.ptr,
                    out_strides.len,
                    perm.ptr,
                    perm.len,
                    in_shape.len,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoTransposeExF(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    in_shape.ptr,
                    in_shape.len,
                    out_shape.ptr,
                    out_shape.len,
                    in_strides.ptr,
                    in_strides.len,
                    out_strides.ptr,
                    out_strides.len,
                    perm.ptr,
                    perm.len,
                    in_shape.len,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoTransposeExD(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    in_shape.ptr,
                    in_shape.len,
                    out_shape.ptr,
                    out_shape.len,
                    in_strides.ptr,
                    in_strides.len,
                    out_strides.ptr,
                    out_strides.len,
                    perm.ptr,
                    perm.len,
                    in_shape.len,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try out_tensor.hasNaN(stream)) {
                return error.HasNan;
            }

            return out_tensor.move();
        }

        fn computeRollaxisPerm(nd: usize, axis: usize, start: usize, perm: []usize) void {
            var temp_perm: [GPUTensor(T).max_rank]usize = undefined;
            var idx: usize = 0;
            for (0..nd) |i| {
                if (i != axis) {
                    temp_perm[idx] = i;
                    idx += 1;
                }
            }
            const adjusted_start = if (start > idx) idx else start;
            for (0..adjusted_start) |i| {
                perm[i] = temp_perm[i];
            }
            perm[adjusted_start] = axis;
            for (adjusted_start..nd - 1) |i| {
                perm[i + 1] = temp_perm[i];
            }
        }

        pub fn rollaxis(
            self: *const Self,
            axis: usize,
            start: usize,
            stream: *const Stream,
        ) !Self {
            const in_shape = self.base.getShapeConst();
            const nd = in_shape.len;

            var perm: [GPUTensor(T).max_rank]usize = undefined;
            computeRollaxisPerm(nd, axis, start, perm[0..nd]);

            // Compute output shape
            var out_shape: [GPUTensor(T).max_rank]usize = undefined;
            for (0..nd) |i| {
                out_shape[i] = in_shape[perm[i]];
            }

            var out_tensor = try Self.initAsync(out_shape[0..nd], stream);
            errdefer out_tensor.deinitAsync(stream);

            switch (T) {
                Bf16 => try err.checkCuda(c.tomoRollaxisB(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    in_shape.ptr,
                    nd,
                    self.base.getStrides().ptr,
                    nd,
                    axis,
                    start,
                    nd,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoRollaxisH(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    in_shape.ptr,
                    nd,
                    self.base.getStrides().ptr,
                    nd,
                    axis,
                    start,
                    nd,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoRollaxisF(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    in_shape.ptr,
                    nd,
                    self.base.getStrides().ptr,
                    nd,
                    axis,
                    start,
                    nd,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoRollaxisD(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    in_shape.ptr,
                    nd,
                    self.base.getStrides().ptr,
                    nd,
                    axis,
                    start,
                    nd,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try out_tensor.hasNaN(stream)) {
                return error.HasNan;
            }

            return out_tensor.move();
        }

        pub fn swapaxes(
            self: *const Self,
            axis1: usize,
            axis2: usize,
            stream: *const Stream,
        ) !Self {
            const in_shape = self.base.getShapeConst();
            const nd = in_shape.len;

            var out_shape = try std.heap.page_allocator.dupe(usize, in_shape);
            defer std.heap.page_allocator.free(out_shape);

            // Swap axes in output shape
            const tmp = out_shape[axis1];
            out_shape[axis1] = out_shape[axis2];
            out_shape[axis2] = tmp;

            var out_tensor = try Self.initAsync(out_shape, stream);
            errdefer out_tensor.deinitAsync(stream);

            switch (T) {
                Bf16 => try err.checkCuda(c.tomoSwapaxesB(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    in_shape.ptr,
                    nd,
                    out_shape.ptr,
                    nd,
                    self.base.getStrides().ptr,
                    nd,
                    out_tensor.base.getStrides().ptr,
                    nd,
                    axis1,
                    axis2,
                    nd,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoSwapaxesH(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    in_shape.ptr,
                    nd,
                    out_shape.ptr,
                    nd,
                    self.base.getStrides().ptr,
                    nd,
                    out_tensor.base.getStrides().ptr,
                    nd,
                    axis1,
                    axis2,
                    nd,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoSwapaxesF(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    in_shape.ptr,
                    nd,
                    out_shape.ptr,
                    nd,
                    self.base.getStrides().ptr,
                    nd,
                    out_tensor.base.getStrides().ptr,
                    nd,
                    axis1,
                    axis2,
                    nd,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoSwapaxesD(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    in_shape.ptr,
                    nd,
                    out_shape.ptr,
                    nd,
                    self.base.getStrides().ptr,
                    nd,
                    out_tensor.base.getStrides().ptr,
                    nd,
                    axis1,
                    axis2,
                    nd,
                    self.calcLen(),
                    out_tensor.calcLen(),
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try out_tensor.hasNaN(stream)) {
                return error.HasNan;
            }

            return out_tensor.move();
        }

        pub fn im2col(
            self: *const Self,
            kernel: [2]usize,
            stride: [2]usize,
            padding: [2]usize,
            dilation: [2]usize,
            stream: *const Stream,
        ) !Self {
            const in_shape = self.base.getShapeConst();
            if (in_shape.len != 4) return error.InvalidInputShape;

            const n = in_shape[0];
            const c_ = in_shape[1];
            const h = in_shape[2];
            const w = in_shape[3];
            const kh, const kw = kernel;
            const sy, const sx = stride;
            const ph, const pw = padding;
            const dy, const dx = dilation;

            const out_h = (h + 2 * ph - dy * (kh - 1) - 1) / sy + 1;
            const out_w = (w + 2 * pw - dx * (kw - 1) - 1) / sx + 1;
            const out_shape = &.{ n, c_ * kh * kw, out_h, out_w };

            var out_tensor = try Self.initAsync(out_shape, stream);
            errdefer out_tensor.deinitAsync(stream);

            switch (T) {
                Bf16 => try err.checkCuda(c.tomoIm2colB(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    n,
                    c_,
                    h,
                    w,
                    kh,
                    kw,
                    sy,
                    sx,
                    ph,
                    pw,
                    dy,
                    dx,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoIm2colH(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    n,
                    c_,
                    h,
                    w,
                    kh,
                    kw,
                    sy,
                    sx,
                    ph,
                    pw,
                    dy,
                    dx,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoIm2colF(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    n,
                    c_,
                    h,
                    w,
                    kh,
                    kw,
                    sy,
                    sx,
                    ph,
                    pw,
                    dy,
                    dx,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoIm2colD(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    n,
                    c_,
                    h,
                    w,
                    kh,
                    kw,
                    sy,
                    sx,
                    ph,
                    pw,
                    dy,
                    dx,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try out_tensor.hasNaN(stream)) {
                return error.HasNan;
            }

            return out_tensor.move();
        }

        pub fn col2im(
            self: *const Self,
            output_shape: []const usize,
            kernel: [2]usize,
            stride: [2]usize,
            padding: [2]usize,
            dilation: [2]usize,
            stream: *const Stream,
        ) !Self {
            if (output_shape.len != 4) return error.InvalidOutputShape;
            const n = output_shape[0];
            const c_ = output_shape[1];
            const h = output_shape[2];
            const w = output_shape[3];
            const kh, const kw = kernel;
            const sy, const sx = stride;
            const ph, const pw = padding;
            const dx, const dy = dilation;

            var out_tensor = try Self.initAsync(output_shape, stream);
            errdefer out_tensor.deinitAsync(stream);

            switch (T) {
                Bf16 => try err.checkCuda(c.tomoCol2imB(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    n,
                    c_,
                    h,
                    w,
                    kh,
                    kw,
                    sy,
                    sx,
                    ph,
                    pw,
                    dx,
                    dy,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoCol2imH(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    n,
                    c_,
                    h,
                    w,
                    kh,
                    kw,
                    sy,
                    sx,
                    ph,
                    pw,
                    dx,
                    dy,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoCol2imF(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    n,
                    c_,
                    h,
                    w,
                    kh,
                    kw,
                    sy,
                    sx,
                    ph,
                    pw,
                    dx,
                    dy,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoCol2imD(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    n,
                    c_,
                    h,
                    w,
                    kh,
                    kw,
                    sy,
                    sx,
                    ph,
                    pw,
                    dx,
                    dy,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try out_tensor.hasNaN(stream)) {
                return error.HasNan;
            }

            return out_tensor.move();
        }

        pub fn im2col1d(
            self: *const Self,
            kernel_size: usize,
            stride: usize,
            padding: usize,
            dilation: usize,
            stream: *const Stream,
        ) !Self {
            const in_shape = self.base.getShapeConst();
            if (in_shape.len != 3) {
                return error.InvalidInputShape; // We expect [N, C, L]
            }

            const n = in_shape[0];
            const c_ = in_shape[1];
            const l = in_shape[2];

            // Compute output length
            //   out_l = floor((L + 2*pad - (kernel_size-1)*dilation - 1) / stride) + 1
            const out_l = (l + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

            // The output shape for im2col1d is [N, C*k, out_l]
            const out_shape = &[_]usize{ n, c_ * kernel_size, out_l };

            var out_tensor = try Self.initAsync(out_shape, stream);
            errdefer out_tensor.deinitAsync(stream);

            // Switch on T to call the correct tomoIm2col1dX function
            switch (T) {
                Bf16 => try err.checkCuda(c.tomoIm2col1dB(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    n,
                    c_,
                    l,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoIm2col1dH(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    n,
                    c_,
                    l,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoIm2col1dF(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    n,
                    c_,
                    l,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoIm2col1dD(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    n,
                    c_,
                    l,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try out_tensor.hasNaN(stream)) {
                return error.HasNan;
            }

            return out_tensor.move();
        }

        pub fn col2im1d(
            self: *const Self,
            output_shape: []const usize,
            kernel_size: usize,
            stride: usize,
            padding: usize,
            dilation: usize,
            stream: *const Stream,
        ) !Self {
            // We expect output_shape to be [N, C, L]
            if (output_shape.len != 3) {
                return error.InvalidOutputShape;
            }

            const n = output_shape[0];
            const c_ = output_shape[1];
            const l = output_shape[2];

            var out_tensor = try Self.initAsync(output_shape, stream);
            errdefer out_tensor.deinitAsync(stream);

            // Switch on T to call the correct tomoCol2im1dX function
            switch (T) {
                Bf16 => try err.checkCuda(c.tomoCol2im1dB(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    n,
                    c_,
                    l,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoCol2im1dH(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    n,
                    c_,
                    l,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoCol2im1dF(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    n,
                    c_,
                    l,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoCol2im1dD(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    n,
                    c_,
                    l,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try out_tensor.hasNaN(stream)) {
                return error.HasNan;
            }

            return out_tensor.move();
        }

        pub fn maxPool2dForward(
            self: *const Self,
            kernel_size: [2]usize,
            stride: [2]usize,
            padding: [2]usize,
            stream: *const Stream,
        ) !Self {
            // Validate input shape = [N, C, H, W]
            const shape_in = self.base.getShapeConst();
            if (shape_in.len != 4) return error.InvalidInputShape;

            const N = shape_in[0];
            const C = shape_in[1];
            const H = shape_in[2];
            const W = shape_in[3];

            const kH, const kW = kernel_size;
            const sH, const sW = stride;
            const pH, const pW = padding;

            // compute outH, outW for standard forward pooling

            const outH = (H + 2 * pH - kH) / sH + 1;
            const outW = (W + 2 * pW - kW) / sW + 1;

            // Allocate output: [N, C, outH, outW]
            const out_shape = &[_]usize{ N, C, outH, outW };
            var out_tensor = try Self.initAsync(out_shape, stream);
            errdefer out_tensor.deinitAsync(stream);

            // Switch on T to call the correct function
            switch (T) {
                Bf16 => try err.checkCuda(c.tomoMaxPool2dForwardB(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoMaxPool2dForwardH(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoMaxPool2dForwardF(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoMaxPool2dForwardD(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try out_tensor.hasNaN(stream)) {
                return error.HasNan;
            }

            return out_tensor.move();
        }

        pub fn maxPool2dBackward(
            x: *const Self, // forward input, shape [N,C,H,W]
            gy: *const Self, // upstream gradient, shape [N,C,outH,outW]
            kernel_size: [2]usize,
            stride: [2]usize,
            padding: [2]usize,
            stream: *const Stream,
        ) !Self {
            const shape_in = x.base.getShapeConst();
            if (shape_in.len != 4) return error.InvalidInputShape;
            const shape_gy = gy.base.getShapeConst();
            if (shape_gy.len != 4) return error.InvalidInputShape;

            const N = shape_in[0];
            const C = shape_in[1];
            const H = shape_in[2];
            const W = shape_in[3];

            const outH = shape_gy[2];
            const outW = shape_gy[3];

            const kH, const kW = kernel_size;
            const sH, const sW = stride;
            const pH, const pW = padding;

            // Allocate gx: same shape as x
            var gx = try Self.initAsync(shape_in, stream);
            errdefer gx.deinitAsync(stream);

            // Zero-initialize gx
            try err.checkCuda(c.cudaMemsetAsync(gx.ptr.?, 0, gx.calcLen() * @sizeOf(T), stream.stream));

            // Switch on T, call the correct tomoMaxPool2dBackwardX
            switch (T) {
                Bf16 => try err.checkCuda(c.tomoMaxPool2dBackwardB(
                    @ptrCast(x.ptr.?),
                    @ptrCast(gy.ptr.?),
                    @ptrCast(gx.ptr.?),
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoMaxPool2dBackwardH(
                    @ptrCast(x.ptr.?),
                    @ptrCast(gy.ptr.?),
                    @ptrCast(gx.ptr.?),
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoMaxPool2dBackwardF(
                    x.ptr.?,
                    gy.ptr.?,
                    gx.ptr.?,
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoMaxPool2dBackwardD(
                    x.ptr.?,
                    gy.ptr.?,
                    gx.ptr.?,
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try gx.hasNaN(stream)) {
                return error.HasNan;
            }

            return gx.move();
        }

        pub fn avgPool2dForward(
            self: *const Self,
            kernel_size: [2]usize,
            stride: [2]usize,
            padding: [2]usize,
            stream: *const Stream,
        ) !Self {
            // shape_in = [N, C, H, W]
            const shape_in = self.base.getShapeConst();
            if (shape_in.len != 4) return error.InvalidInputShape;

            const N = shape_in[0];
            const C = shape_in[1];
            const H = shape_in[2];
            const W = shape_in[3];

            const kH, const kW = kernel_size;
            const sH, const sW = stride;
            const pH, const pW = padding;

            // standard formula
            const outH = (H + 2 * pH - kH) / sH + 1;
            const outW = (W + 2 * pW - kW) / sW + 1;

            const out_shape = &[_]usize{ N, C, outH, outW };
            var out_tensor = try Self.initAsync(out_shape, stream);
            errdefer out_tensor.deinitAsync(stream);

            switch (T) {
                Bf16 => try err.checkCuda(c.tomoAvgPool2dForwardB(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoAvgPool2dForwardH(
                    @ptrCast(self.ptr.?),
                    @ptrCast(out_tensor.ptr.?),
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoAvgPool2dForwardF(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoAvgPool2dForwardD(
                    self.ptr.?,
                    out_tensor.ptr.?,
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try out_tensor.hasNaN(stream)) {
                return error.HasNan;
            }

            return out_tensor.move();
        }

        pub fn avgPool2dBackward(
            gy: *const Self, // [N, C, outH, outW]
            shape_in: []const usize, // original [N, C, H, W] shape
            kernel_size: [2]usize,
            stride: [2]usize,
            padding: [2]usize,
            stream: *const Stream,
        ) !Self {
            // shape_in = [N,C,H,W]
            if (shape_in.len != 4) return error.InvalidInputShape;

            const N = shape_in[0];
            const C = shape_in[1];
            const H = shape_in[2];
            const W = shape_in[3];

            const shape_gy = gy.base.getShapeConst();
            if (shape_gy.len != 4) return error.InvalidInputShape;
            const outH = shape_gy[2];
            const outW = shape_gy[3];

            const kH, const kW = kernel_size;
            const sH, const sW = stride;
            const pH, const pW = padding;

            // Allocate gX => same shape as input
            var gx = try Self.initAsync(shape_in, stream);
            errdefer gx.deinitAsync(stream);

            // zero gX
            try err.checkCuda(c.cudaMemsetAsync(gx.ptr.?, 0, gx.calcLen() * @sizeOf(T), stream.stream));

            switch (T) {
                Bf16 => try err.checkCuda(c.tomoAvgPool2dBackwardB(
                    @ptrCast(gy.ptr.?),
                    @ptrCast(gx.ptr.?),
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoAvgPool2dBackwardH(
                    @ptrCast(gy.ptr.?),
                    @ptrCast(gx.ptr.?),
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoAvgPool2dBackwardF(
                    gy.ptr.?,
                    gx.ptr.?,
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoAvgPool2dBackwardD(
                    N,
                    C,
                    H,
                    W,
                    outH,
                    outW,
                    kH,
                    kW,
                    sH,
                    sW,
                    pH,
                    pW,
                    stream.stream,
                )),
                else => unreachable,
            }

            if (is_debugging and try gx.hasNaN(stream)) {
                return error.HasNan;
            }

            return gx.move();
        }

        pub fn embeddingForward(
            weight: *const GPUTensor(T),
            indices: *const GPUTensor(usize),
            stream: *const Stream,
        ) !GPUTensor(T) {
            // Extract shapes from input tensors
            const num_embeddings = weight.base.getShapeConst()[0];
            const embedding_dim = weight.base.getShapeConst()[1];
            const batch_size = indices.base.getShapeConst()[0];
            const sequence_length = indices.base.getShapeConst()[1];

            // Create output tensor with shape [batch_size, sequence_length, embedding_dim]
            var output = try GPUTensor(T).initAsync(&[_]usize{ batch_size, sequence_length, embedding_dim }, stream);
            errdefer output.deinitAsync(stream);

            // Call the appropriate C function based on the data type T
            switch (T) {
                Bf16 => try err.checkCuda(c.tomoEmbeddingForwardB(
                    @ptrCast(weight.ptr.?), // Cast to raw Bf16 pointer
                    indices.ptr.?, // Direct usize pointer
                    @ptrCast(output.ptr.?), // Cast to raw Bf16 pointer
                    num_embeddings,
                    embedding_dim,
                    batch_size,
                    sequence_length,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoEmbeddingForwardH(
                    @ptrCast(weight.ptr.?), // Cast to raw f16 pointer
                    indices.ptr.?, // Direct usize pointer
                    @ptrCast(output.ptr.?), // Cast to raw f16 pointer
                    num_embeddings,
                    embedding_dim,
                    batch_size,
                    sequence_length,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoEmbeddingForwardF(
                    weight.ptr.?, // Direct f32 pointer
                    indices.ptr.?, // Direct usize pointer
                    output.ptr.?, // Direct f32 pointer
                    num_embeddings,
                    embedding_dim,
                    batch_size,
                    sequence_length,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoEmbeddingForwardD(
                    weight.ptr.?, // Direct f64 pointer
                    indices.ptr.?, // Direct usize pointer
                    output.ptr.?, // Direct f64 pointer
                    num_embeddings,
                    embedding_dim,
                    batch_size,
                    sequence_length,
                    stream.stream,
                )),
                else => unreachable, // Only Bf16, f16, f32, f64 are supported
            }

            if (is_debugging and try output.hasNaN(stream)) {
                return error.HasNan;
            }

            // Return the output tensor, transferring ownership
            return output.move();
        }

        pub fn embeddingBackward(
            grad_output: *const GPUTensor(T),
            indices: *const GPUTensor(usize),
            num_embeddings: usize,
            stream: *const Stream,
        ) !GPUTensor(T) {
            // Extract shapes from input tensors
            const batch_size = grad_output.base.getShapeConst()[0];
            const sequence_length = grad_output.base.getShapeConst()[1];
            const embedding_dim = grad_output.base.getShapeConst()[2];

            var grad_weight = try GPUTensor(T).initAsync(&[_]usize{ num_embeddings, embedding_dim }, stream);
            errdefer grad_weight.deinitAsync(stream);
            try grad_weight.fill(0.0, stream);

            // Call the appropriate C function based on the data type T
            switch (T) {
                Bf16 => try err.checkCuda(c.tomoEmbeddingBackwardB(
                    @ptrCast(grad_output.ptr.?), // Cast to raw Bf16 pointer
                    indices.ptr.?, // Direct usize pointer
                    @ptrCast(grad_weight.ptr.?), // Cast to raw Bf16 pointer
                    num_embeddings,
                    embedding_dim,
                    batch_size,
                    sequence_length,
                    stream.stream,
                )),
                f16 => try err.checkCuda(c.tomoEmbeddingBackwardH(
                    @ptrCast(grad_output.ptr.?), // Cast to raw f16 pointer
                    indices.ptr.?, // Direct usize pointer
                    @ptrCast(grad_weight.ptr.?), // Cast to raw f16 pointer
                    num_embeddings,
                    embedding_dim,
                    batch_size,
                    sequence_length,
                    stream.stream,
                )),
                f32 => try err.checkCuda(c.tomoEmbeddingBackwardF(
                    grad_output.ptr.?, // Direct f32 pointer
                    indices.ptr.?, // Direct usize pointer
                    grad_weight.ptr.?, // Direct f32 pointer
                    num_embeddings,
                    embedding_dim,
                    batch_size,
                    sequence_length,
                    stream.stream,
                )),
                f64 => try err.checkCuda(c.tomoEmbeddingBackwardD(
                    grad_output.ptr.?, // Direct f64 pointer
                    indices.ptr.?, // Direct usize pointer
                    grad_weight.ptr.?, // Direct f64 pointer
                    num_embeddings,
                    embedding_dim,
                    batch_size,
                    sequence_length,
                    stream.stream,
                )),
                else => unreachable, // Only Bf16, f16, f32, f64 are supported
            }

            if (is_debugging and try grad_weight.hasNaN(stream)) {
                return error.HasNan;
            }

            return grad_weight.move();
        }
    };
}
