const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const Bf16 = @import("bf16.zig").BF16;

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

            return items.move();
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

            return out_tensor.move();
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

            return out_tensor.move();
        }

        pub fn rollaxis(
            self: *const Self,
            axis: usize,
            start: usize,
            stream: *const Stream,
        ) !Self {
            const in_shape = self.base.getShapeConst();
            const nd = in_shape.len;

            var out_tensor = try Self.initAsync(in_shape, stream);
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

            return out_tensor.move();
        }
    };
}
