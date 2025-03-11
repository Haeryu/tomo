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
                    self.base.getShapeConst().len,
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
                    self.base.getShapeConst().len,
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
                    self.base.getShapeConst().len,
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
                    self.base.getShapeConst().len,
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
                    self.base.getShapeConst().len,
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
                    self.base.getShapeConst().len,
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
                    self.base.getShapeConst().len,
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
                    self.base.getShapeConst().len,
                    out_size,
                    stream.stream,
                )),
                else => unreachable,
            }

            return gx.move();
        }

        pub fn toOneHot(indices: *const GPUTensor(usize), num_classes: usize, stream: *const Stream) !Self {
            if (indices.base.rank != 1) return error.InvalidIndicesRank;

            const batch_size = indices.base.getShapeConst()[0];

            var one_hot = try Self.initAsync(&[_]usize{ batch_size, num_classes }, stream);
            errdefer one_hot.deinitAsync(stream);

            switch (T) {
                Bf16 => try err.checkCuda(c.tomoOneHotB(indices.ptr.?, @ptrCast(one_hot.ptr.?), batch_size, num_classes, stream.stream)),
                f16 => try err.checkCuda(c.tomoOneHotH(indices.ptr.?, @ptrCast(one_hot.ptr.?), batch_size, num_classes, stream.stream)),
                f32 => try err.checkCuda(c.tomoOneHotF(one_hot.ptr.?, indices.ptr.?, batch_size, num_classes, stream.stream)),
                f64 => try err.checkCuda(c.tomoOneHotD(indices.ptr.?, one_hot.ptr.?, batch_size, num_classes, stream.stream)),
                else => unreachable,
            }

            return one_hot.move();
        }
    };
}
