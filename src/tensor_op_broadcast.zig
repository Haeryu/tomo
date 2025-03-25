const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const Bf16 = @import("bf16.zig").BF16;

pub fn TensorOpBroadCast(comptime T: type) type {
    return struct {
        const Self = GPUTensor(T);

        pub fn broadcastTo(
            self: *const Self,
            new_shape: []const usize,
            stream: *const Stream,
        ) !GPUTensor(T) {
            if (std.mem.eql(usize, self.base.getShapeConst(), new_shape)) {
                return self.cloneAsync(stream);
            }

            // for (self.base.getShapeConst(), new_shape) |in_dim, out_dim| {
            //     if (out_dim == 0 or out_dim % in_dim != 0) {
            //         return error.InvalidBroadcast;
            //     }
            // }

            const self_shape = self.base.getShapeConst();
            const max_dims = @max(self_shape.len, new_shape.len);

            // Allocate arrays for padded shapes (assuming a reasonable max dimension limit)
            var padded_self_shape: [GPUTensor(T).max_rank]usize = undefined;
            var padded_new_shape: [GPUTensor(T).max_rank]usize = undefined;

            // Pad self_shape with 1s on the left
            for (0..max_dims) |i| {
                if (i < max_dims - self_shape.len) {
                    padded_self_shape[i] = 1;
                } else {
                    padded_self_shape[i] = self_shape[i - (max_dims - self_shape.len)];
                }
            }

            // Pad new_shape with 1s on the left (if needed, though typically new_shape is larger)
            for (0..max_dims) |i| {
                if (i < max_dims - new_shape.len) {
                    padded_new_shape[i] = 1;
                } else {
                    padded_new_shape[i] = new_shape[i - (max_dims - new_shape.len)];
                }
            }

            // Check compatibility and proceed with broadcasting
            for (padded_self_shape[0..max_dims], padded_new_shape[0..max_dims]) |in_dim, out_dim| {
                if (in_dim != out_dim and in_dim != 1) {
                    return error.BroadcastDimensionMismatch;
                }
            }

            var out = try GPUTensor(T).initAsync(new_shape, stream);
            errdefer out.deinitAsync(stream);

            //   try out.fill(if (T == Bf16) Bf16.fromF32(0.0) else 0.0, stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoBroadcastToB(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShape().len,
                        stream.stream,
                    ));
                },
                f16 => {
                    try err.checkCuda(c.tomoBroadcastToH(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                f32 => {
                    try err.checkCuda(c.tomoBroadcastToF(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                f64 => {
                    try err.checkCuda(c.tomoBroadcastToD(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                else => unreachable,
            }

            return out;
        }

        pub fn computeOutShape(
            allocator: std.mem.Allocator,
            in_shape: []const usize,
            axes: ?[]const isize,
            keepdims: bool,
        ) ![]usize {
            // If `axes` is null, sum over all dimensions [0..in_shape.len).
            if (axes == null) {
                var all_axes = try allocator.alloc(isize, in_shape.len);
                defer allocator.free(all_axes);

                for (0..in_shape.len) |i| {
                    all_axes[i] = @intCast(i);
                }
                return computeOutShape(allocator, in_shape, all_axes, keepdims);
            }

            // Mark which axes we're summing.
            var to_sum = try allocator.alloc(bool, in_shape.len);
            defer allocator.free(to_sum);
            @memset(to_sum, false);

            for (axes.?) |axis| {
                var a = axis;
                if (a < 0) {
                    a += @intCast(in_shape.len);
                }
                if (a < 0 or a >= in_shape.len) {
                    return error.InvalidAxis;
                }
                to_sum[@intCast(a)] = true;
            }

            // Build the output shape
            var out_arrlist = std.ArrayList(usize).init(allocator);
            defer out_arrlist.deinit();

            // If keepdims==false, skip those dimensions entirely
            // If keepdims==true, append "1" for those dimensions
            for (in_shape, 0..) |dim, i| {
                if (to_sum[i]) {
                    if (keepdims) {
                        try out_arrlist.append(1);
                    } else {
                        // skip dimension
                    }
                } else {
                    try out_arrlist.append(dim);
                }
            }

            return try out_arrlist.toOwnedSlice();
        }

        pub fn sum(
            self: *const Self,
            allocator: std.mem.Allocator,
            axes: ?[]const isize,
            keepdims: bool,
            stream: *const Stream,
        ) !GPUTensor(T) {
            const new_shape = try computeOutShape(allocator, self.base.getShapeConst(), axes, keepdims);
            defer allocator.free(new_shape);

            // Always create output with keepdims=true shape
            var out = try GPUTensor(T).initAsync(new_shape, stream);
            errdefer out.deinitAsync(stream);
            try out.fill(if (T == Bf16) Bf16.fromF32(0.0) else 0.0, stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoSumToB(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShape().len,
                        stream.stream,
                    ));
                },
                f16 => {
                    try err.checkCuda(c.tomoSumToH(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                f32 => {
                    try err.checkCuda(c.tomoSumToF(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                f64 => {
                    try err.checkCuda(c.tomoSumToD(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                else => unreachable,
            }

            // if (!keepdims) {
            //     try out.squeeze(allocator);
            // }

            return out;
        }

        pub fn sumTo(
            self: *const Self,
            allocator: std.mem.Allocator,
            shape: []const usize,
            stream: *const Stream,
        ) !GPUTensor(T) {
            const target_ndim = shape.len;
            const lead = self.base.getShapeConst().len - target_ndim;

            var lead_axis: [Self.max_rank]usize = undefined;
            var lead_axis_count: usize = 0;
            var i: usize = 0;
            while (i < lead) : (i += 1) {
                lead_axis[lead_axis_count] = i;
                lead_axis_count += 1;
            }

            var ones_axis: [Self.max_rank]usize = undefined;
            var ones_axis_count: usize = 0;
            i = 0;
            while (i < target_ndim) : (i += 1) {
                if (shape[i] == 1) {
                    // Offset by the number of lead dimensions.
                    ones_axis[ones_axis_count] = i + lead;
                    ones_axis_count += 1;
                }
            }

            var sum_axes: [Self.max_rank]isize = undefined;
            var sum_axes_count: usize = 0;
            i = 0;
            while (i < lead_axis_count) : (i += 1) {
                sum_axes[sum_axes_count] = @intCast(lead_axis[i]);
                sum_axes_count += 1;
            }
            i = 0;
            while (i < ones_axis_count) : (i += 1) {
                sum_axes[sum_axes_count] = @intCast(ones_axis[i]);
                sum_axes_count += 1;
            }

            return try self.sum(allocator, sum_axes[0..sum_axes_count], true, stream);
        }

        pub fn transpose(
            self: *const Self,
            stream: *const Stream,
        ) !Self {
            var res = try Self.initAsync(&.{ self.base.getCol(), self.base.getRow() }, stream);
            errdefer res.deinitAsync(stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoTransposeB(@ptrCast(self.ptr.?), self.base.getRow(), self.base.getCol(), @ptrCast(res.ptr), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoTransposeH(@ptrCast(self.ptr.?), self.base.getRow(), self.base.getCol(), @ptrCast(res.ptr), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoTransposeF(self.ptr.?, self.base.getRow(), self.base.getCol(), res.ptr, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoTransposeD(self.ptr.?.?, self.base.getRow(), self.base.getCol(), res.ptr, stream.stream));
                },
                else => unreachable,
            }

            return res.move();
        }

        pub fn max(
            self: *const Self,
            allocator: std.mem.Allocator,
            axes: ?[]const isize,
            keepdims: bool,
            stream: *const Stream,
        ) !GPUTensor(T) {
            const new_shape = try computeOutShape(allocator, self.base.getShapeConst(), axes, keepdims);
            defer allocator.free(new_shape);

            // Create output tensor with keepdims=true shape.
            var out = try GPUTensor(T).initAsync(new_shape, stream);
            errdefer out.deinitAsync(stream);
            // For max reduction, initialize with negative infinity.
            // try out.fill(if (T == Bf16) Bf16.fromF32(-std.math.inf(f32)) else if (T == f16) -std.math.inf(f32) else -std.math.inf(f32), stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoMaxToB(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShape().len,
                        stream.stream,
                    ));
                },
                f16 => {
                    try err.checkCuda(c.tomoMaxToH(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                f32 => {
                    try err.checkCuda(c.tomoMaxToF(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                f64 => {
                    try err.checkCuda(c.tomoMaxToD(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                else => unreachable,
            }

            // if (!keepdims) {
            //     try out.squeeze(allocator);
            // }

            return out;
        }

        pub fn min(
            self: *const Self,
            allocator: std.mem.Allocator,
            axes: ?[]const isize,
            keepdims: bool,
            stream: *const Stream,
        ) !GPUTensor(T) {
            const new_shape = try computeOutShape(allocator, self.base.getShapeConst(), axes, keepdims);
            defer allocator.free(new_shape);

            // Create output tensor with keepdims=true shape.
            var out = try GPUTensor(T).initAsync(new_shape, stream);
            errdefer out.deinitAsync(stream);
            // For min reduction, initialize with positive infinity.
            //   try out.fill(if (T == Bf16) Bf16.fromF32(std.math.inf(f32)) else if (T == f16) std.math.inf(f32) else std.math.inf(f32), stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoMinToB(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShape().len,
                        stream.stream,
                    ));
                },
                f16 => {
                    try err.checkCuda(c.tomoMinToH(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShape().len,
                        stream.stream,
                    ));
                },
                f32 => {
                    try err.checkCuda(c.tomoMinToF(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                f64 => {
                    try err.checkCuda(c.tomoMinToD(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                else => unreachable,
            }

            // if (!keepdims) {
            //     try out.squeeze(allocator);
            // }

            return out;
        }

        pub fn argmax(
            self: *const Self,
            allocator: std.mem.Allocator,
            axes: ?[]const isize,
            keepdims: bool,
            stream: *const Stream,
        ) !GPUTensor(usize) {
            const new_shape = try computeOutShape(allocator, self.base.getShapeConst(), axes, keepdims);
            defer allocator.free(new_shape);

            // Create output tensor with keepdims=true shape.
            var out = try GPUTensor(usize).initAsync(new_shape, stream);
            errdefer out.deinitAsync(stream);
            // For min reduction, initialize with positive infinity.
            //   try out.fill(if (T == Bf16) Bf16.fromF32(std.math.inf(f32)) else if (T == f16) std.math.inf(f32) else std.math.inf(f32), stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoArgmaxH(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        //  self.calcLen(),
                        out.calcLen(),
                        self.base.getShape().len,
                        stream.stream,
                    ));
                },
                f16 => {
                    try err.checkCuda(c.tomoArgmaxB(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        // self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                f32 => {
                    try err.checkCuda(c.tomoArgmaxF(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        //self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                f64 => {
                    try err.checkCuda(c.tomoArgmaxD(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        // self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                else => unreachable,
            }

            // if (!keepdims) {
            //     try out.squeeze(allocator);
            // }

            return out;
        }

        pub fn argmin(
            self: *const Self,
            allocator: std.mem.Allocator,
            axes: ?[]const isize,
            keepdims: bool,
            stream: *const Stream,
        ) !GPUTensor(usize) {
            const new_shape = try computeOutShape(allocator, self.base.getShapeConst(), axes, keepdims);
            defer allocator.free(new_shape);

            // Create output tensor with keepdims=true shape.
            var out = try GPUTensor(usize).initAsync(new_shape, stream);
            errdefer out.deinitAsync(stream);
            // For min reduction, initialize with positive infinity.
            //   try out.fill(if (T == Bf16) Bf16.fromF32(std.math.inf(f32)) else if (T == f16) std.math.inf(f32) else std.math.inf(f32), stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoArgminH(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShape().len,
                        stream.stream,
                    ));
                },
                f16 => {
                    try err.checkCuda(c.tomoArgminB(
                        @ptrCast(self.ptr.?),
                        @ptrCast(out.ptr.?),
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShape().len,
                        stream.stream,
                    ));
                },
                f32 => {
                    try err.checkCuda(c.tomoArgminF(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                f64 => {
                    try err.checkCuda(c.tomoArgminD(
                        self.ptr.?,
                        out.ptr.?,
                        self.base.getShapeConst().ptr,
                        self.base.getShapeConst().len,
                        new_shape.ptr,
                        new_shape.len,
                        self.base.getStrides().ptr,
                        self.base.getStrides().len,
                        out.base.getStrides().ptr,
                        out.base.getStrides().len,
                        self.calcLen(),
                        out.calcLen(),
                        self.base.getShapeConst().len,
                        stream.stream,
                    ));
                },
                else => unreachable,
            }

            // if (!keepdims) {
            //     try out.squeeze(allocator);
            // }

            return out;
        }

        pub fn mean(
            self: *const Self,
            allocator: std.mem.Allocator,
            axes: ?[]const isize,
            keepdims: bool,
            stream: *const Stream,
        ) !GPUTensor(T) {
            const shape = self.base.getShapeConst();

            const reduced_axes = axes orelse blk: {
                const axes_all = try allocator.alloc(isize, shape.len);
                errdefer allocator.free(axes_all);

                for (0..shape.len) |i| {
                    axes_all[i] = @intCast(i);
                }

                break :blk axes_all;
            };
            defer if (axes == null) allocator.free(reduced_axes);

            var num_elements: usize = 1;
            for (reduced_axes) |ax| {
                const axis: usize = if (ax >= 0) @intCast(ax) else @as(usize, @intCast(ax)) + shape.len;
                num_elements *= shape[axis];
            }
            if (num_elements == 0) num_elements = self.calcLen();

            var out = try self.sum(allocator, axes, keepdims, stream);
            errdefer out.deinitAsync(stream);

            const scale_factor = 1.0 / @as(T, @floatFromInt(num_elements));
            try out.scale(scale_factor, stream);

            return out;
        }

        pub fn meanInt(
            self: *const Self,
            comptime F: type,
            allocator: std.mem.Allocator,
            axes: ?[]const isize,
            keepdims: bool,
            stream: *const Stream,
        ) !GPUTensor(F) {
            // var out = try self.sum(allocator, axes, keepdims, stream);
            // defer out.deinitAsync(stream);

            var outf = try self.cast(F, stream);
            defer outf.deinitAsync(stream);

            return try outf.mean(allocator, axes, keepdims, stream);
        }

        pub fn countElementsAlongAxes(self: *const Self, axes: ?[]const isize) usize {
            const shape = self.base.getShapeConst();
            if (axes == null) {
                return self.calcLen();
            }
            var count: usize = 1;
            for (axes.?) |ax| {
                const d: usize = if (ax < 0) @as(usize, @intCast(ax)) + shape.len else @as(usize, @intCast(ax));
                count *= shape[d];
            }
            return count;
        }

        pub fn varianceBiased(
            self: *const Self,
            allocator: std.mem.Allocator,
            axes: ?[]const isize,
            keepdims: bool,
            stream: *const Stream,
        ) !GPUTensor(T) {
            var x = try self.cloneAsync(stream);
            defer x.deinitAsync(stream);

            var m = try x.mean(allocator, axes, true, stream);
            defer m.deinitAsync(stream);

            var m_broad = try m.broadcastTo(x.base.getShapeConst(), stream);
            defer m_broad.deinitAsync(stream);

            try x.sub(&m_broad, stream);
            try x.square(stream);

            return try x.mean(allocator, axes, keepdims, stream);
        }

        pub fn varianceUnBiased(
            self: *const Self,
            allocator: std.mem.Allocator,
            axes: ?[]const isize,
            keepdims: bool,
            stream: *const Stream,
        ) !GPUTensor(T) {
            var biased = try self.varianceBiased(allocator, axes, keepdims, stream);
            errdefer biased.deinitAsync(stream);

            const n = biased.countElementsAlongAxes(axes);
            if (n <= 1) return error.InvalidShape;

            // scale out by n/(n-1)
            const scale_val = @as(T, @floatFromInt(n)) / @as(T, @floatFromInt(n - 1));
            try biased.scale(@as(T, scale_val), stream);

            return biased.move();
        }
    };
}
