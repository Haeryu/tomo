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

        pub fn broadCastTo(
            self: *Self,
            new_shape: []const usize,
            stream: *const Stream,
        ) !GPUTensor(T) {
            for (self.base.getShapeConst(), new_shape) |in_dim, out_dim| {
                if (out_dim % in_dim != 0) {
                    return error.InvalidBroadcast;
                }
            }
            var out = try GPUTensor(T).initAsync(new_shape, stream);
            errdefer out.deinitAsync(stream);

            //   try out.fill(if (T == Bf16) Bf16.fromF32(0.0) else 0.0, stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoBroadcastToB(
                        @ptrCast(self.ptr),
                        @ptrCast(out.ptr),
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
                        @ptrCast(self.ptr),
                        @ptrCast(out.ptr),
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
                f32 => {
                    try err.checkCuda(c.tomoBroadcastToF(
                        self.ptr,
                        out.ptr,
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
                f64 => {
                    try err.checkCuda(c.tomoBroadcastToD(
                        self.ptr,
                        out.ptr,
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
                else => unreachable,
            }

            return out;
        }

        pub fn computeOutShape(
            allocator: std.mem.Allocator,
            in_shape: []const usize,
            axes: []const isize,
            keepdims: bool,
        ) !std.meta.Tuple(&.{ []usize, []usize }) {
            // Allocate a temporary boolean array to mark summed axes.
            var to_sum = try allocator.alloc(bool, in_shape.len);
            defer allocator.free(to_sum);
            @memset(to_sum, false);
            // Mark each axis specified in `axes`.
            for (axes) |axis| {
                var a = axis;
                if (a < 0) {
                    a += @intCast(in_shape.len);
                }
                if (a < 0 or a >= in_shape.len) {
                    return error.InvalidAxis;
                }
                to_sum[@intCast(a)] = true;
            }

            // Create an ArrayList to build the output shape.
            var out = std.ArrayList(usize).init(allocator);
            defer out.deinit();

            var out_keep_dims_shape = std.ArrayList(usize).init(allocator);
            defer out_keep_dims_shape.deinit();

            for (in_shape, 0..) |dim, i| {
                if (to_sum[i]) {
                    if (keepdims) {
                        try out.append(1);
                    }
                    try out_keep_dims_shape.append(1);
                } else {
                    try out.append(dim);
                    try out_keep_dims_shape.append(dim);
                }
            }

            return .{
                try out.toOwnedSlice(),
                try out_keep_dims_shape.toOwnedSlice(),
            };
        }

        pub fn sum(
            self: *Self,
            allocator: std.mem.Allocator,
            axes: []const isize,
            keepdims: bool,
            stream: *const Stream,
        ) !GPUTensor(T) {
            const new_shape, const new_shape_keepdims = try computeOutShape(allocator, self.base.getShape(), axes, keepdims);
            defer allocator.free(new_shape);
            defer allocator.free(new_shape_keepdims);

            // Always create output with keepdims=true shape
            var out = try GPUTensor(T).initAsync(new_shape_keepdims, stream);
            errdefer out.deinitAsync(stream);
            try out.fill(if (T == Bf16) Bf16.fromF32(0.0) else 0.0, stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoSumToB(
                        @ptrCast(self.ptr),
                        @ptrCast(out.ptr),
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape_keepdims.ptr,
                        new_shape_keepdims.len,
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
                        @ptrCast(self.ptr),
                        @ptrCast(out.ptr),
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape_keepdims.ptr,
                        new_shape_keepdims.len,
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
                    try err.checkCuda(c.tomoSumToF(
                        self.ptr,
                        out.ptr,
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape_keepdims.ptr,
                        new_shape_keepdims.len,
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
                f64 => {
                    try err.checkCuda(c.tomoSumToD(
                        self.ptr,
                        out.ptr,
                        self.base.getShape().ptr,
                        self.base.getShape().len,
                        new_shape_keepdims.ptr,
                        new_shape_keepdims.len,
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
                else => unreachable,
            }

            if (!keepdims) {
                try out.squeeze(allocator);
            }

            return out;
        }
    };
}
