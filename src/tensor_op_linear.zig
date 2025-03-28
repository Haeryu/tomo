const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const Bf16 = @import("bf16.zig").BF16;

pub fn TensorOpLinear(comptime T: type) type {
    return struct {
        const Self = GPUTensor(T);

        pub fn linear(self: *const Self, other: *const Self, bias: ?*const Self, stream: *const Stream) !Self {
            std.debug.assert(self.base.getCol() == other.base.getRow());
            if (bias) |b| {
                std.debug.assert(b.base.getRow() == self.base.getRow());
                std.debug.assert(b.base.getCol() == other.base.getCol());
            }
            var res = try Self.initAsync(&.{ self.base.getRow(), other.base.getCol() }, stream);
            errdefer res.deinitAsync(stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoLinearB(
                        @ptrCast(self.ptr.?),
                        @ptrCast(other.ptr.?),
                        self.base.getRow(),
                        self.base.getCol(),
                        other.base.getCol(),
                        if (bias) |b| @ptrCast(b.ptr) else null,
                        @ptrCast(res.ptr),
                        stream.stream,
                    ));
                },
                f16 => {
                    try err.checkCuda(c.tomoLinearH(
                        @ptrCast(self.ptr.?),
                        @ptrCast(other.ptr.?),
                        self.base.getRow(),
                        self.base.getCol(),
                        other.base.getCol(),
                        if (bias) |b| @ptrCast(b.ptr) else null,
                        @ptrCast(res.ptr),
                        stream.stream,
                    ));
                },
                f32 => {
                    try err.checkCuda(c.tomoLinearF(
                        self.ptr.?,
                        other.ptr.?,
                        self.base.getRow(),
                        self.base.getCol(),
                        other.base.getCol(),
                        if (bias) |b| b.ptr else null,
                        res.ptr,
                        stream.stream,
                    ));
                },
                f64 => {
                    try err.checkCuda(c.tomoLinearD(
                        self.ptr.?,
                        other.ptr.?,
                        self.base.getRow(),
                        self.base.getCol(),
                        other.base.getCol(),
                        if (bias) |b| b.ptr else null,
                        res.ptr,
                        stream.stream,
                    ));
                },
                else => unreachable,
            }

            return res.move();
        }

        fn splitMatrixDims(shape: []const usize) !struct {
            batch_shape: []const usize,
            M: usize,
            K: usize,
        } {
            if (shape.len < 2) {
                return error.InvalidShape;
            }
            const rank = shape.len;
            return .{
                .batch_shape = shape[0..(rank - 2)],
                .M = shape[rank - 2],
                .K = shape[rank - 1],
            };
        }

        fn assertSameBatchShape(lhs_batch: []const usize, rhs_batch: []const usize) !void {
            // same length
            if (lhs_batch.len != rhs_batch.len) {
                return error.IncompatibleBatch;
            }
            // same dimensions
            for (lhs_batch, rhs_batch) |lhs_dim, rhs_dim| {
                if (lhs_dim != rhs_dim) {
                    return error.IncompatibleBatch;
                }
            }
        }

        fn computeBatchSize(batch_shape: []const usize) usize {
            var bs: usize = 1;
            for (batch_shape) |dim| {
                bs *= dim;
            }
            return bs;
        }

        pub fn linearImp(self: *const Self, other: *const Self, bias: ?*const Self, stream: *const Stream) !Self {
            const lhs_dims = try splitMatrixDims(self.base.getShapeConst());
            const rhs_dims = try splitMatrixDims(other.base.getShapeConst());
            std.debug.assert(lhs_dims.K == rhs_dims.M);
            try assertSameBatchShape(lhs_dims.batch_shape, rhs_dims.batch_shape);

            if (bias) |b| {
                const bias_dims = try splitMatrixDims(b.base.getShapeConst());
                try assertSameBatchShape(lhs_dims.batch_shape, rhs_dims.batch_shape);
                std.debug.assert(bias_dims.M == lhs_dims.M);
                std.debug.assert(bias_dims.K == rhs_dims.K);
            }

            const M = lhs_dims.M;
            const N = rhs_dims.K;

            const batch_size = computeBatchSize(lhs_dims.batch_shape);

            var out_shape: [Self.max_rank]usize = undefined;
            const out_rank = lhs_dims.batch_shape.len + 2;

            std.mem.copyForwards(usize, out_shape[0..lhs_dims.batch_shape.len], lhs_dims.batch_shape);

            out_shape[out_rank - 2] = M;
            out_shape[out_rank - 1] = N;

            var res = try Self.initAsync(out_shape[0..out_rank], stream);
            errdefer res.deinitAsync(stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoLinearImpB(
                        @ptrCast(self.ptr.?),
                        @ptrCast(other.ptr.?),
                        self.base.getRow(),
                        self.base.getCol(),
                        other.base.getCol(),
                        if (bias) |b| @ptrCast(b.ptr) else null,
                        @ptrCast(res.ptr),
                        batch_size,
                        stream.stream,
                    ));
                },
                f16 => {
                    try err.checkCuda(c.tomoLinearImpH(
                        @ptrCast(self.ptr.?),
                        @ptrCast(other.ptr.?),
                        self.base.getRow(),
                        self.base.getCol(),
                        other.base.getCol(),
                        if (bias) |b| @ptrCast(b.ptr) else null,
                        @ptrCast(res.ptr),
                        batch_size,
                        stream.stream,
                    ));
                },
                f32 => {
                    try err.checkCuda(c.tomoLinearImpF(
                        self.ptr.?,
                        other.ptr.?,
                        self.base.getRow(),
                        self.base.getCol(),
                        other.base.getCol(),
                        if (bias) |b| b.ptr else null,
                        res.ptr,
                        batch_size,
                        stream.stream,
                    ));
                },
                f64 => {
                    try err.checkCuda(c.tomoLinearImpD(
                        self.ptr.?,
                        other.ptr.?,
                        self.base.getRow(),
                        self.base.getCol(),
                        other.base.getCol(),
                        if (bias) |b| b.ptr else null,
                        res.ptr,
                        batch_size,
                        stream.stream,
                    ));
                },
                else => unreachable,
            }

            return res.move();
        }
    };
}
