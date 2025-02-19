const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const TensorOpReduction = @import("tensor_op_reduction.zig").TensorOpReduction;

pub fn TensorOpElemwise(comptime T: type, comptime rank: comptime_int) type {
    return struct {
        const Self = GPUTensor(T, rank);

        pub fn axpy(
            self: *const Self,
            alpha: T,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null); // initialized and synced

            if (self.base.countElem() != result.base.countElem()) {
                return error.SizeMismatch;
            }

            const n = self.base.countElem();

            const incx: i64 = 1;
            const incy: i64 = 1;

            try err.checkCublas(c.cublasSetStream(cuda_context.cublas_handle, stream.stream));

            switch (T) {
                f32 => {
                    try err.checkCublas(c.cublasSaxpy_64(
                        cuda_context.cublas_handle,
                        @intCast(n),
                        &alpha,
                        @ptrCast(self.ptr.?),
                        incx,
                        @ptrCast(result.ptr.?),
                        incy,
                    ));
                },
                f64 => {
                    try err.checkCublas(c.cublasDaxpy_64(
                        cuda_context.cublas_handle,
                        @intCast(n),
                        &alpha,
                        @ptrCast(self.ptr.?),
                        incx,
                        @ptrCast(result.ptr.?),
                        incy,
                    ));
                },
                else => unreachable,
            }
        }

        pub fn gmm(
            self: *const Self,
            side: c.cublasSideMode_t,
            x: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null);
            const n = self.base.countElem();

            try err.checkCublas(c.cublasSetStream(cuda_context.cublas_handle, stream.stream));

            switch (T) {
                f32 => {
                    try err.checkCublas(c.cublasSdgmm_64(
                        cuda_context.cublas_handle,
                        side,
                        1,
                        @intCast(n),
                        @ptrCast(result.ptr.?),
                        1,
                        @ptrCast(x.ptr.?),
                        1,
                        @ptrCast(result.ptr.?),
                        1,
                    ));
                },
                f64 => {
                    try err.checkCublas(c.cublasDdgmm_64(
                        cuda_context.cublas_handle,
                        side,
                        1,
                        @intCast(n),
                        @ptrCast(result.ptr.?),
                        1,
                        @ptrCast(x.ptr.?),
                        1,
                        @ptrCast(result.ptr.?),
                        1,
                    ));
                },

                else => unreachable,
            }
        }

        pub fn matmulTransposed(
            self: *const Self,
            other: anytype,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null); // initialized and synced

            if (self.base.shape.len != 2 or other.base.shape.len != 2) {
                return error.OperationNotSupported;
            }

            if (self.base.shape[1] != other.base.shape[0]) {
                return error.SizeMismatch;
            }

            const m = self.base.shape[0];
            const k = self.base.shape[1];
            const n = other.base.shape[1];

            const alpha: T = 1;
            const beta: T = 0;

            try err.checkCublas(c.cublasSetStream(cuda_context.cublas_handle, stream.stream));

            switch (T) {
                f32 => {
                    try err.checkCublas(c.cublasSgemm_64(
                        cuda_context.cublas_handle,
                        c.CUBLAS_OP_T,
                        c.CUBLAS_OP_T,
                        @intCast(n),
                        @intCast(m),
                        @intCast(k),
                        &alpha,
                        @ptrCast(self.ptr.?),
                        @intCast(self.base.shape[1]),
                        @ptrCast(other.ptr.?),
                        @intCast(other.base.shape[1]),
                        &beta,
                        @ptrCast(result.ptr.?),
                        @intCast(result.base.shape[1]),
                    ));
                },
                f64 => {
                    try err.checkCublas(c.cublasDgemm_64(
                        cuda_context.cublas_handle,
                        c.CUBLAS_OP_T,
                        c.CUBLAS_OP_T,
                        @intCast(n),
                        @intCast(m),
                        @intCast(k),
                        &alpha,
                        @ptrCast(other.ptr.?),
                        @intCast(k),
                        @ptrCast(self.ptr.?),
                        @intCast(m),
                        &beta,
                        @ptrCast(result.ptr.?),
                        @intCast(n),
                    ));
                },
                else => unreachable,
            }
        }

        pub fn add(
            self: *const Self,
            other: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null); // initialized and synced
            try result.writeAsync(other.ptr.?, other.getLen(), 0, stream);
            try self.axpy(1, cuda_context.cublas_handle, stream, result);
        }

        pub fn sub(
            self: *const Self,
            other: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null); // initialized and synced\
            try result.writeAsync(other.ptr.?, other.getLen(), 0, stream);
            try self.axpy(
                other,
                -1,
                cuda_context.cublas_handle,
                stream,
                result,
            );
        }

        pub fn mul(
            self: *const Self,
            other: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null); // initialized and synced
            try result.writeAsync(other.ptr.?, other.getLen(), 0, stream);
            try self.gmm(
                c.CUBLAS_SIDE_RIGHT,
                other,
                cuda_context,
                stream,
                result,
            );
        }

        pub fn transpose(
            self: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null); // initialized and synced
            if (rank != 2) return error.OperationNotSupported;

            const m = self.base.shape[0];
            const n = self.base.shape[1];

            try err.checkCublas(c.cublasSetStream(cuda_context.cublas_handle, stream.stream));

            const alpha: T = 1;
            const beta: T = 0;
            switch (T) {
                f32 => {
                    try err.checkCublas(c.cublasSgeam(
                        cuda_context.cublas_handle,
                        c.CUBLAS_OP_T,
                        c.CUBLAS_OP_N,
                        @intCast(m),
                        @intCast(n),
                        &alpha,
                        @ptrCast(self.ptr.?),
                        @intCast(n),
                        &beta,
                        @ptrCast(result.ptr.?),
                        @intCast(m),
                        @ptrCast(result.ptr.?),
                        @intCast(m),
                    ));
                },
                f64 => {
                    try err.checkCublas(c.cublasDgeam(
                        cuda_context.cublas_handle,
                        c.CUBLAS_OP_T,
                        c.CUBLAS_OP_N,
                        @intCast(m),
                        @intCast(n),
                        &alpha,
                        @ptrCast(self.ptr.?),
                        @intCast(n),
                        &beta,
                        @ptrCast(result.ptr.?),
                        @intCast(m),
                        @ptrCast(result.ptr.?),
                        @intCast(m),
                    ));
                },
                else => unreachable,
            }
        }

        pub fn sin(self: *Self, stream: *const Stream) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoSinF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSinD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn cos(self: *Self, stream: *const Stream) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoCosF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoCosD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn tan(self: *Self, stream: *const Stream) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoTanF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoTanD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn relu(self: *Self, stream: *const Stream) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoReluF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoReluD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn leakyRelu(self: *Self, stream: *const Stream) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoLeakyReluF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoLeakyReluD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn inv(self: *Self, stream: *const Stream) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoInvF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoInvD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn elu(
            self: *Self,
            alpha: T,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoEluF(self.ptr, self.getLen(), alpha, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoEluD(self.ptr, self.getLen(), alpha, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn selu(
            self: *Self,
            alpha: T,
            lambda: T,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoSeluF(self.ptr, self.getLen(), alpha, lambda, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSeluD(self.ptr, self.getLen(), alpha, lambda, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn softplus(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoSoftplusF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSoftplusD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn sigmoid(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoSigmoidF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSigmoidD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn tanh(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoTanhF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoTanhD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn swish(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoSwishF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSwishD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn gelu(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoGeluF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoGeluD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn hardSigmoid(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoHardSigmoidF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoHardSigmoidD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn hardSwish(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoHardSwishF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoHardSwishD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn softsign(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoSoftsignF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSoftsignD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn square(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoSquareF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSquareD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn sqrt(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoSqrtF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSqrtD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn log(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoLogF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoLogD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn exp(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoExpF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoExpD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn abs(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoAbsF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoAbsD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn scale(
            self: *Self,
            factor: T,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoScaleF(self.ptr, self.getLen(), factor, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoScaleD(self.ptr, self.getLen(), factor, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn powf(
            self: *Self,
            exponent: T,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoPowfF(self.ptr, self.getLen(), exponent, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoPowfD(self.ptr, self.getLen(), exponent, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn pow(
            self: *Self,
            exponent: i32,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoPowF(self.ptr, self.getLen(), @intCast(exponent), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoPowD(self.ptr, self.getLen(), @intCast(exponent), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn clamp(
            self: *Self,
            lower: T,
            upper: T,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoClampF(self.ptr, self.getLen(), lower, upper, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoClampD(self.ptr, self.getLen(), lower, upper, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn ceil(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoCeilF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoCeilD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn floor(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoFloorF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoFloorD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn shift(
            self: *Self,
            offset: T,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoShiftF(self.ptr, self.getLen(), offset, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoShiftD(self.ptr, self.getLen(), offset, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn scaleShift(
            self: *Self,
            factor: T,
            offset: T,
            stream: *const Stream,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoScaleShiftF(self.ptr, self.getLen(), factor, offset, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoScaleShiftD(self.ptr, self.getLen(), factor, offset, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn softmax(
            self: *Self,
            stream: *const Stream,
        ) !void {
            var max = -std.math.floatMax(T);
            try self.max(stream, &max);
            try stream.sync();
            try self.shift(-max, stream);
            try self.exp(stream);
            var sum: T = 0.0;
            try self.sumReduce(stream, &sum);
            try stream.sync();
            try self.scale(1.0 / sum, stream);
        }
    };
}
