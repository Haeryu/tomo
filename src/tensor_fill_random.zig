const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const BF16 = @import("bf16.zig").BF16;

const is_debugging = @import("builtin").mode == .Debug;

pub fn TensorFillRandom(comptime T: type) type {
    return struct {
        const Self = GPUTensor(T);

        pub fn fillUniform(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            switch (T) {
                BF16 => {
                    const seed = std.time.microTimestamp();
                    try err.checkCurand(c.tomoFillUniformB(@ptrCast(self.ptr.?), self.calcLen(), @intCast(seed), stream.stream));
                },
                f16 => {
                    const seed = std.time.microTimestamp();
                    try err.checkCurand(c.tomoFillUniformH(@ptrCast(self.ptr.?), self.calcLen(), @intCast(seed), stream.stream));
                },
                f32 => {
                    try cuda_context.setCurandStream(stream);
                    try err.checkCurand(c.curandGenerateUniform(cuda_context.curand_generator, self.ptr.?, self.calcLen()));
                },
                f64 => {
                    try cuda_context.setCurandStream(stream);
                    try err.checkCurand(c.curandGenerateUniformDouble(cuda_context.curand_generator, self.ptr.?, self.calcLen()));
                },
                else => unreachable,
            }

            if (is_debugging and try self.hasNaN(stream)) {
                return error.HasNan;
            }
        }

        pub fn fillUniformRange(
            self: *Self,
            min: if (T != BF16) T else f32,
            max: if (T != BF16) T else f32,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            try cuda_context.setCurandStream(stream);
            try self.fillUniform(cuda_context, stream);
            const range = max - min;
            const offset = min;
            try self.scaleShift(range, offset, stream);
        }

        pub fn fillNormalDistribution(
            self: *Self,
            mean: if (T != BF16) T else f32,
            stddev: if (T != BF16) T else f32,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            try cuda_context.setCurandStream(stream);

            switch (T) {
                BF16 => {
                    const seed = std.time.microTimestamp();
                    try err.checkCurand(c.tomoFillNormalB(@ptrCast(self.ptr.?), self.calcLen(), mean, stddev, @intCast(seed), stream.stream));
                },
                f16 => {
                    const seed = std.time.microTimestamp();
                    try err.checkCurand(c.tomoFillNormalH(@ptrCast(self.ptr.?), self.calcLen(), @floatCast(mean), @floatCast(stddev), @intCast(seed), stream.stream));
                },
                f32 => {
                    try err.checkCurand(c.curandGenerateNormal(cuda_context.curand_generator, self.ptr.?, self.calcLen(), mean, stddev));
                },
                f64 => {
                    try err.checkCurand(c.curandGenerateNormalDouble(cuda_context.curand_generator, self.ptr.?, self.calcLen(), mean, stddev));
                },
                else => unreachable,
            }

            if (is_debugging and try self.hasNaN(stream)) {
                return error.HasNan;
            }
        }

        pub fn fillXavierUniform(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            const fan_in, const fan_out = self.base.computeFanInOut();
            const limit = @sqrt(6.0 / @as(if (T != BF16) T else f32, @floatFromInt(fan_in + fan_out)));
            try self.fillUniformRange(-limit, limit, cuda_context, stream);
        }

        pub fn fillHeNormal(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            const fan_in, _ = self.base.computeFanInOut();
            const stddev = @sqrt(2.0 / @as(if (T != BF16) T else f32, @floatFromInt(fan_in)));
            try self.fillNormalDistribution(0.0, stddev, cuda_context, stream);
        }

        pub fn fillHeUniform(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            const fan_in, _ = self.base.computeFanInOut();
            const limit = @sqrt(6.0 / @as(if (T != BF16) T else f32, @floatFromInt(fan_in)));
            try self.fillUniformRange(-limit, limit, cuda_context, stream);
        }
    };
}
