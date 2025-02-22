const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;

pub fn TensorFillRandom(comptime T: type, comptime rank: comptime_int) type {
    return struct {
        const Self = GPUTensor(T, rank);

        pub fn fillUniform(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            try cuda_context.setCurandStream(stream);

            switch (T) {
                f32 => {
                    try err.checkCurand(c.curandGenerateUniform(cuda_context.curand_generator, self.ptr, self.calcLen()));
                },
                f64 => {
                    try err.checkCurand(c.curandGenerateUniformDouble(cuda_context.curand_generator, self.ptr, self.calcLen()));
                },
                else => unreachable,
            }
        }

        pub fn fillUniformRange(
            self: *Self,
            min: T,
            max: T,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            try cuda_context.setCurandStream(stream);
            try self.fillUniform(cuda_context, stream);
            const range = max - min;
            const offset = min;
            try self.scaleShift(range, offset, cuda_context, stream);
        }

        pub fn fillNormalDistribution(
            self: *Self,
            mean: T,
            stddev: T,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            try cuda_context.setCurandStream(stream);

            switch (T) {
                f32 => {
                    try err.checkCurand(c.curandGenerateNormal(cuda_context.curand_generator, self.ptr, self.calcLen(), mean, stddev));
                },
                f64 => {
                    try err.checkCurand(c.curandGenerateNormalDouble(cuda_context.curand_generator, self.ptr, self.calcLen(), mean, stddev));
                },
                else => unreachable,
            }
        }

        pub fn fillXavierUniform(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            const fan_in, const fan_out = self.base.computeFanInOut();
            const limit = @sqrt(6.0 / @as(T, @floatFromInt(fan_in + fan_out)));
            try self.fillUniformRange(-limit, limit, cuda_context, stream);
        }

        pub fn fillHeNormal(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            const fan_in, _ = self.base.computeFanInOut();
            const stddev = @sqrt(2.0 / @as(T, @floatFromInt(fan_in)));
            try self.fillNormalDistribution(0.0, stddev, cuda_context, stream);
        }

        pub fn fillHeUniform(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            const fan_in, _ = self.base.computeFanInOut();
            const limit = @sqrt(3.0 / fan_in);
            try self.fillUniformRange(-limit, limit, cuda_context, stream);
        }
    };
}
