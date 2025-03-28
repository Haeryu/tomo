const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const BF16 = @import("bf16.zig").BF16;

pub fn TensorFillRandom(comptime T: type) type {
    return struct {
        const Self = GPUTensor(T);

        pub fn fillUniform(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            try cuda_context.setCurandStream(stream);

            switch (T) {
                BF16 => {
                    var f32_buf: GPUTensor(f32) = try .initAsync(self.base.getShapeConst(), stream);
                    defer f32_buf.deinitAsync(stream);

                    try f32_buf.fillUniform(cuda_context, stream);

                    var bf16_buf = try f32_buf.cast(BF16, stream);
                    defer bf16_buf.deinitAsync(stream);

                    std.mem.swap(?[*]T, &self.ptr, &bf16_buf.ptr);
                },
                f16 => {
                    var f32_buf: GPUTensor(f32) = try .initAsync(self.base.getShapeConst(), stream);
                    defer f32_buf.deinitAsync(stream);

                    try f32_buf.fillUniform(cuda_context, stream);

                    var f16_buf = try f32_buf.cast(f16, stream);
                    defer f16_buf.deinitAsync(stream);

                    std.mem.swap(?[*]T, &self.ptr, &f16_buf.ptr);
                },
                f32 => {
                    try err.checkCurand(c.curandGenerateUniform(cuda_context.curand_generator, self.ptr.?, self.calcLen()));
                },
                f64 => {
                    try err.checkCurand(c.curandGenerateUniformDouble(cuda_context.curand_generator, self.ptr.?, self.calcLen()));
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
            const range = if (T != BF16) max - min else BF16.sub(max, min);
            const offset = min;
            try self.scaleShift(range, offset, stream);
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
                BF16 => {
                    var f32_buf: GPUTensor(f32) = try .initAsync(self.base.getShapeConst(), stream);
                    defer f32_buf.deinitAsync(stream);

                    try f32_buf.fillNormalDistribution(mean.toF32(), stddev.toF32(), cuda_context, stream);

                    var bf16_buf = try f32_buf.cast(BF16, stream);
                    defer bf16_buf.deinitAsync(stream);

                    std.mem.swap(?[*]T, &self.ptr, &bf16_buf.ptr);
                },
                f16 => {
                    var f32_buf: GPUTensor(f32) = try .initAsync(self.base.getShapeConst(), stream);
                    defer f32_buf.deinitAsync(stream);

                    try f32_buf.fillNormalDistribution(mean, stddev, cuda_context, stream);

                    var f16_buf = try f32_buf.cast(f16, stream);
                    defer f16_buf.deinitAsync(stream);

                    std.mem.swap(?[*]T, &self.ptr, &f16_buf.ptr);
                },
                f32 => {
                    try err.checkCurand(c.curandGenerateNormal(cuda_context.curand_generator, self.ptr.?, self.calcLen(), mean, stddev));
                },
                f64 => {
                    try err.checkCurand(c.curandGenerateNormalDouble(cuda_context.curand_generator, self.ptr.?, self.calcLen(), mean, stddev));
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
            const limit = if (T != BF16) @sqrt(6.0 / @as(T, @floatFromInt(fan_in + fan_out))) else BF16.div(BF16.fromF32(6.0), BF16.fromF32(@floatFromInt(fan_in + fan_out)));
            try self.fillUniformRange(if (T != BF16) -limit else limit.neg(), limit, cuda_context, stream);
        }

        pub fn fillHeNormal(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            const fan_in, _ = self.base.computeFanInOut();
            const stddev = if (T != BF16) @sqrt(2.0 / @as(T, @floatFromInt(fan_in))) else BF16.div(BF16.fromF32(2.0), BF16.fromF32(@floatFromInt(fan_in)));
            try self.fillNormalDistribution(if (T != BF16) 0.0 else BF16.fromF32(0.0), stddev, cuda_context, stream);
        }

        pub fn fillHeUniform(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            const fan_in, _ = self.base.computeFanInOut();
            const limit = if (T != BF16) @sqrt(6.0 / @as(T, @floatFromInt(fan_in))) else BF16.div(BF16.fromF32(6.0), BF16.fromF32(@floatFromInt(fan_in)));
            try self.fillUniformRange(if (T != BF16) -limit else limit.neg(), limit, cuda_context, stream);
        }
    };
}
