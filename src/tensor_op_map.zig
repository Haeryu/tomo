const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const TensorOpReduction = @import("tensor_op_reduction.zig").TensorOpReduction;
const Bf16 = @import("bf16.zig").BF16;

pub fn TensorOpMap(comptime T: type) type {
    return struct {
        const Self = GPUTensor(T);

        pub fn sin(self: *Self, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoSinB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoSinH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoSinF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSinD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn cos(self: *Self, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoCosB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoCosH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoCosF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoCosD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn tan(self: *Self, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoTanB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoTanH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoTanF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoTanD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn relu(self: *Self, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoReluB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoReluH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoReluF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoReluD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn leakyRelu(self: *Self, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoLeakyReluB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoLeakyReluH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoLeakyReluF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoLeakyReluD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn inv(self: *Self, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoInvB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoInvH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoInvF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoInvD(self.ptr.?, self.calcLen(), stream.stream));
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
                Bf16 => {
                    try err.checkCuda(c.tomoEluB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoEluH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoEluF(self.ptr.?, self.calcLen(), alpha, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoEluD(self.ptr.?, self.calcLen(), alpha, stream.stream));
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
                Bf16 => {
                    try err.checkCuda(c.tomoSeluB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoSeluH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoSeluF(self.ptr.?, self.calcLen(), alpha, lambda, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSeluD(self.ptr.?, self.calcLen(), alpha, lambda, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn softplus(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoSoftplusB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoSoftplusH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoSoftplusF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSoftplusD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn sigmoid(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoSigmoidB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoSigmoidH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoSigmoidF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSigmoidD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn tanh(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoTanhB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoTanhH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoTanhF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoTanhD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn swish(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoSwishB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoSwishH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoSwishF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSwishD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn gelu(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoGeluB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoGeluH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoGeluF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoGeluD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn hardSigmoid(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoHardSigmoidB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoHardSigmoidH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoHardSigmoidF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoHardSigmoidD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn hardSwish(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoHardSwishB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoHardSwishH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoHardSwishF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoHardSwishD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn softsign(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoSoftsignB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoSoftsignH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoSoftsignF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSoftsignD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn square(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoSquareB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoSquareH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoSquareF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSquareD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn sqrt(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoSqrtB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoSqrtH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoSqrtF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSqrtD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn log(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoLogB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoLogH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoLogF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoLogD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn exp(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoExpB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoExpH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoExpF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoExpD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn abs(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoAbsB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoAbsH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoAbsF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoAbsD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn scale(
            self: *Self,
            factor: if (T != Bf16) T else f32,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoScaleB(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(Bf16.fromF32(factor)), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoScaleH(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(factor), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoScaleF(self.ptr.?, self.calcLen(), factor, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoScaleD(self.ptr.?, self.calcLen(), factor, stream.stream));
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
                Bf16 => {
                    try err.checkCuda(c.tomoPowfB(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(exponent), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoPowfH(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(exponent), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoPowfF(self.ptr.?, self.calcLen(), exponent, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoPowfD(self.ptr.?, self.calcLen(), exponent, stream.stream));
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
                Bf16 => {
                    try err.checkCuda(c.tomoPowB(@ptrCast(self.ptr.?), self.calcLen(), @intCast(exponent), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoPowH(@ptrCast(self.ptr.?), self.calcLen(), @intCast(exponent), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoPowF(self.ptr.?, self.calcLen(), @intCast(exponent), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoPowD(self.ptr.?, self.calcLen(), @intCast(exponent), stream.stream));
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
                Bf16 => {
                    try err.checkCuda(c.tomoClampB(@ptrCast(self.ptr.?), self.calcLen(), lower, upper, stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoClampH(@ptrCast(self.ptr.?), self.calcLen(), lower, upper, stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoClampF(self.ptr.?, self.calcLen(), lower, upper, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoClampD(self.ptr.?, self.calcLen(), lower, upper, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn ceil(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoCeilB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoCeilH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoCeilF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoCeilD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn floor(
            self: *Self,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoFloorB(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoFloorH(@ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoFloorF(self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoFloorD(self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn shift(
            self: *Self,
            offset: if (T != Bf16) T else f32,
            stream: *const Stream,
        ) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoShiftB(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(Bf16.fromF32(offset)), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoShiftH(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(offset), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoShiftF(self.ptr.?, self.calcLen(), offset, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoShiftD(self.ptr.?, self.calcLen(), offset, stream.stream));
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
                Bf16 => {
                    try err.checkCuda(c.tomoScaleShiftB(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(factor), @bitCast(offset), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoScaleShiftH(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(factor), @bitCast(offset), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoScaleShiftF(self.ptr.?, self.calcLen(), factor, offset, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoScaleShiftD(self.ptr.?, self.calcLen(), factor, offset, stream.stream));
                },
                else => unreachable,
            }
        }

        // pub fn softmax(
        //     self: *Self,
        //     stream: *const Stream,
        // ) !void {
        //     switch (T) {
        //         Bf16 => {
        //             var max = Bf16.fromF32(0.0);
        //             try self.max(stream, &max);
        //             try stream.sync();
        //             try self.shift(max.neg(), stream);
        //             try self.exp(stream);
        //             var sum = Bf16.fromF32(0.0);
        //             try self.sumReduce(stream, &sum);
        //             try stream.sync();
        //             try self.scale(Bf16.fromF32(1.0).div(sum), stream);
        //         },
        //         else => {
        //             var max: T = 0.0;
        //             try self.max(stream, &max);
        //             try stream.sync();
        //             try self.shift(-max, stream);
        //             try self.exp(stream);
        //             var sum: T = 0.0;
        //             try self.sumReduce(stream, &sum);
        //             try stream.sync();
        //             try self.scale(1.0 / sum, stream);
        //         },
        //     }
        // }

        pub fn gt(self: *Self, num: T, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoGtB(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoGtH(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoGtF(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoGtD(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                usize => {
                    try err.checkCuda(c.tomoGtUZ(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn gtEq(self: *Self, num: T, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoGtEqB(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoGtEqH(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoGtEqF(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoGtEqD(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                usize => {
                    try err.checkCuda(c.tomoGtEqUZ(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn lt(self: *Self, num: T, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoLtB(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoLtH(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoLtF(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoLtD(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                usize => {
                    try err.checkCuda(c.tomoLtUZ(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn ltEq(self: *Self, num: T, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoLtEqB(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoLtEqH(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoLtEqF(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoLtEqD(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                usize => {
                    try err.checkCuda(c.tomoLtEqUZ(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn eq(self: *Self, num: T, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoEqB(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoEqH(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoEqF(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoEqD(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                usize => {
                    try err.checkCuda(c.tomoEqUZ(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn neq(self: *Self, num: T, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoNeqB(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoNeqH(@ptrCast(self.ptr.?), self.calcLen(), @bitCast(num), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoNeqF(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoNeqD(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                usize => {
                    try err.checkCuda(c.tomoNeqUZ(self.ptr.?, self.calcLen(), num, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn maskedFill(self: *Self, mask: *const Self, num: T, stream: *const Stream) !void {
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoMaskedFillB(@ptrCast(self.ptr.?), @ptrCast(mask.ptr), @bitCast(num), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoMaskedFillH(@ptrCast(self.ptr.?), @ptrCast(mask.ptr), @bitCast(num), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoMaskedFillF(self.ptr.?, mask.ptr, num, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoMaskedFillD(self.ptr.?, mask.ptr, num, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn tril(self: *Self, fill: T, stream: *const Stream) !void {
            const rows = self.base.getRow();
            const cols = self.base.getCol();
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoTrilB(@ptrCast(self.ptr.?), rows, cols, @bitCast(fill), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoTrilH(@ptrCast(self.ptr.?), rows, cols, @bitCast(fill), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoTrilF(self.ptr.?, rows, cols, fill, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoTrilD(self.ptr.?, rows, cols, fill, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn triu(self: *Self, fill: T, stream: *const Stream) !void {
            const rows = self.base.getRow();
            const cols = self.base.getCol();
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoTriuB(@ptrCast(self.ptr.?), rows, cols, @bitCast(fill), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoTriuH(@ptrCast(self.ptr.?), rows, cols, @bitCast(fill), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoTriuF(self.ptr.?, rows, cols, fill, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoTriuD(self.ptr.?, rows, cols, fill, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn arange(self: *Self, start: T, step: T, stream: *const Stream) !void {
            const len = self.calcLen();
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoArangeB(@ptrCast(self.ptr.?), start, step, len, stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoArangeD(@ptrCast(self.ptr.?), start, step, len, stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoArangeF(self.ptr.?, start, step, len, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoArangeD(self.ptr.?, start, step, len, stream.stream));
                },
                usize => {
                    try err.checkCuda(c.tomoArangeUZ(self.ptr.?, start, step, len, stream.stream));
                },
                else => unreachable,
            }
        }
    };
}
