const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const Bf16 = @import("bf16.zig").BF16;

pub fn TensorOpElemwise(comptime T: type) type {
    return struct {
        const Self = GPUTensor(T);

        pub fn add(self: *Self, other: *const Self, stream: *const Stream) !void {
            std.debug.assert(std.mem.eql(usize, self.base.getShapeConst(), other.base.getShapeConst()));
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoAddB(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoAddH(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoAddF(self.ptr.?, other.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoAddD(self.ptr.?, other.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn sub(self: *Self, other: *const Self, stream: *const Stream) !void {
            std.debug.assert(std.mem.eql(usize, self.base.getShapeConst(), other.base.getShapeConst()));
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoSubB(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoSubH(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoSubF(self.ptr.?, other.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSubD(self.ptr.?, other.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn product(self: *Self, other: *const Self, stream: *const Stream) !void {
            std.debug.assert(std.mem.eql(usize, self.base.getShapeConst(), other.base.getShapeConst()));
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoProductB(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoProductH(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoProductF(self.ptr.?, other.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoProductD(self.ptr.?, other.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn divide(self: *Self, other: *const Self, stream: *const Stream) !void {
            std.debug.assert(std.mem.eql(usize, self.base.getShapeConst(), other.base.getShapeConst()));
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoDivideB(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoDivideH(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoDivideF(self.ptr.?, other.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoDivideD(self.ptr.?, other.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn equal(self: *Self, other: *const Self, stream: *const Stream) !void {
            std.debug.assert(std.mem.eql(usize, self.base.getShapeConst(), other.base.getShapeConst()));
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoEqualB(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoEqualH(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoEqualF(self.ptr.?, other.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoEqualD(self.ptr.?, other.ptr.?, self.calcLen(), stream.stream));
                },
                usize => {
                    try err.checkCuda(c.tomoEqualUz(self.ptr.?, other.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn equalApprox(self: *Self, other: *const Self, eps: T, stream: *const Stream) !void {
            std.debug.assert(std.mem.eql(usize, self.base.getShapeConst(), other.base.getShapeConst()));
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoEqualApproxB(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), eps, stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoEqualApproxH(@ptrCast(self.ptr.?), @ptrCast(other.ptr.?), self.calcLen(), eps, stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoEqualApproxF(self.ptr.?, other.ptr.?, self.calcLen(), eps, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoEqualApproxD(self.ptr.?, other.ptr.?, self.calcLen(), eps, stream.stream));
                },

                else => unreachable,
            }
        }

        pub fn reluBackward(self: *Self, x: *const Self, stream: *const Stream) !void {
            std.debug.assert(std.mem.eql(usize, self.base.getShapeConst(), x.base.getShapeConst()));
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoReluBackwardB(@ptrCast(x.ptr.?), @ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoReluBackwardH(@ptrCast(x.ptr.?), @ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoReluBackwardF(x.ptr.?, self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoReluBackwardD(x.ptr.?, self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn leakyReluBackward(self: *Self, x: *const Self, stream: *const Stream) !void {
            std.debug.assert(std.mem.eql(usize, self.base.getShapeConst(), x.base.getShapeConst()));
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoLeakyReluBackwardB(@ptrCast(x.ptr.?), @ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoLeakyReluBackwardH(@ptrCast(x.ptr.?), @ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoLeakyReluBackwardF(x.ptr.?, self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoLeakyReluBackwardD(x.ptr.?, self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn geluBackward(self: *Self, x: *const Self, stream: *const Stream) !void {
            std.debug.assert(std.mem.eql(usize, self.base.getShapeConst(), x.base.getShapeConst()));
            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoGeluBackwardB(@ptrCast(x.ptr.?), @ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoGeluBackwardH(@ptrCast(x.ptr.?), @ptrCast(self.ptr.?), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoGeluBackwardF(x.ptr.?, self.ptr.?, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoGeluBackwardD(x.ptr.?, self.ptr.?, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }
        }
    };
}
