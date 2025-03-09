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
    };
}
