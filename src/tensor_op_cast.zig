const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const BF16 = @import("bf16.zig").BF16;

pub fn TensorOpCast(comptime T: type, comptime rank: comptime_int) type {
    return struct {
        const Self = GPUTensor(T, rank);

        pub fn cast(self: *const Self, comptime U: type, stream: *const Stream, res: GPUTensor(U, rank)) !void {
            switch (T) {
                BF16 => {
                    switch (U) {
                        f16 => {
                            try c.tomoBtoH(@ptrCast(self.ptr), self.getLen(), stream.stream, @ptrCast(res.ptr));
                        },
                        f32 => {
                            try c.tomoBtoF(@ptrCast(self.ptr), self.getLen(), stream.stream, res.ptr);
                        },
                        f64 => {
                            try c.tomoBtoD(@ptrCast(self.ptr), self.getLen(), stream.stream, res.ptr);
                        },
                        else => unreachable,
                    }
                },
                f16 => {
                    switch (U) {
                        BF16 => {
                            try c.tomoHtoB(@ptrCast(self.ptr), self.getLen(), stream.stream, @ptrCast(res.ptr));
                        },
                        f32 => {
                            try c.tomoHtoF(@ptrCast(self.ptr), self.getLen(), stream.stream, res.ptr);
                        },
                        f64 => {
                            try c.tomoHtoD(@ptrCast(self.ptr), self.getLen(), stream.stream, res.ptr);
                        },
                        else => unreachable,
                    }
                },
                f32 => {
                    switch (U) {
                        BF16 => {
                            try c.tomoFtoB(self.ptr, self.getLen(), stream.stream, @ptrCast(res.ptr));
                        },
                        f16 => {
                            try c.tomoFtoH(self.ptr, self.getLen(), stream.stream, @ptrCast(res.ptr));
                        },
                        f64 => {
                            try c.tomoFtoD(self.ptr, self.getLen(), stream.stream, res.ptr);
                        },
                        else => unreachable,
                    }
                },
                f64 => {
                    switch (U) {
                        BF16 => {
                            try c.tomoDtoB(self.ptr, self.getLen(), stream.stream, @ptrCast(res.ptr));
                        },
                        f16 => {
                            try c.tomoDtoH(self.ptr, self.getLen(), stream.stream, @ptrCast(res.ptr));
                        },
                        f32 => {
                            try c.tomoDtoF(self.ptr, self.getLen(), stream.stream, res.ptr);
                        },
                        else => unreachable,
                    }
                },
                else => unreachable,
            }
        }
    };
}
