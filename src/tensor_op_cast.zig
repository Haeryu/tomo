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
                            try err.checkCuda(c.tomoBtoH(@ptrCast(self.ptr), self.getLen(), stream.stream, @ptrCast(res.ptr)));
                        },
                        f32 => {
                            try err.checkCuda(c.tomoBtoF(@ptrCast(self.ptr), self.getLen(), stream.stream, res.ptr));
                        },
                        f64 => {
                            try err.checkCuda(c.tomoBtoD(@ptrCast(self.ptr), self.getLen(), stream.stream, res.ptr));
                        },
                        else => unreachable,
                    }
                },
                f16 => {
                    switch (U) {
                        BF16 => {
                            try err.checkCuda(c.tomoHtoB(@ptrCast(self.ptr), self.getLen(), stream.stream, @ptrCast(res.ptr)));
                        },
                        f32 => {
                            try err.checkCuda(c.tomoHtoF(@ptrCast(self.ptr), self.getLen(), stream.stream, res.ptr));
                        },
                        f64 => {
                            try err.checkCuda(c.tomoHtoD(@ptrCast(self.ptr), self.getLen(), stream.stream, res.ptr));
                        },
                        else => unreachable,
                    }
                },
                f32 => {
                    switch (U) {
                        BF16 => {
                            try err.checkCuda(c.tomoFtoB(self.ptr, self.getLen(), stream.stream, @ptrCast(res.ptr)));
                        },
                        f16 => {
                            try err.checkCuda(c.tomoFtoH(self.ptr, self.getLen(), stream.stream, @ptrCast(res.ptr)));
                        },
                        f64 => {
                            try err.checkCuda(c.tomoFtoD(self.ptr, self.getLen(), stream.stream, res.ptr));
                        },
                        else => unreachable,
                    }
                },
                f64 => {
                    switch (U) {
                        BF16 => {
                            try err.checkCuda(c.tomoDtoB(self.ptr, self.getLen(), stream.stream, @ptrCast(res.ptr)));
                        },
                        f16 => {
                            try err.checkCuda(c.tomoDtoH(self.ptr, self.getLen(), stream.stream, @ptrCast(res.ptr)));
                        },
                        f32 => {
                            try err.checkCuda(c.tomoDtoF(self.ptr, self.getLen(), stream.stream, res.ptr));
                        },
                        else => unreachable,
                    }
                },
                else => unreachable,
            }
        }
    };
}
