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

        pub fn linear(self: *const Self, other: *const Self, bias: ?*const Self, stream: *const Stream) !void {
            std.debug.assert(self.base.getCol() == other.base.getRow());
            if (bias) |b| {
                std.debug.assert(b.base.getRow() == self.base.getRow());
                std.debug.assert(b.base.getCol() == other.base.getCol());
            }
            var res = try Self.initAsync(.{ self.base.getRow(), other.base.getCol() }, stream);
            errdefer res.deinitAsync(stream);

            switch (T) {
                Bf16 => {
                    try err.checkCuda(c.tomoLinearB(
                        @ptrCast(self.ptr),
                        @ptrCast(other.ptr),
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
                        @ptrCast(self.ptr),
                        @ptrCast(other.ptr),
                        self.base.getRow(),
                        self.base.getCol(),
                        other.base.getCol(),
                        if (bias) |b| @ptrCast(b.ptr) else null,
                        @ptrCast(res.ptr),
                        stream.stream,
                    ));
                },
                f32 => {
                    try err.checkCuda(c.tomoLinearH(
                        self.ptr,
                        other.ptr,
                        self.base.getRow(),
                        self.base.getCol(),
                        other.base.getCol(),
                        if (bias) |b| b.ptr else null,
                        res.ptr,
                        stream.stream,
                    ));
                },
                f64 => {
                    try err.checkCuda(c.tomoLinearH(
                        self.ptr,
                        other.ptr,
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
    };
}
