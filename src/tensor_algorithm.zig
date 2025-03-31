const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const BF16 = @import("bf16.zig").BF16;

const is_debugging = @import("builtin").mode == .Debug;

pub fn TensorAlgorithm(comptime T: type) type {
    return struct {
        const Self = GPUTensor(T);

        pub fn fill(self: *Self, val: if (T != BF16) T else f32, stream: *const Stream) !void {
            switch (T) {
                BF16 => {
                    try err.checkCuda(c.tomoFillB(@ptrCast(self.ptr), self.calcLen(), @bitCast(BF16.fromF32(val)), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoFillH(@ptrCast(self.ptr), self.calcLen(), @bitCast(val), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoFillF(self.ptr, self.calcLen(), val, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoFillD(self.ptr, self.calcLen(), val, stream.stream));
                },
                usize => {
                    try err.checkCuda(c.tomoFillUZ(self.ptr, self.calcLen(), val, stream.stream));
                },
                else => unreachable,
            }

            if (is_debugging and try self.hasNaN(stream)) {
                return error.HasNan;
            }
        }

        pub fn sortDesc(self: *Self, stream: *const Stream) !void {
            switch (T) {
                BF16 => {
                    try err.checkCuda(c.tomoSortDescB(@ptrCast(self.ptr), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoSortDescH(@ptrCast(self.ptr), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoSortDescF(self.ptr, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSortDescD(self.ptr, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }

            if (is_debugging and try self.hasNaN(stream)) {
                return error.HasNan;
            }
        }

        pub fn sortAsc(self: *Self, stream: *const Stream) !void {
            switch (T) {
                BF16 => {
                    try err.checkCuda(c.tomoSortAscB(@ptrCast(self.ptr), self.calcLen(), stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoSortAscH(@ptrCast(self.ptr), self.calcLen(), stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoSortAscF(self.ptr, self.calcLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSortAscD(self.ptr, self.calcLen(), stream.stream));
                },
                else => unreachable,
            }

            if (is_debugging and try self.hasNaN(stream)) {
                return error.HasNan;
            }
        }

        pub fn find(self: *const Self, val: T, stream: *const Stream, i: *usize) !void {
            switch (T) {
                BF16 => {
                    try err.checkCuda(c.tomoFindB(@ptrCast(self.ptr), self.calcLen(), @ptrCast(val), stream.stream, i));
                },
                f16 => {
                    try err.checkCuda(c.tomoFindH(@ptrCast(self.ptr), self.calcLen(), @ptrCast(val), stream.stream, i));
                },
                f32 => {
                    try err.checkCuda(c.tomoFindF(self.ptr, self.calcLen(), val, stream.stream, i));
                },
                f64 => {
                    try err.checkCuda(c.tomoFindD(self.ptr, self.calcLen(), val, stream.stream, i));
                },
                else => unreachable,
            }

            if (is_debugging and try self.hasNaN(stream)) {
                return error.HasNan;
            }
        }

        pub fn hasNaN(self: *const Self, stream: *const Stream) !bool {
            var has_nan: bool = false;
            switch (T) {
                BF16 => {
                    try err.checkCuda(c.tomoHasNaNB(@ptrCast(self.ptr), self.calcLen(), &has_nan, stream.stream));
                },
                f16 => {
                    try err.checkCuda(c.tomoHasNaNH(@ptrCast(self.ptr), self.calcLen(), &has_nan, stream.stream));
                },
                f32 => {
                    try err.checkCuda(c.tomoHasNaNF(self.ptr, self.calcLen(), &has_nan, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoHasNaND(self.ptr, self.calcLen(), &has_nan, stream.stream));
                },
                else => return false,
            }

            try stream.sync();

            return has_nan;
        }
    };
}
