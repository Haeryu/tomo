const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;

pub fn TensorAlgorithm(comptime T: type, comptime rank: comptime_int) type {
    return struct {
        const Self = GPUTensor(T, rank);

        pub fn fill(self: *Self, val: T, stream: *const Stream) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoFillF(self.ptr, self.getLen(), val, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoFillD(self.ptr, self.getLen(), val, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn sortDesc(self: *Self, stream: *const Stream) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoSortDescF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSortDescD(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn sortAsc(self: *Self, stream: *const Stream) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoSortAscF(self.ptr, self.getLen(), stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSortAscF(self.ptr, self.getLen(), stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn find(self: *const Self, val: T, stream: *const Stream, i: *usize) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoFindF(self.ptr, self.getLen(), val, stream, i));
                },
                f64 => {
                    try err.checkCuda(c.tomoFindD(self.ptr, self.getLen(), val, stream, i));
                },
                else => unreachable,
            }
        }
    };
}
