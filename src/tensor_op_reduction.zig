const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const Bf16 = @import("bf16.zig").BF16;

pub fn TensorOpReduction(comptime T: type) type {
    return struct {
        const Self = GPUTensor(T);

        // pub fn sumReduce(
        //     self: *Self,
        //     stream: *const Stream,
        //     result: *T,
        // ) !void {
        //     switch (T) {
        //         Bf16 => {
        //             try err.checkCuda(c.tomoSumReduceB(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f16 => {
        //             try err.checkCuda(c.tomoSumReduceH(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f32 => {
        //             try err.checkCuda(c.tomoSumReduceF(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         f64 => {
        //             try err.checkCuda(c.tomoSumReduceD(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         else => unreachable,
        //     }
        // }

        // pub fn mean(
        //     self: *Self,
        //     stream: *const Stream,
        //     result: *T,
        // ) !void {
        //     switch (T) {
        //         Bf16 => {
        //             try err.checkCuda(c.tomoMeanB(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f16 => {
        //             try err.checkCuda(c.tomoMeanH(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f32 => {
        //             try err.checkCuda(c.tomoMeanF(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         f64 => {
        //             try err.checkCuda(c.tomoMeanD(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         else => unreachable,
        //     }
        // }

        // pub fn min(
        //     self: *Self,
        //     stream: *const Stream,
        //     result: *T,
        // ) !void {
        //     switch (T) {
        //         Bf16 => {
        //             try err.checkCuda(c.tomoMinB(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f16 => {
        //             try err.checkCuda(c.tomoMinH(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f32 => {
        //             try err.checkCuda(c.tomoMinF(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         f64 => {
        //             try err.checkCuda(c.tomoMinD(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         else => unreachable,
        //     }
        // }

        // pub fn max(
        //     self: *Self,
        //     stream: *const Stream,
        //     result: *T,
        // ) !void {
        //     switch (T) {
        //         Bf16 => {
        //             try err.checkCuda(c.tomoMaxB(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f16 => {
        //             try err.checkCuda(c.tomoMaxH(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f32 => {
        //             try err.checkCuda(c.tomoMaxF(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         f64 => {
        //             try err.checkCuda(c.tomoMaxD(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         else => unreachable,
        //     }
        // }

        // pub fn l1Norm(
        //     self: *Self,
        //     stream: *const Stream,
        //     result: *T,
        // ) !void {
        //     switch (T) {
        //         Bf16 => {
        //             try err.checkCuda(c.tomoL1NormB(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f16 => {
        //             try err.checkCuda(c.tomoL1NormH(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f32 => {
        //             try err.checkCuda(c.tomoL1NormF(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         f64 => {
        //             try err.checkCuda(c.tomoL1NormD(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         else => unreachable,
        //     }
        // }

        // pub fn l2Norm(
        //     self: *Self,
        //     stream: *const Stream,
        //     result: *T,
        // ) !void {
        //     switch (T) {
        //         Bf16 => {
        //             try err.checkCuda(c.tomoL2NormB(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f16 => {
        //             try err.checkCuda(c.tomoL2NormH(@ptrCast(self.ptr.?), self.calcLen(), @ptrCast(result), stream.stream));
        //         },
        //         f32 => {
        //             try err.checkCuda(c.tomoL2NormF(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         f64 => {
        //             try err.checkCuda(c.tomoL2NormD(self.ptr.?, self.calcLen(), result, stream.stream));
        //         },
        //         else => unreachable,
        //     }
        // }
    };
}
