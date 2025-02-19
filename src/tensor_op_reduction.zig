const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;

pub fn TensorOpReduction(comptime T: type, comptime rank: comptime_int) type {
    return struct {
        const Self = GPUTensor(T, rank);

        pub fn sumReduce(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *T,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoSumReduceF(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoSumReduceD(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn mean(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *T,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoMeanF(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoMeanD(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn min(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *T,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoMinF(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoMinD(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                else => unreachable,
            }
        }

        // pub fn minIdx(
        //     self: *Self,
        //     cuda_context: *const CudaContext,
        //     stream: *const Stream,
        //     result: *T,
        //     result_idx: *usize,
        // ) !void {
        //     switch (T) {
        //         f32 => {
        //             try err.checkCuda(c.tomoArgMinF(self.ptr, self.getLen(), result, result_idx, cuda_context.threads_per_block, stream.stream));
        //         },
        //         f64 => {
        //             try err.checkCuda(c.tomoArgMinD(self.ptr, self.getLen(), result, result_idx, cuda_context.threads_per_block, stream.stream));
        //         },
        //         else => unreachable,
        //     }
        // }

        pub fn max(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *T,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoMaxF(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoMaxD(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                else => unreachable,
            }
        }

        // pub fn maxIdx(
        //     self: *Self,
        //     cuda_context: *const CudaContext,
        //     stream: *const Stream,
        //     result: *T,
        //     result_idx: *usize,
        // ) !void {
        //     switch (T) {
        //         f32 => {
        //             try err.checkCuda(c.tomoArgMaxF(self.ptr, self.getLen(), result, result_idx, cuda_context.threads_per_block, stream.stream));
        //         },
        //         f64 => {
        //             try err.checkCuda(c.tomoArgMaxD(self.ptr, self.getLen(), result, result_idx, cuda_context.threads_per_block, stream.stream));
        //         },
        //         else => unreachable,
        //     }
        // }

        pub fn l1Norm(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *T,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoL1NormF(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoL1NormD(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                else => unreachable,
            }
        }

        pub fn l2Norm(
            self: *Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *T,
        ) !void {
            switch (T) {
                f32 => {
                    try err.checkCuda(c.tomoL2NormF(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                f64 => {
                    try err.checkCuda(c.tomoL2NormD(self.ptr, self.getLen(), result, cuda_context.threads_per_block, stream.stream));
                },
                else => unreachable,
            }
        }
    };
}
