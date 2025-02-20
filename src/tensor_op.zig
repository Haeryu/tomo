const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const TensorOpReduction = @import("tensor_op_reduction.zig").TensorOpReduction;
const TensorOpElemwise = @import("tensor_op_elemwise.zig").TensorOpElemwise;
const TensorAlgorithm = @import("tensor_algorithm.zig").TensorAlgorithm;

pub fn TensorOp(comptime T: type, comptime rank: comptime_int) type {
    return struct {
        const Self = GPUTensor(T, rank);

        pub usingnamespace TensorOpElemwise(T, rank);
        pub usingnamespace TensorOpReduction(T, rank);
        pub usingnamespace TensorAlgorithm(T, rank);
    };
}
