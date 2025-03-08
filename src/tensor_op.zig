const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const TensorOpReduction = @import("tensor_op_reduction.zig").TensorOpReduction;
const TensorOpMap = @import("tensor_op_map.zig").TensorOpMap;
const TensorAlgorithm = @import("tensor_algorithm.zig").TensorAlgorithm;
const TensorOpCast = @import("tensor_op_cast.zig").TensorOpCast;
const TensorOpBlas = @import("tensor_op_blas.zig").TensorOpBlas;
const TensorOpElemwise = @import("tensor_op_elemwise.zig").TensorOpElemwise;
const TensorOpBroadCast = @import("tensor_op_broadcast.zig").TensorOpBroadCast;
const TensorOpLinear = @import("tensor_op_linear.zig").TensorOpLinear;

pub fn TensorOp(comptime T: type) type {
    return struct {
        const Self = GPUTensor(T);

        pub usingnamespace TensorOpMap(T);
        pub usingnamespace TensorOpReduction(T);
        pub usingnamespace TensorAlgorithm(T);
        pub usingnamespace TensorOpCast(T);
        // pub usingnamespace TensorOpBlas(T);
        pub usingnamespace TensorOpElemwise(T);
        pub usingnamespace TensorOpBroadCast(T);
        pub usingnamespace TensorOpLinear(T);
    };
}
