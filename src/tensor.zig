const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const TensorOp = @import("tensor_op.zig").TensorOp;
const TensorFillRandom = @import("tensor_fill_random.zig").TensorFillRandom;
const BF16 = @import("bf16.zig").BF16;

pub const matmul_epilogue = @import("tensor_op_matmul_epilogue.zig");

pub fn TensorBase(comptime rank: comptime_int) type {
    return struct {
        shape: [rank]usize,
        strides: [rank]usize,
        is_contiguous: bool,

        const Self = @This();

        pub fn init(shape: [rank]usize) Self {
            var strides: [rank]usize = .{0} ** rank;

            // Compute strides based on memory format
            strides[rank - 1] = 1;
            var i: usize = rank - 1;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = strides[i] * shape[i];
            }

            return .{
                .shape = shape,
                .strides = strides,
                .is_contiguous = true,
            };
        }

        pub fn countElem(self: *const Self) usize {
            // const vec: @Vector(rank, usize) = self.shape;
            // return @reduce(.Mul, vec);

            var total: usize = 1;
            for (self.shape) |dim| {
                total *= dim;
            }
            return total;
        }

        pub fn computeFanInOut(self: *Self) std.meta.Tuple(&.{ usize, usize }) {
            // e.g. for a 2D weight matrix [fan_out, fan_in], or
            // for convolution [out_channels, in_channels * kernel_h * kernel_w], etc.
            // This is up to you how you want to interpret 'shape'.
            if (self.shape.len == 2) {
                return .{ self.shape[1], self.shape[0] }; // fan_in, fan_out
            } else {
                // for a 4D conv: [out_channels, in_channels, kernel_h, kernel_w]
                const fan_in = self.shape[1] * self.shape[2] * self.shape[3];
                const fan_out = self.shape[0] * self.shape[2] * self.shape[3];
                return .{ fan_in, fan_out };
            }
        }
    };
}

pub fn CPUTensor(comptime T: type, comptime rank: comptime_int) type {
    return struct {
        base: Base,
        data: []T,

        const Self = @This();
        const Base = TensorBase(rank);

        pub fn getSize(self: *const Self) usize {
            return self.base.countElem() * @sizeOf(T);
        }

        pub fn init(allocator: std.mem.Allocator, shape: [rank]usize) !Self {
            const base = Base.init(shape);
            const data = try allocator.alloc(T, base.countElem());
            errdefer allocator.free(data);

            return .{
                .base = base,
                .data = data,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
        }

        pub fn write(self: *Self, data: []const T, offset: usize) void {
            @memcpy(self.data[offset .. offset + data.len], data);
        }

        pub fn writeFromDevice(
            self: *Self,
            ptr: [*]const T,
            len: usize,
            offset: usize,
            stream: *const Stream,
        ) !void {
            try err.checkCuda(c.cudaMemcpyAsync(
                @ptrCast(&self.data.ptr[offset]),
                @ptrCast(ptr),
                len * @sizeOf(T),
                c.cudaMemcpyDeviceToHost,
                stream.stream,
            ));
        }

        pub fn clone(self: *const Self, allocator: std.mem.Allocator) !Self {
            var cloned = try Self.init(allocator, self.shape);
            errdefer cloned.deinit(allocator);

            cloned.write(self.data, 0);

            return cloned;
        }

        pub fn at(self: *const Self, indices: [rank]usize) *T {
            var offset: usize = 0;
            for (indices, self.base.strides) |idx, stride| {
                offset += idx * stride;
            }
            return &self.data[offset];
        }

        fn computeContiguity(self: *Self) void {
            var expected_stride: usize = 1;
            self.is_contiguous = true;
            for (0..rank) |i| {
                const dim = self.shape[rank - 1 - i];
                if (self.strides[rank - 1 - i] != expected_stride) {
                    self.is_contiguous = false;
                    break;
                }
                expected_stride *= dim;
            }
        }

        pub fn transpose(self: *Self, perm: [rank]usize) !void {
            var new_shape: [rank]usize = undefined;
            var new_strides: [rank]usize = undefined;

            if (perm.len != rank) {
                return error.InvalidPermutation;
            }

            for (perm, 0..) |p, i| {
                new_shape[i] = self.shape[p];
                new_strides[i] = self.strides[p];
            }

            self.shape = new_shape;
            self.strides = new_strides;

            self.computeContiguity();
        }

        pub fn reshape(
            self: *Self,
            allocator: std.mem.Allocator,
            comptime new_rank: comptime_int,
            new_shape: [new_rank]usize,
        ) !CPUTensor(T, new_rank) {
            if (!self.base.is_contiguous) {
                return error.ReshapingNotContiguousTensor;
            }

            const old_size = self.base.countElem();
            var new_size: usize = 1;
            for (new_shape) |dim| {
                new_size *= dim;
            }

            if (old_size != new_size) return error.InvalidReshape;

            var new_tensor = try CPUTensor(T, new_rank).init(allocator, new_shape);
            errdefer new_tensor.deinit(allocator);

            new_tensor.write(self.data, 0);

            return new_tensor;
        }

        pub fn contiguousHost(self: *Self, allocator: std.mem.Allocator) !Self {
            if (self.is_contiguous) {
                return try self.clone(allocator);
            }

            var new_tensor = try Self.init(allocator, self.shape);
            errdefer new_tensor.deinit(allocator);

            // Iterate over all elements using multi-dimensional indexing
            var indices: [rank]usize = .{0} ** rank;

            while (true) {
                new_tensor.at(indices).* = self.at(indices).*;

                // Increment multi-dimensional indices
                var dim: usize = rank;
                while (dim > 0) {
                    dim -= 1;
                    indices[dim] += 1;
                    if (indices[dim] < self.shape[dim]) break;
                    indices[dim] = 0;
                    if (dim == 0) return new_tensor; // Exit when the last index resets
                }
            }
        }

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            var indices: [rank]usize = .{0} ** rank;
            try self.printRecursive(fmt, &indices, 0, 0, writer);
        }

        /// Print indentation (two spaces per indent level)
        fn printIndent(writer: anytype, indent: usize) !void {
            var i: usize = 0;
            while (i < indent) : (i += 1) {
                try writer.print("  ", .{});
            }
        }

        /// Recursively print the tensor
        fn printRecursive(
            self: *const Self,
            comptime fmt: []const u8,
            indices: *[rank]usize,
            depth: usize,
            indent: usize,
            writer: anytype,
        ) !void {
            if (depth == rank - 1) {
                // At the last dimension: print a row
                try printIndent(writer, indent);
                try writer.print("[ ", .{});
                var i: usize = 0;
                while (i < self.base.shape[depth]) : (i += 1) {
                    indices[depth] = i;
                    try writer.print("{" ++ fmt ++ "}, ", .{self.at(indices.*).*});
                }
                try writer.print("],\n", .{});
            } else {
                // Print an opening bracket for the current level
                try printIndent(writer, indent);
                try writer.print("[\n", .{});
                var i: usize = 0;
                while (i < self.base.shape[depth]) : (i += 1) {
                    indices[depth] = i;
                    try self.printRecursive(fmt, indices, depth + 1, indent + 1, writer);
                }
                // Print a closing bracket for the current level
                try printIndent(writer, indent);
                try writer.print("]\n", .{});
            }
        }

        pub fn toDevice(self: *const Self, stream: *const Stream) !GPUTensor(T, rank) {
            var out = try GPUTensor(T, rank).initAsync(self.base.shape, stream);
            errdefer out.deinitAsync(stream);

            try out.writeFromHostAsync(self.data, 0, stream);

            return out;
        }
    };
}

pub fn GPUTensor(comptime T: type, comptime rank: comptime_int) type {
    return struct {
        ptr: ?[*]T = null,
        base: Base = undefined,

        const Self = @This();
        const Base = TensorBase(rank);
        pub const Elem = T;

        pub fn calcLen(self: *const Self) usize {
            return self.base.countElem();
        }

        pub fn getSize(self: *const Self) usize {
            return self.base.countElem() * @sizeOf(T);
        }

        pub fn initSync(shape: [rank]usize) !Self {
            const base = Base.init(shape);
            var ptr: ?[*]T = null;
            try err.checkCuda(c.cudaMalloc(@ptrCast(&ptr), base.countElem() * @sizeOf(T)));
            return .{
                .ptr = ptr,
                .base = base,
            };
        }

        pub fn initAsync(shape: [rank]usize, stream: *const Stream) !Self {
            const base = Base.init(shape);
            var ptr: ?[*]T = null;
            try err.checkCuda(c.cudaMallocAsync(@ptrCast(&ptr), base.countElem() * @sizeOf(T), stream.stream));
            return .{
                .ptr = ptr,
                .base = base,
            };
        }

        pub fn deinitSync(self: *Self) void {
            if (self.ptr) |ptr| {
                _ = c.cudaFree(@ptrCast(ptr));
                self.ptr = null;
            }
        }

        pub fn deinitAsync(self: *Self, stream: *const Stream) void {
            if (self.ptr) |ptr| {
                _ = c.cudaFreeAsync(@ptrCast(ptr), stream.stream);
                self.ptr = null;
            }
        }

        pub fn writeAsync(self: *Self, device_ptr: [*]const T, len: usize, offset: usize, stream: *const Stream) !void {
            try err.checkCuda(c.cudaMemcpyAsync(
                @ptrCast(self.ptr.? + offset),
                @ptrCast(device_ptr),
                len * @sizeOf(T),
                c.cudaMemcpyDeviceToDevice,
                stream.stream,
            ));
        }

        pub fn writeFromHostAsync(self: *Self, data: []const T, offset: usize, stream: *const Stream) !void {
            try err.checkCuda(c.cudaMemcpyAsync(
                @ptrCast(self.ptr.? + offset),
                @ptrCast(data.ptr),
                data.len * @sizeOf(T),
                c.cudaMemcpyHostToDevice,
                stream.stream,
            ));
        }

        pub fn cloneAsync(self: *const Self, stream: *const Stream) !Self {
            //std.debug.assert(cloned.ptr == null);

            var cloned = try Self.initAsync(self.base.shape, stream);
            errdefer cloned.deinitAsync(stream);

            try cloned.writeAsync(self.ptr.?, self.calcLen(), 0, stream);

            return cloned;
        }

        pub fn writeSync(self: *Self, device_ptr: [*]T, len: usize, offset: usize) !void {
            try err.checkCuda(c.cudaMemcpy(
                @ptrCast(self.ptr.? + offset),
                @ptrCast(device_ptr),
                len * @sizeOf(T),
                c.cudaMemcpyDeviceToDevice,
            ));
        }

        pub fn writeFromHostSync(self: *Self, data: []T, offset: usize) !void {
            try err.checkCuda(c.cudaMemcpy(
                @ptrCast(self.ptr.? + offset),
                @ptrCast(data.ptr),
                data.len * @sizeOf(T),
                c.cudaMemcpyHostToDevice,
            ));
        }

        pub fn cloneSync(self: *const Self) !Self {
            var cloned = try Self.initSync(self.shape);
            errdefer cloned.deinitSync();

            try cloned.writeSync(self.ptr, self.calcLen(), 0);

            return cloned;
        }

        pub fn reshape(
            self: *const Self,
            comptime new_rank: comptime_int,
            new_tensor: *GPUTensor(T, new_rank),
            new_shape: [new_rank]usize,
            stream: *const Stream,
        ) !void {
            std.debug.assert(new_tensor.ptr == null);

            const old_size = self.base.countElem();
            var new_size: usize = 1;
            for (new_shape) |dim| {
                new_size *= dim;
            }

            if (old_size != new_size) return error.InvalidReshape;

            try new_tensor.initAsync(new_shape, stream);
            errdefer new_tensor.deinitAsync(stream);

            try new_tensor.writeAsync(self.ptr, self.calcLen(), 0, stream);
        }

        // TODO
        // pub fn contiguousDevice(self: *Self, stream: *const Stream) !Self {}

        pub fn getCudaDatatype() c_int {
            switch (T) {
                BF16 => return c.CUDA_R_16BF,
                f16 => return c.CUDA_R_16F,
                f32 => return c.CUDA_R_32F,
                f64 => return c.CUDA_R_64F,
                i8 => return c.CUDA_R_8I,
                u8 => return c.CUDA_R_8U,
                i16 => return c.CUDA_R_16I,
                u16 => return c.CUDA_R_16U,
                i32 => return c.CUDA_R_32I,
                u32 => return c.CUDA_R_32U,
                i64 => return c.CUDA_R_64I,
                u64 => return c.CUDA_R_64U,
                else => unreachable,
            }
        }

        pub fn toHost(self: *const Self, allocator: std.mem.Allocator, stream: *const Stream) !CPUTensor(T, rank) {
            var host_tensor = try CPUTensor(T, rank).init(allocator, self.base.shape);
            errdefer host_tensor.deinit(allocator);

            try host_tensor.writeFromDevice(self.ptr.?, self.calcLen(), 0, stream);

            return host_tensor;
        }

        pub usingnamespace TensorOp(T, rank);
        pub usingnamespace TensorFillRandom(T, rank);
    };
}
