const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const Bf16 = @import("bf16.zig").BF16;

pub fn TensorOpBlas(comptime T: type, comptime rank: comptime_int) type {
    return struct {
        const Self = GPUTensor(T, rank);

        pub fn axpy(
            self: *const Self,
            alpha: T,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null); // initialized and synced

            if (self.base.countElem() != result.base.countElem()) {
                return error.SizeMismatch;
            }

            const n = self.base.countElem();

            const incx: i64 = 1;
            const incy: i64 = 1;

            try err.checkCublas(c.cublasSetStream(cuda_context.cublas_handle, stream.stream));

            switch (T) {
                f32 => {
                    try err.checkCublas(c.cublasSaxpy_64(
                        cuda_context.cublas_handle,
                        @intCast(n),
                        &alpha,
                        @ptrCast(self.ptr.?),
                        incx,
                        @ptrCast(result.ptr.?),
                        incy,
                    ));
                },
                f64 => {
                    try err.checkCublas(c.cublasDaxpy_64(
                        cuda_context.cublas_handle,
                        @intCast(n),
                        &alpha,
                        @ptrCast(self.ptr.?),
                        incx,
                        @ptrCast(result.ptr.?),
                        incy,
                    ));
                },
                else => unreachable,
            }
        }

        pub fn dgmm(
            self: *const Self,
            side: c.cublasSideMode_t,
            x: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null);
            const n = self.base.countElem();

            try err.checkCublas(c.cublasSetStream(cuda_context.cublas_handle, stream.stream));

            switch (T) {
                f32 => {
                    try err.checkCublas(c.cublasSdgmm_64(
                        cuda_context.cublas_handle,
                        side,
                        1,
                        @intCast(n),
                        @ptrCast(result.ptr.?),
                        1,
                        @ptrCast(x.ptr.?),
                        1,
                        @ptrCast(result.ptr.?),
                        1,
                    ));
                },
                f64 => {
                    try err.checkCublas(c.cublasDdgmm_64(
                        cuda_context.cublas_handle,
                        side,
                        1,
                        @intCast(n),
                        @ptrCast(result.ptr.?),
                        1,
                        @ptrCast(x.ptr.?),
                        1,
                        @ptrCast(result.ptr.?),
                        1,
                    ));
                },

                else => unreachable,
            }
        }

        pub fn add(
            self: *const Self,
            other: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null); // initialized and synced
            try result.writeAsync(other.ptr.?, other.calcLen(), 0, stream);
            try self.axpy(1, cuda_context.cublas_handle, stream, result);
        }

        pub fn addAssign(
            self: *Self,
            other: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            try other.axpy(1, cuda_context.cublas_handle, stream, self);
        }

        pub fn sub(
            self: *const Self,
            other: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null); // initialized and synced\
            try result.writeAsync(other.ptr.?, other.calcLen(), 0, stream);
            try result.axpy(-1.0, cuda_context.cublas_handle, stream, self);
        }

        pub fn subAssign(
            self: *Self,
            other: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            try other.axpy(-1.0, cuda_context.cublas_handle, stream, self);
        }

        pub fn mul(
            self: *const Self,
            other: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            result: *Self,
        ) !void {
            std.debug.assert(result.ptr != null); // initialized and synced
            try result.writeAsync(other.ptr.?, other.calcLen(), 0, stream);
            try self.dgmm(
                c.CUBLAS_SIDE_RIGHT,
                other,
                cuda_context,
                stream,
                result,
            );
        }

        pub fn mulAssign(
            self: *Self,
            other: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !void {
            try other.dgmm(
                c.CUBLAS_SIDE_RIGHT,
                self,
                cuda_context,
                stream,
                self,
            );
        }

        // TODO: make function to create and set cublaslt matrix layout to use it at cublasLtMatrixTransform
        // TODO: support many activation function => make class that control CUBLASLT_MATMUL_DESC_EPILOGUE_XXX
        // TODO: make small piece (policy) -> fuse

        pub fn createCublasLtMatrixLayout(self: *const Self) !c.cublasLtMatrixLayout_t {
            const batch_count: c_int = if (rank == 2) 0 else @intCast(self.base.shape[0]);
            const row: u64 = self.base.shape[self.base.shape.len - 2];
            const col: u64 = self.base.shape[self.base.shape.len - 1];
            const stride: u64 = row * col;

            var layout: c.cublasLtMatrixLayout_t = null;
            try err.checkCublas(
                c.cublasLtMatrixLayoutCreate(&layout, @intCast(Self.getCudaDatatype()), col, row, @intCast(col)),
            );
            errdefer _ = c.cublasLtMatrixLayoutDestroy(layout);

            const cuda_data_type: u32 = @intCast(Self.getCudaDatatype());
            try err.checkCublas(c.cublasLtMatrixLayoutSetAttribute(
                layout,
                c.CUBLASLT_MATRIX_LAYOUT_TYPE,
                &cuda_data_type,
                @sizeOf(@TypeOf(cuda_data_type)),
            ));

            if (batch_count != 0) {
                try err.checkCublas(c.cublasLtMatrixLayoutSetAttribute(
                    layout,
                    c.CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                    &batch_count,
                    @sizeOf(@TypeOf(batch_count)),
                ));

                try err.checkCublas(c.cublasLtMatrixLayoutSetAttribute(
                    layout,
                    c.CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    &stride,
                    @sizeOf(@TypeOf(stride)),
                ));
            }

            return layout;
        }

        fn TypeToCublasComputeType(comptime F: type) c.cublasComputeType_t {
            return switch (F) {
                f16 => c.CUBLAS_COMPUTE_16F,
                f32 => c.CUBLAS_COMPUTE_32F,
                f64 => c.CUBLAS_COMPUTE_64F,
                else => unreachable,
            };
        }

        fn TypeToCudaDataType(comptime F: type) c.cudaDataType_t {
            return switch (F) {
                f16 => c.CUDA_R_16F,
                f32 => c.CUDA_R_32F,
                f64 => c.CUDA_R_64F,
                else => unreachable,
            };
        }

        pub fn matmulTransposed(
            self: *const Self,
            self_transpose: bool,
            other: anytype,
            other_transpose: bool,
            add_tensor: anytype,
            add_transpose: bool,
            comptime CublasComputeType: type,
            alpha: if (CublasComputeType == f16) f16 else f32,
            beta: if (CublasComputeType == f16) f16 else f32,
            comptime EpilogueT: type,
            epilogue_config: EpilogueT.Config,
            //cublas_compute_type: c.cublasComputeType_t,
            stream: *const Stream,
            cuda_context: *const CudaContext,
            out: anytype,
        ) !void {
            comptime std.debug.assert(rank == 2 or rank == 3);

            var matmul_desc: c.cublasLtMatmulDesc_t = null;
            try err.checkCublas(c.cublasLtMatmulDescCreate(
                &matmul_desc,
                TypeToCublasComputeType(CublasComputeType),
                c.CUDA_R_32F,
            ));
            defer _ = c.cublasLtMatmulDescDestroy(matmul_desc);

            const op_no_t: c.cublasOperation_t = c.CUBLAS_OP_N;
            const op_t: c.cublasOperation_t = c.CUBLAS_OP_T;
            try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                matmul_desc,
                c.CUBLASLT_MATMUL_DESC_TRANSA,
                if (self_transpose) &op_no_t else &op_t,
                @sizeOf(c.cublasOperation_t),
            ));
            try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                matmul_desc,
                c.CUBLASLT_MATMUL_DESC_TRANSB,
                if (other_transpose) &op_no_t else &op_t,
                @sizeOf(c.cublasOperation_t),
            ));
            try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                matmul_desc,
                c.CUBLASLT_MATMUL_DESC_TRANSC,
                if (add_transpose) &op_no_t else &op_t,
                @sizeOf(c.cublasOperation_t),
            ));

            const self_layout = try self.createCublasLtMatrixLayout();
            defer _ = c.cublasLtMatrixLayoutDestroy(self_layout);

            const other_layout = try other.createCublasLtMatrixLayout();
            defer _ = c.cublasLtMatrixLayoutDestroy(other_layout);

            const add_layout = try add_tensor.createCublasLtMatrixLayout();
            defer _ = c.cublasLtMatrixLayoutDestroy(add_layout);

            const out_layout = try out.createCublasLtMatrixLayout();
            defer _ = c.cublasLtMatrixLayoutDestroy(out_layout);

            var preference: c.cublasLtMatmulPreference_t = null;
            try err.checkCublas(c.cublasLtMatmulPreferenceCreate(&preference));
            defer _ = c.cublasLtMatmulPreferenceDestroy(preference);

            try err.checkCublas(c.cublasLtMatmulPreferenceSetAttribute(
                preference,
                c.CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &cuda_context.cublaslt_workspace_size,
                @sizeOf(@TypeOf(cuda_context.cublaslt_workspace_size)),
            ));

            try EpilogueT.setEpilogue(matmul_desc, epilogue_config);

            const scale_type: c.cudaDataType_t = if (CublasComputeType == f16) c.CUDA_R_16F else c.CUDA_R_32F;
            try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                matmul_desc,
                c.CUBLASLT_MATMUL_DESC_SCALE_TYPE,
                &scale_type,
                @sizeOf(@TypeOf(scale_type)),
            ));

            var return_algo_count: c_int = 0;
            var heuristic: c.cublasLtMatmulHeuristicResult_t = .{};
            try err.checkCublas(c.cublasLtMatmulAlgoGetHeuristic(
                cuda_context.cublaslt_handle,
                matmul_desc,
                self_layout,
                other_layout,
                add_layout,
                out_layout,
                preference,
                1,
                &heuristic,
                &return_algo_count,
            ));

            if (return_algo_count == 0) {
                return err.CublasError.INTERNAL_ERROR;
            }

            try err.checkCublas(c.cublasLtMatmul(
                cuda_context.cublaslt_handle,
                matmul_desc,
                &alpha,
                self.ptr,
                self_layout,
                other.ptr,
                other_layout,
                &beta,
                add_tensor.ptr,
                add_layout,
                out.ptr,
                out_layout,
                &heuristic.algo,
                cuda_context.cublaslt_workspace,
                cuda_context.cublaslt_workspace_size,
                stream.stream,
            ));
        }

        pub fn tranformTransposed(
            self: *const Self,
            self_transpose: bool,
            other: anytype,
            other_transpose: bool,
            comptime CublasScaleType: type,
            alpha: CublasScaleType,
            beta: CublasScaleType,
            cuda_context: *const CudaContext,
            stream: *const Stream,
            out: anytype,
        ) !void {
            var transform_desc: c.cublasLtMatrixTransformDesc_t = null;
            try err.checkCublas(c.cublasLtMatrixTransformDescCreate(
                &transform_desc,
                TypeToCudaDataType(CublasScaleType),
            ));
            defer _ = c.cublasLtMatrixTransformDescDestroy(transform_desc);

            const op_no_t: c.cublasOperation_t = c.CUBLAS_OP_N;
            const op_t: c.cublasOperation_t = c.CUBLAS_OP_T;
            try err.checkCublas(c.cublasLtMatrixTransformDescSetAttribute(
                transform_desc,
                c.CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
                // row major -> already transposed
                if (self_transpose) &op_no_t else &op_t,
                @sizeOf(c.cublasOperation_t),
            ));
            try err.checkCublas(c.cublasLtMatrixTransformDescSetAttribute(
                transform_desc,
                c.CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB,
                // row major -> already transposed
                if (other_transpose) &op_no_t else &op_t,
                @sizeOf(c.cublasOperation_t),
            ));

            const self_layout = try self.createCublasLtMatrixLayout();
            defer _ = c.cublasLtMatrixLayoutDestroy(self_layout);

            const other_layout = try other.createCublasLtMatrixLayout();
            defer _ = c.cublasLtMatrixLayoutDestroy(other_layout);

            const out_layout = try out.createCublasLtMatrixLayout();
            defer _ = c.cublasLtMatrixLayoutDestroy(out_layout);

            // already set
            // const scale_type: c.cudaDataType_t = TypeToCudaDataType(CublasScaleType);
            // try err.checkCublas(c.cublasLtMatrixTransformDescSetAttribute(
            //     transform_desc,
            //     c.CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
            //     // row major -> already transposed
            //     &scale_type,
            //     @sizeOf(scale_type),
            // ));

            try err.checkCublas(c.cublasLtMatrixTransform(
                cuda_context.cublaslt_handle,
                transform_desc,
                &alpha,
                self.ptr,
                self_layout,
                &beta,
                other.ptr,
                other_layout,
                out.ptr,
                out_layout,
                stream.stream,
            ));
        }
    };
}
