const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");
const Stream = @import("stream.zig").Stream;
const CudaContext = @import("cuda_context.zig").CudaContext;
const GPUTensor = @import("tensor.zig").GPUTensor;
const Bf16 = @import("bf16.zig").BF16;

pub fn TensorOpBlas(comptime T: type) type {
    return struct {
        const Self = GPUTensor(T);

        // pub fn axpy(
        //     self: *const Self,
        //     alpha: T,
        //     cuda_context: *const CudaContext,
        //     stream: *const Stream,
        //     result: *Self,
        // ) !void {
        //     std.debug.assert(result.ptr != null); // initialized and synced

        //     if (self.base.countElem() != result.base.countElem()) {
        //         return error.SizeMismatch;
        //     }

        //     const n = self.base.countElem();

        //     const incx: i64 = 1;
        //     const incy: i64 = 1;

        //     try err.checkCublas(c.cublasSetStream(cuda_context.cublas_handle, stream.stream));

        //     switch (T) {
        //         f32 => {
        //             try err.checkCublas(c.cublasSaxpy_64(
        //                 cuda_context.cublas_handle,
        //                 @intCast(n),
        //                 &alpha,
        //                 @ptrCast(self.ptr.?),
        //                 incx,
        //                 @ptrCast(result.ptr.?),
        //                 incy,
        //             ));
        //         },
        //         f64 => {
        //             try err.checkCublas(c.cublasDaxpy_64(
        //                 cuda_context.cublas_handle,
        //                 @intCast(n),
        //                 &alpha,
        //                 @ptrCast(self.ptr.?),
        //                 incx,
        //                 @ptrCast(result.ptr.?),
        //                 incy,
        //             ));
        //         },
        //         else => unreachable,
        //     }
        // }

        // pub fn dgmm(
        //     self: *const Self,
        //     side: c.cublasSideMode_t,
        //     x: *const Self,
        //     cuda_context: *const CudaContext,
        //     stream: *const Stream,
        //     result: *Self,
        // ) !void {
        //     std.debug.assert(result.ptr != null);
        //     const n = self.base.countElem();

        //     try err.checkCublas(c.cublasSetStream(cuda_context.cublas_handle, stream.stream));

        //     switch (T) {
        //         f32 => {
        //             try err.checkCublas(c.cublasSdgmm_64(
        //                 cuda_context.cublas_handle,
        //                 side,
        //                 1,
        //                 @intCast(n),
        //                 @ptrCast(result.ptr.?),
        //                 1,
        //                 @ptrCast(x.ptr.?),
        //                 1,
        //                 @ptrCast(result.ptr.?),
        //                 1,
        //             ));
        //         },
        //         f64 => {
        //             try err.checkCublas(c.cublasDdgmm_64(
        //                 cuda_context.cublas_handle,
        //                 side,
        //                 1,
        //                 @intCast(n),
        //                 @ptrCast(result.ptr.?),
        //                 1,
        //                 @ptrCast(x.ptr.?),
        //                 1,
        //                 @ptrCast(result.ptr.?),
        //                 1,
        //             ));
        //         },

        //         else => unreachable,
        //     }
        // }

        pub fn add(
            self: *const Self,
            other: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !Self {
            return try self.tranform(
                false,
                other,
                false,
                T,
                if (T == Bf16) Bf16.fromF32(1.0) else 1.0,
                if (T == Bf16) Bf16.fromF32(1.0) else 1.0,
                cuda_context,
                stream,
            );
        }

        pub fn sub(
            self: *const Self,
            other: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !Self {
            return try self.tranform(
                false,
                other,
                false,
                T,
                if (T == Bf16) Bf16.fromF32(1.0) else 1.0,
                if (T == Bf16) Bf16.fromF32(-1.0) else -1.0,
                cuda_context,
                stream,
            );
        }

        pub fn scale(
            self: *const Self,
            factor: T,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !Self {
            return try self.tranform(
                false,
                null,
                false,
                T,
                factor,
                if (T == Bf16) Bf16.fromF32(0.0) else 0.0,
                cuda_context,
                stream,
            );
        }

        pub fn createCublasLtMatrixLayout(self: *const Self) !c.cublasLtMatrixLayout_t {
            const batch_count: c_int = @intCast(self.base.getBatch());
            const row: u64 = self.base.getRow();
            const col: u64 = self.base.getCol();
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

        // fn TypeToCublasComputeType(comptime F: type) c.cublasComputeType_t {
        //     return switch (F) {
        //         f16 => c.CUBLAS_COMPUTE_16F,
        //         f32 => c.CUBLAS_COMPUTE_32F,
        //         f64 => c.CUBLAS_COMPUTE_64F,
        //         else => unreachable,
        //     };
        // }

        fn TypeToCudaDataType(comptime F: type) c.cudaDataType_t {
            return switch (F) {
                Bf16 => c.CUDA_R_16BF,
                f16 => c.CUDA_R_16F,
                f32 => c.CUDA_R_32F,
                f64 => c.CUDA_R_64F,
                else => unreachable,
            };
        }

        pub fn matmulTransposed(
            self: *const Self,
            self_transpose: bool,
            other_tensor: anytype,
            other_transpose: bool,
            add_tensor: anytype,
            add_transpose: bool,
            comptime cublas_compute_type: c.cublasComputeType_t,
            alpha: if (cublas_compute_type == c.CUBLAS_COMPUTE_16F) f16 else f32,
            beta: if (cublas_compute_type == c.CUBLAS_COMPUTE_16F) f16 else f32,
            comptime EpilogueT: type,
            epilogue_config: EpilogueT.Config,
            //cublas_compute_type: c.cublasComputeType_t,
            stream: *const Stream,
            cuda_context: *const CudaContext,
            comptime OutType: type,
        ) !GPUTensor(OutType) {
            if (@TypeOf(other_tensor) == @TypeOf(null)) {
                if (T == Bf16) {
                    std.debug.assert(beta.val.x == Bf16.fromF32(0.0).val.x);
                } else {
                    std.debug.assert(beta == 0.0);
                }
            }

            var out_tensor = try GPUTensor(OutType).initAsync(if (self.base.rank == 3) &.{
                self.base.getBatch(),
                other_tensor.base.getCol(),
                self.base.getRow(),
            } else &.{
                other_tensor.base.getCol(),
                self.base.getRow(),
            }, stream);
            errdefer out_tensor.deinitAsync(stream);

            var matmul_desc: c.cublasLtMatmulDesc_t = null;
            try err.checkCublas(c.cublasLtMatmulDescCreate(
                &matmul_desc,
                cublas_compute_type,
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

            if (@TypeOf(add_tensor) != @TypeOf(null)) {
                try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    c.CUBLASLT_MATMUL_DESC_TRANSC,
                    if (add_transpose) &op_no_t else &op_t,
                    @sizeOf(c.cublasOperation_t),
                ));
            }

            const self_layout = try self.createCublasLtMatrixLayout();
            defer _ = c.cublasLtMatrixLayoutDestroy(self_layout);

            const other_layout = try other_tensor.createCublasLtMatrixLayout();
            defer _ = c.cublasLtMatrixLayoutDestroy(other_layout);

            const out_layout = try out_tensor.createCublasLtMatrixLayout();
            defer _ = c.cublasLtMatrixLayoutDestroy(out_layout);

            const add_layout = if (@TypeOf(add_tensor) != @TypeOf(null)) try add_tensor.createCublasLtMatrixLayout() else out_layout;
            defer {
                if (@TypeOf(add_tensor) != @TypeOf(null)) {
                    _ = c.cublasLtMatrixLayoutDestroy(add_layout);
                }
            }

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

            const scale_type: c.cudaDataType_t = if (cublas_compute_type == c.CUBLAS_COMPUTE_16F) c.CUDA_R_16F else c.CUDA_R_32F;
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
                other_tensor.ptr,
                other_layout,
                &beta,
                if (@TypeOf(add_tensor) != @TypeOf(null)) add_tensor.ptr else out_tensor.ptr,
                add_layout,
                out_tensor.ptr,
                out_layout,
                &heuristic.algo,
                cuda_context.cublaslt_workspace,
                cuda_context.cublaslt_workspace_size,
                stream.stream,
            ));

            return out_tensor;
        }

        pub fn matmul(
            self: *const Self,
            self_transpose: bool,
            other_tensor: anytype,
            other_transpose: bool,
            add_tensor: anytype,
            add_transpose: bool,
            comptime cublas_compute_type: c.cublasComputeType_t,
            alpha: if (cublas_compute_type == c.CUBLAS_COMPUTE_16F) f16 else f32,
            beta: if (cublas_compute_type == c.CUBLAS_COMPUTE_16F) f16 else f32,
            comptime EpilogueT: type,
            epilogue_config: EpilogueT.Config,
            //cublas_compute_type: c.cublasComputeType_t,
            stream: *const Stream,
            cuda_context: *const CudaContext,
            comptime OutType: type,
        ) !GPUTensor(OutType) {
            var matmul_res = try self.matmulTransposed(
                self_transpose,
                other_tensor,
                other_transpose,
                add_tensor,
                add_transpose,
                cublas_compute_type,
                alpha,
                beta,
                EpilogueT,
                epilogue_config,
                stream,
                cuda_context,
                OutType,
            );
            defer matmul_res.deinitAsync(stream);

            return try matmul_res.transpose(cuda_context, stream);
        }

        pub fn tranformTransposed(
            self: *const Self,
            self_transpose: bool,
            other_tensor: anytype,
            other_transpose: bool,
            comptime CublasScaleType: type,
            alpha: CublasScaleType,
            beta: CublasScaleType,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !Self {
            if (@TypeOf(other_tensor) == @TypeOf(null)) {
                if (T == Bf16) {
                    std.debug.assert(beta.val.x == Bf16.fromF32(0.0).val.x);
                } else {
                    std.debug.assert(beta == 0.0);
                }
            }

            var out_tensor = try Self.initAsync(switch (self.base.rank) {
                3 => &.{
                    self.base.getBatch(),
                    self.base.getCol(),
                    self.base.getRow(),
                },
                2 => &.{
                    self.base.getCol(),
                    self.base.getRow(),
                },
                1 => &.{
                    self.base.getCol(),
                    self.base.getRow(),
                },
                else => unreachable,
            }, stream);
            errdefer out_tensor.deinitAsync(stream);

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

            if (@TypeOf(other_tensor) != @TypeOf(null)) {
                try err.checkCublas(c.cublasLtMatrixTransformDescSetAttribute(
                    transform_desc,
                    c.CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB,
                    // row major -> already transposed
                    if (other_transpose) &op_no_t else &op_t,
                    @sizeOf(c.cublasOperation_t),
                ));
            }

            const self_layout = try self.createCublasLtMatrixLayout();
            defer _ = c.cublasLtMatrixLayoutDestroy(self_layout);

            const other_layout = if (@TypeOf(other_tensor) != @TypeOf(null)) try other_tensor.createCublasLtMatrixLayout() else null;
            defer {
                if (@TypeOf(other_tensor) != @TypeOf(null)) {
                    _ = c.cublasLtMatrixLayoutDestroy(other_layout);
                }
            }

            const out_layout = try out_tensor.createCublasLtMatrixLayout();
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
                if (@TypeOf(other_tensor) != @TypeOf(null)) other_tensor.ptr else null,
                if (@TypeOf(other_tensor) != @TypeOf(null)) other_layout else null,
                out_tensor.ptr,
                out_layout,
                stream.stream,
            ));

            return out_tensor;
        }

        pub fn tranform(
            self: *const Self,
            self_transpose: bool,
            other_tensor: anytype,
            other_transpose: bool,
            comptime CublasScaleType: type,
            alpha: CublasScaleType,
            beta: CublasScaleType,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !Self {
            var transform_t = try self.tranformTransposed(
                self_transpose,
                other_tensor,
                other_transpose,
                CublasScaleType,
                alpha,
                beta,
                cuda_context,
                stream,
            );
            defer transform_t.deinitAsync(stream);

            return try transform_t.transpose(cuda_context, stream);
        }

        pub fn transpose(
            self: *const Self,
            cuda_context: *const CudaContext,
            stream: *const Stream,
        ) !Self {
            return try self.tranformTransposed(
                false,
                null,
                false,
                T,
                if (T == Bf16) Bf16.fromF32(1.0) else 1.0,
                if (T == Bf16) Bf16.fromF32(0.0) else 0.0,
                cuda_context,
                stream,
            );
        }
    };
}
