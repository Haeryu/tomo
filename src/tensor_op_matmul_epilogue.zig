const c = @import("c_trans.zig");
const err = @import("error.zig");

// pub const CUBLASLT_EPILOGUE_DEFAULT: c_int = 1;
// pub const CUBLASLT_EPILOGUE_RELU: c_int = 2;
// pub const CUBLASLT_EPILOGUE_RELU_AUX: c_int = 130;
// pub const CUBLASLT_EPILOGUE_BIAS: c_int = 4;
// pub const CUBLASLT_EPILOGUE_RELU_BIAS: c_int = 6;
// pub const CUBLASLT_EPILOGUE_RELU_AUX_BIAS: c_int = 134;
// pub const CUBLASLT_EPILOGUE_GELU: c_int = 32;
// pub const CUBLASLT_EPILOGUE_GELU_AUX: c_int = 160;
// pub const CUBLASLT_EPILOGUE_GELU_BIAS: c_int = 36;
// pub const CUBLASLT_EPILOGUE_GELU_AUX_BIAS: c_int = 164;

pub fn Epilogue(comptime BiasTensor: type, comptime AuxTensor: type) type {
    return struct {
        pub const Activation = enum {
            none,
            relu,
            gelu,
        };

        pub const Config = struct {
            activation: Activation = .none,
            bias_tensor: ?BiasTensor = null,
            aux_tensor: ?AuxTensor = null,
        };

        const Self = @This();

        fn selectEpilogue(config: Config) c.cublasLtEpilogue_t {
            switch (config.activation) {
                .none => {
                    if (config.bias_tensor != null) {
                        return c.CUBLASLT_EPILOGUE_BIAS;
                    } else {
                        return c.CUBLASLT_EPILOGUE_DEFAULT;
                    }
                },
                .relu => {
                    if (config.bias_tensor != null) {
                        if (config.aux_tensor != null) {
                            return c.CUBLASLT_EPILOGUE_RELU_AUX_BIAS;
                        } else {
                            return c.CUBLASLT_EPILOGUE_RELU_BIAS;
                        }
                    } else {
                        if (config.aux_tensor != null) {
                            return c.CUBLASLT_EPILOGUE_RELU_AUX;
                        } else {
                            return c.CUBLASLT_EPILOGUE_RELU;
                        }
                    }
                },
                .gelu => {
                    if (config.bias_tensor != null) {
                        if (config.aux_tensor != null) {
                            return c.CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
                        } else {
                            return c.CUBLASLT_EPILOGUE_GELU_BIAS;
                        }
                    } else {
                        if (config.aux_tensor != null) {
                            return c.CUBLASLT_EPILOGUE_GELU_AUX;
                        } else {
                            return c.CUBLASLT_EPILOGUE_GELU;
                        }
                    }
                },
            }
        }

        pub fn setAuxTensor(matmul_desc: c.cublasLtMatmulDesc_t, config: Config) !void {
            if (config.aux_tensor) |aux_buf| {
                const aux_data_type: c.cublasDataType_t = @intCast(@TypeOf(aux_buf).getCudaDatatype());
                const row: u64 = aux_buf.base.shape[aux_buf.base.shape.len - 2];
                const col: u64 = aux_buf.base.shape[aux_buf.base.shape.len - 1];
                const batch_stride: i64 = @intCast(row * col);
                try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    c.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
                    @ptrCast(&aux_data_type),
                    @sizeOf(@TypeOf(aux_data_type)),
                ));
                try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    c.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                    @ptrCast(&aux_buf.ptr),
                    @sizeOf(@TypeOf(aux_buf.ptr)),
                ));
                try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    c.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                    &col,
                    @sizeOf(@TypeOf(col)),
                ));
                try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    c.CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
                    &batch_stride,
                    @sizeOf(@TypeOf(batch_stride)),
                ));
            }
        }

        pub fn setBiasTensor(matmul_desc: c.cublasLtMatmulDesc_t, config: Config) !void {
            if (config.bias_tensor) |bias_buf| {
                const bias_data_type: c.cublasDataType_t = @intCast(@TypeOf(bias_buf).getCudaDatatype());
                const row: u64 = bias_buf.base.shape[bias_buf.base.shape.len - 2];
                const col: u64 = bias_buf.base.shape[bias_buf.base.shape.len - 1];
                const batch_stride: i64 = @intCast(row * col);
                try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    c.CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    &bias_data_type,
                    @sizeOf(@TypeOf(bias_data_type)),
                ));
                try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    c.CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    @ptrCast(&bias_buf.ptr),
                    @sizeOf(@TypeOf(bias_buf.ptr)),
                ));
                try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    c.CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
                    @ptrCast(&batch_stride),
                    @sizeOf(@TypeOf(batch_stride)),
                ));
            }
        }

        pub fn setEpilogue(matmul_desc: c.cublasLtMatmulDesc_t, config: Config) !void {
            const epilogue = selectEpilogue(config);

            if (AuxTensor != void) {
                try setAuxTensor(matmul_desc, config);
            }

            if (BiasTensor != void) {
                try setBiasTensor(matmul_desc, config);
            }

            try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                matmul_desc,
                c.CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epilogue,
                @sizeOf(@TypeOf(epilogue)),
            ));
        }
    };
}

// pub const CUBLASLT_EPILOGUE_DRELU: c_int = 136;
// pub const CUBLASLT_EPILOGUE_DRELU_BGRAD: c_int = 152;
// pub const CUBLASLT_EPILOGUE_DGELU: c_int = 192;
// pub const CUBLASLT_EPILOGUE_DGELU_BGRAD: c_int = 208;
// pub const CUBLASLT_EPILOGUE_BGRADA: c_int = 256;
// pub const CUBLASLT_EPILOGUE_BGRADB: c_int = 512;
pub fn DEpilogue(comptime BiasGradTensor: type) type {
    return struct {
        pub const DActivation = enum {
            none,
            drelu,
            dgelu,
        };

        pub const DBiasApply = enum {
            none,
            a,
            b,
            out,
        };

        pub const Config = struct {
            activation: DActivation = .none,
            bias_grad_apply: DBiasApply = .none,
            bias_grad_tensor: ?BiasGradTensor = null,
        };

        const Self = @This();

        pub fn selectEpilogue(config: Config) c.cublasLtEpilogue_t {
            switch (config.activation) {
                .none => {
                    switch (config.bias_grad_apply) {
                        .a => {
                            return c.CUBLASLT_EPILOGUE_BGRADA;
                        },
                        .b => {
                            return c.CUBLASLT_EPILOGUE_BGRADB;
                        },
                        else => unreachable,
                    }
                },
                .drelu => {
                    switch (config.bias_grad_apply) {
                        .none => {
                            return c.CUBLASLT_EPILOGUE_DRELU;
                        },
                        .out => {
                            return c.CUBLASLT_EPILOGUE_DRELU_BGRAD;
                        },
                        else => unreachable,
                    }
                },
                .dgelu => {
                    switch (config.bias_grad_apply) {
                        .none => {
                            return c.CUBLASLT_EPILOGUE_DGELU;
                        },
                        .out => {
                            return c.CUBLASLT_EPILOGUE_DGELU_BGRAD;
                        },
                        else => unreachable,
                    }
                },
            }
        }

        pub fn setBiasGradTensor(matmul_desc: c.cublasLtMatmulDesc_t, config: Config) !void {
            if (config.bias_grad_tensor) |bias_grad_tensor| {
                const bias_data_type: c.cublasDataType_t = @intCast(@TypeOf(bias_grad_tensor).getCudaDatatype());
                const row: u64 = bias_grad_tensor.base.shape[bias_grad_tensor.base.shape.len - 2];
                const col: u64 = bias_grad_tensor.base.shape[bias_grad_tensor.base.shape.len - 1];
                const batch_stride: i64 = @intCast(row * col);
                try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    c.CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    &bias_data_type,
                    @sizeOf(@TypeOf(bias_data_type)),
                ));
                try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    c.CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    @ptrCast(&bias_grad_tensor.ptr),
                    @sizeOf(@TypeOf(bias_grad_tensor.ptr)),
                ));
                try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                    matmul_desc,
                    c.CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
                    @ptrCast(&batch_stride),
                    @sizeOf(@TypeOf(batch_stride)),
                ));
            }
        }

        pub fn setEpilogue(matmul_desc: c.cublasLtMatmulDesc_t, config: Config) !void {
            const epilogue = selectEpilogue(config);

            if (BiasGradTensor != void) {
                try setBiasGradTensor(matmul_desc, config);
            }

            try err.checkCublas(c.cublasLtMatmulDescSetAttribute(
                matmul_desc,
                c.CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epilogue,
                @sizeOf(@TypeOf(epilogue)),
            ));
        }
    };
}

// pub fn Matmul(comptime Epilogue: type) type {
//     return struct {};
// }
