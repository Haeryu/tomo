const c = @import("c_trans.zig");
const err = @import("error.zig");

pub const CudaContext = struct {
    cublas_handle: c.cublasHandle_t = null,
    threads_per_block: c_uint,

    pub fn init() !CudaContext {
        var cublas_handle: c.cublasHandle_t = null;
        try err.checkCublas(c.cublasCreate(&cublas_handle));
        errdefer _ = c.cublasDestroy(cublas_handle);

        var threads_per_block: c_int = 0;
        try err.checkCuda(c.cudaDeviceGetAttribute(&threads_per_block, c.cudaDevAttrMaxThreadsPerBlock, 0));
        threads_per_block = @min(threads_per_block, 256);

        return .{
            .cublas_handle = cublas_handle,
            .threads_per_block = @intCast(threads_per_block),
        };
    }

    pub fn deinit(self: *CudaContext) void {
        _ = c.cublasDestroy(self.cublas_handle);
        self.cublas_handle = null;
    }
};
