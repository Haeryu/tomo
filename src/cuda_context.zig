const c = @import("c_trans.zig");
const err = @import("error.zig");
const std = @import("std");

const Stream = @import("stream.zig").Stream;

pub const CudaContext = struct {
    cublas_handle: c.cublasHandle_t = null,
    threads_per_block: c_uint = 0,
    curand_generator: c.curandGenerator_t = null,
    cublaslt_handle: c.cublasLtHandle_t = null,
    cublaslt_workspace_size: usize = 32 * 1024 * 1024,
    cublaslt_workspace: ?*anyopaque = null,

    pub fn init() !CudaContext {
        var cublas_handle: c.cublasHandle_t = null;
        try err.checkCublas(c.cublasCreate(&cublas_handle));
        errdefer _ = c.cublasDestroy(cublas_handle);

        var threads_per_block: c_int = 0;
        try err.checkCuda(c.cudaDeviceGetAttribute(&threads_per_block, c.cudaDevAttrMaxThreadsPerBlock, 0));
        threads_per_block = @min(threads_per_block, 256);

        var curand_generator: c.curandGenerator_t = null;
        try err.checkCurand(c.curandCreateGenerator(&curand_generator, c.CURAND_RNG_PSEUDO_DEFAULT));
        errdefer _ = c.curandDestroyGenerator(curand_generator);
        try err.checkCurand(c.curandSetPseudoRandomGeneratorSeed(curand_generator, @intCast(std.time.timestamp())));

        var cublaslt_handle: c.cublasLtHandle_t = null;
        try err.checkCublas(c.cublasLtCreate(&cublaslt_handle));
        errdefer _ = c.cublasLtDestroy(cublaslt_handle);

        const cublaslt_workspace_size = 32 * 1024 * 1024;
        var cublaslt_workspace: ?*anyopaque = null;
        try err.checkCuda(c.cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));
        errdefer _ = c.cudaFree(cublaslt_workspace);

        return .{
            .cublas_handle = cublas_handle,
            .threads_per_block = @intCast(threads_per_block),
            .curand_generator = curand_generator,
            .cublaslt_handle = cublaslt_handle,
            .cublaslt_workspace_size = cublaslt_workspace_size,
            .cublaslt_workspace = cublaslt_workspace,
        };
    }

    pub fn setCurandStream(self: *const CudaContext, stream: *const Stream) !void {
        try err.checkCurand(c.curandSetStream(self.curand_generator, stream.stream));
    }

    pub fn deinit(self: *CudaContext) void {
        _ = c.cudaFree(self.cublaslt_workspace);
        _ = c.cublasLtDestroy(self.cublaslt_handle);
        _ = c.curandDestroyGenerator(self.curand_generator);
        _ = c.cublasDestroy(self.cublas_handle);
        self.cublas_handle = null;
    }
};
