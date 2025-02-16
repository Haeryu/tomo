pub usingnamespace @cImport({
    @cInclude("cuda.h");
    @cInclude("cuda_runtime.h");
    @cInclude("device_launch_parameters.h");
    @cInclude("cublas_v2.h");
    @cInclude("cudnn.h");
    @cInclude("curand.h");
    @cInclude("cufft.h");
    @cInclude("cusparse.h");
    @cInclude("cusolverDn.h");
    @cInclude("cusolverMg.h");
    @cInclude("tomo_cuda.h");
});
