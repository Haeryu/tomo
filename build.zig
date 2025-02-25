const std = @import("std");

pub fn build(b: *std.Build) void {
    const cuda_path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\";
    const cudnn_path = "C:\\Program Files\\NVIDIA\\CUDNN\\v9.7\\";

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib_mod = b.addModule("tomo", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });

    lib_mod.addIncludePath(.{ .cwd_relative = cuda_path ++ "include" });
    lib_mod.addIncludePath(.{ .cwd_relative = cudnn_path ++ "include\\12.8" });
    lib_mod.addLibraryPath(.{ .cwd_relative = cuda_path ++ "bin" });
    lib_mod.addLibraryPath(.{ .cwd_relative = cudnn_path ++ "bin\\12.8" });

    lib_mod.addIncludePath(b.path("src/kernel/"));
    lib_mod.addLibraryPath(b.path("src/kernel/out/"));

    lib_mod.linkSystemLibrary("cudart64_12", .{});
    lib_mod.linkSystemLibrary("cublas64_12", .{});
    lib_mod.linkSystemLibrary("cublasLt64_12", .{});
    lib_mod.linkSystemLibrary("curand64_10", .{});
    lib_mod.linkSystemLibrary("cufft64_11", .{});
    lib_mod.linkSystemLibrary("cudnn64_9", .{});
    lib_mod.linkSystemLibrary("cusolver64_11", .{});
    lib_mod.linkSystemLibrary("cusolverMg64_11", .{});
    lib_mod.linkSystemLibrary("cusparse64_12", .{});
    lib_mod.linkSystemLibrary("tomo_kernels", .{});
    lib_mod.addObjectFile(b.path("src/kernel/out/tomo_kernels.lib"));

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe_mod.addImport("tomo_lib", lib_mod);

    const exe = b.addExecutable(.{
        .name = "tomo",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    buildCuda(b);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const lib_unit_tests = b.addTest(.{
        .root_module = lib_mod,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}

fn buildCuda(b: *std.Build) void {
    const py = b.addSystemCommand(&.{"python"});
    py.addArg("./src/kernel/compile.py");

    const copy_step = b.step("copy", "copy dll to zig-out");
    copy_step.makeFn = copy;

    copy_step.dependOn(&py.step);

    const cuda_step = b.step("cuda", "compile cuda");
    cuda_step.dependOn(copy_step);
}

fn copy(_: *std.Build.Step, _: std.Build.Step.MakeOptions) anyerror!void {
    std.fs.cwd().copyFile(
        "./src/kernel/out/tomo_kernels.dll",
        std.fs.cwd(),
        "./zig-out/bin/tomo_kernels.dll",
        .{},
    ) catch unreachable;
}
