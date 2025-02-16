const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");

pub const cuda_pinned_allocator: std.mem.Allocator = .{
    .ptr = undefined,
    .vtable = &.{
        .alloc = alloc,
        .resize = std.mem.Allocator.noResize,
        .free = free,
    },
};

fn alloc(
    _: *anyopaque,
    len: usize,
    _: u8,
    _: usize,
) ?[*]u8 {
    var ptr: ?[*]u8 = null;
    err.checkCuda(c.cudaHostAlloc(@ptrCast(&ptr), len, c.cudaHostAllocDefault)) catch {
        ptr = null;
    };

    return ptr;
}

fn free(_: *anyopaque, buf: []u8, _: u8, _: usize) void {
    _ = c.cudaFreeHost(@ptrCast(buf.ptr));
}
