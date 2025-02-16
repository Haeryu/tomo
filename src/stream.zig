const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");

pub const Stream = struct {
    stream: c.cudaStream_t = null,

    pub fn create() !Stream {
        var stream: c.cudaStream_t = null;
        try err.checkCuda(c.cudaStreamCreate(&stream));
        return .{ .stream = stream };
    }

    pub fn destroy(self: *Stream) void {
        if (self.stream) |s| {
            _ = c.cudaStreamDestroy(s);
            self.stream = null;
        }
    }

    pub fn sync(self: *const Stream) !void {
        try err.checkCuda(c.cudaStreamSynchronize(self.stream));
    }
};
