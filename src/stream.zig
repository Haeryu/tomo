const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");

const Event = @import("event.zig").Event;
const Graph = @import("graph.zig").Graph;

pub const Stream = struct {
    stream: c.cudaStream_t = null,

    pub fn create() !Stream {
        var stream: c.cudaStream_t = null;
        try err.checkCuda(c.cudaStreamCreate(&stream));
        return .{ .stream = stream };
    }

    pub fn default() Stream {
        return .{ .stream = null };
    }

    pub fn destroy(self: *Stream) void {
        self.sync() catch {};
        if (self.stream) |stream| {
            _ = c.cudaStreamDestroy(stream);
        }
        self.stream = null;
    }

    pub fn sync(self: *const Stream) !void {
        try err.checkCuda(c.cudaStreamSynchronize(self.stream));
    }

    pub fn waitEvent(self: *const Stream, event: *const Event, flags: c_int) !void {
        try err.checkCuda(c.cudaStreamWaitEvent(self.stream, event.event, flags));
    }

    pub fn beginCapture(self: *const Stream, mode: c.cudaStreamCaptureMode) !void {
        try err.checkCuda(c.cudaStreamBeginCapture(self.stream, mode));
    }

    pub fn endCapture(self: *const Stream) !Graph {
        var graph: c.cudaGraph_t = null;
        try err.checkCuda(c.cudaStreamEndCapture(self.stream, &graph));
        return .{ .graph = graph };
    }
};
