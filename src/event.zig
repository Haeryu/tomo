const std = @import("std");
const c = @import("c_trans.zig");
const err = @import("error.zig");

const Stream = @import("stream.zig").Stream;

pub const Event = struct {
    event: c.cudaEvent_t = null,

    pub fn create() !Event {
        var event: c.cudaEvent_t = null;
        try err.checkCuda(c.cudaEventCreate(&event));
        return .{ .event = event };
    }

    pub fn destroy(self: *Event) void {
        _ = c.cudaEventDestroy(self.event);
        self.event = null;
    }

    pub fn record(self: *const Event, stream: *const Stream) !void {
        try err.checkCuda(c.cudaEventRecord(self.event, stream.stream));
    }

    pub fn wait(self: *const Event) !void {
        try err.checkCuda(c.cudaEventSynchronize(self.event));
    }
};
