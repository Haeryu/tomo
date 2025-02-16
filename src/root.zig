//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
// pub const c = @import("c.zig");
pub const c = @import("c_trans.zig");
pub const tensor = @import("tensor.zig");
pub const stream = @import("stream.zig");
pub const allocator = @import("allocator.zig");
pub const err = @import("error.zig");
pub const cuda_context = @import("cuda_context.zig");

test {}
