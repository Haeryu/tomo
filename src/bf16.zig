const std = @import("std");
const c = @import("c_trans.zig");

pub const BF16 = extern struct {
    val: c.__nv_bfloat16_raw,

    pub fn format(
        self: BF16,
        comptime fmt: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        try writer.print("{" ++ fmt ++ "}", .{self.toF32()});
    }

    pub fn fromF16(val: f16) BF16 {
        return .{
            .val = c.tomoF16ToBf16(@bitCast(val)),
        };
    }

    pub fn fromF32(val: f32) BF16 {
        return .{
            .val = c.tomoF32ToBf16(val),
        };
    }

    pub fn fromF64(val: f64) BF16 {
        return .{
            .val = c.tomoF64ToBf16(val),
        };
    }

    pub fn toF16(self: BF16) f16 {
        return @bitCast(c.tomoBf16ToF16(self.val));
    }

    pub fn toF32(self: BF16) f32 {
        return c.tomoBf16ToF32(self.val);
    }

    pub fn toF64(self: BF16) f64 {
        return c.tomoBf16ToF64(self.val);
    }

    pub fn add(self: BF16, other: BF16) BF16 {
        return .{ .val = c.tomoBf16Add(self.val, other.val) };
    }

    pub fn sub(self: BF16, other: BF16) BF16 {
        return .{ .val = c.tomoBf16Sub(self.val, other.val) };
    }

    pub fn mul(self: BF16, other: BF16) BF16 {
        return .{ .val = c.tomoBf16Mul(self.val, other.val) };
    }

    pub fn div(self: BF16, other: BF16) BF16 {
        return .{ .val = c.tomoBf16Div(self.val, other.val) };
    }

    pub fn neg(self: BF16) BF16 {
        return .{ .val = c.tomoBf16Neg(self.val) };
    }
};
