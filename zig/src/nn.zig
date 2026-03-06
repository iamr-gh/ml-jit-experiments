const std = @import("std");
const t = @import("tensor.zig"); // might change naming

fn basicMatmul() t.Tensor(f32, .{ 2, 1 }) {
    const x = t.tensor(f32, .{ 2, 1 }, .{ 1, 2 });
    const A = t.tensor(f32, .{ 2, 2 }, .{ 1, 2, 3, 4 });

    return A.matmul(x);
}

fn mul_func(x: anytype) t.Tensor(f32, .{ 2, 1 }) {
    const A = t.tensor(f32, .{ 2, 2 }, .{ 1, 2, 3, 4 });
    return A.matmul(x);
}

fn dense_layer(x: anytype, w: anytype, b: anytype) @TypeOf(w.matmul(x).add(b).ew(t.sigmoid)) {
    return w.matmul(x).add(b).ew(t.sigmoid);
}

pub fn singleNode() void {
    std.debug.print("Writing a basic mat mul pieces\n", .{});

    const y = basicMatmul();
    std.debug.print("{f}\n", .{y});

    const x = t.tensor(f32, .{ 2, 1 }, .{ 1, 2 });
    const y2 = mul_func(x);
    std.debug.print("{f}\n", .{y2});

    const w = t.tensor(f32, .{ 2, 2 }, .{ 1, 0, 0, 1 });
    const b = t.tensor(f32, .{ 2, 1 }, .{ 1, -1 });
    const y3 = dense_layer(x, w, b);
    std.debug.print("{f}\n", .{y3});
}

test "dense layer applies bias and sigmoid" {
    const x = t.tensor(f32, .{ 2, 1 }, .{ 0, 1 });
    const w = t.tensor(f32, .{ 2, 2 }, .{ 1, 0, 0, 1 });
    const b = t.tensor(f32, .{ 2, 1 }, .{ 1, -1 });

    const y = dense_layer(x, w, b);

    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), y.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), y.data[1], 0.0001);
}
