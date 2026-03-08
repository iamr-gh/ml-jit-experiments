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

fn dense_layer(x: anytype, w: anytype, b: anytype) @TypeOf(w.matmul(x)) {
    // y = sigmoid(Wx + b)
    return w.matmul(x).add(b).ew(t.sigmoid);
}

fn one_attn_head(x: anytype, w_q: anytype, w_k: anytype, w_v: anytype) @TypeOf(x.matmul(w_v)) {
    const q = x.matmul(w_q);
    const k = x.matmul(w_k);
    const v = x.matmul(w_v);
    const d_k = q.lastDimValue();

    return q.matmul(k.T()).divScalar(@sqrt(d_k)).softmax().matmul(v);
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

test "one attention head applies scaled dot product attention" {
    const x = t.tensor(f32, .{ 2, 2 }, .{ 1, 0, 0, 1 });
    const w_q = t.tensor(f32, .{ 2, 2 }, .{ 1, 0, 0, 1 });
    const w_k = t.tensor(f32, .{ 2, 2 }, .{ 1, 0, 0, 1 });
    const w_v = t.tensor(f32, .{ 2, 2 }, .{ 1, 0, 0, 1 });

    const y = one_attn_head(x, w_q, w_k, w_v);

    try std.testing.expectApproxEqAbs(@as(f32, 0.66976154), y.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.33023846), y.data[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.33023846), y.data[2], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.66976154), y.data[3], 0.0001);
}

test "one attention head example with batched sequence input" {
    const x = t.tensor(f32, .{ 1, 2, 2 }, .{
        1, 0,
        0, 1,
    });
    const w_q = t.tensor(f32, .{ 2, 2 }, .{
        1, 0,
        0, 1,
    });
    const w_k = t.tensor(f32, .{ 2, 2 }, .{
        1, 0,
        0, 1,
    });
    const w_v = t.tensor(f32, .{ 2, 2 }, .{
        1, 0,
        0, 1,
    });

    const y = one_attn_head(x, w_q, w_k, w_v);

    try std.testing.expectEqual(@as(usize, 2), @TypeOf(y).lastDimSize());
    try std.testing.expectApproxEqAbs(@as(f32, 0.66976154), y.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.33023846), y.data[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.33023846), y.data[2], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.66976154), y.data[3], 0.0001);
}
