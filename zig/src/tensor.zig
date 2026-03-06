const std = @import("std");

fn product(comptime shape: anytype) usize {
    comptime var p: usize = 1;
    inline for (shape) |dim| {
        p *= dim;
    }
    return p;
}

fn rankOf(comptime shape: anytype) usize {
    return shape.len;
}

fn Strides(comptime shape: anytype) [shape.len]usize {
    comptime var strides: [shape.len]usize = undefined;
    comptime var acc: usize = 1;
    comptime var i: usize = shape.len;
    while (i > 0) {
        i -= 1;
        strides[i] = acc;
        acc *= shape[i];
    }
    return strides;
}

pub fn Tensor(comptime T: type, comptime shape: anytype) type {
    const rank = rankOf(shape);
    const len = product(shape);
    const strides = Strides(shape);

    return struct {
        const Self = @This();

        pub const Element = T;
        pub const Shape = shape;
        pub const Rank = rank;
        pub const Len = len;

        data: [len]T,

        pub fn zeros() Self {
            return .{ .data = [_]T{0} ** len };
        }

        pub fn fromSlice(comptime values: anytype) Self {
            if (values.len != len) {
                @compileError("fromSlice length does not match tensor size");
            }
            return .{ .data = values };
        }

        pub fn at(self: *const Self, comptime idx: [rank]usize) T {
            return self.data[flatIndex(idx)];
        }

        pub fn set(self: *Self, comptime idx: [rank]usize, value: T) void {
            self.data[flatIndex(idx)] = value;
        }

        fn flatIndex(comptime idx: [rank]usize) usize {
            comptime var flat: usize = 0;
            inline for (idx, 0..) |v, i| {
                if (v >= shape[i]) {
                    @compileError("tensor index out of bounds");
                }
                flat += v * strides[i];
            }
            return flat;
        }

        pub fn add(self: Self, other: Self) Self {
            var out: Self = undefined;
            inline for (0..len) |i| {
                out.data[i] = self.data[i] + other.data[i];
            }
            return out;
        }

        pub fn mulScalar(self: Self, scalar: T) Self {
            var out: Self = undefined;
            inline for (0..len) |i| {
                out.data[i] = self.data[i] * scalar;
            }
            return out;
        }

        pub fn reshape(self: Self, comptime new_shape: anytype) Tensor(T, new_shape) {
            comptime {
                if (product(new_shape) != len) {
                    @compileError("reshape requires equal element count");
                }
            }
            return .{ .data = self.data };
        }

        pub fn matmul(
            self: Self,
            other: anytype,
        ) Tensor(T, .{ shape[0], @TypeOf(other).Shape[1] }) {
            const Other = @TypeOf(other);

            if (rank != 2 or Other.Rank != 2) {
                @compileError("matmul currently supports rank-2 tensors only");
            }
            if (shape[1] != Other.Shape[0]) {
                @compileError("matmul shape mismatch");
            }
            if (T != Other.Element) {
                @compileError("matmul element types must match");
            }

            const M = shape[0];
            const K = shape[1];
            const N = Other.Shape[1];

            var out = Tensor(T, .{ M, N }).zeros();

            inline for (0..M) |i| {
                inline for (0..N) |j| {
                    var sum: T = 0;
                    inline for (0..K) |k| {
                        sum += self.data[i * K + k] * other.data[k * N + j];
                    }
                    out.data[i * N + j] = sum;
                }
            }
            return out;
        }

        pub fn eql(self: Self, other: Self) bool {
            inline for (0..len) |i| {
                if (self.data[i] != other.data[i]) return false;
            }
            return true;
        }

        pub fn format(
            self: Self,
            comptime _: []const u8,
            _: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            try writer.print("Tensor(", .{});
            inline for (shape, 0..) |dim, i| {
                if (i != 0) try writer.print("x", .{});
                try writer.print("{}", .{dim});
            }
            try writer.print(")[", .{});
            inline for (0..len) |i| {
                if (i != 0) try writer.print(", ", .{});
                try writer.print("{}", .{self.data[i]});
            }
            try writer.print("]", .{});
        }
    };
}

test "tensor add" {
    const T = Tensor(f32, .{ 2, 2 });

    const a = T.fromSlice(.{ 1, 2, 3, 4 });
    const b = T.fromSlice(.{ 10, 20, 30, 40 });
    const c = a.add(b);

    try std.testing.expect(c.eql(T.fromSlice(.{ 11, 22, 33, 44 })));
}

test "tensor reshape" {
    const A = Tensor(i32, .{ 2, 3 });
    const B = Tensor(i32, .{ 3, 2 });

    const a = A.fromSlice(.{ 1, 2, 3, 4, 5, 6 });
    const b = a.reshape(.{ 3, 2 });

    try std.testing.expect(b.eql(B.fromSlice(.{ 1, 2, 3, 4, 5, 6 })));
}

test "tensor matmul" {
    const A = Tensor(f32, .{ 2, 3 });
    const B = Tensor(f32, .{ 3, 2 });
    const C = Tensor(f32, .{ 2, 2 });

    const a = A.fromSlice(.{ 1, 2, 3, 4, 5, 6 });
    const b = B.fromSlice(.{ 7, 8, 9, 10, 11, 12 });

    const c = a.matmul(b);

    try std.testing.expect(c.eql(C.fromSlice(.{
        58,  64,
        139, 154,
    })));
}
