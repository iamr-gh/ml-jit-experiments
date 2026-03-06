const std = @import("std");

fn product(comptime shape: anytype) usize {
    comptime var p: usize = 1;
    inline for (shape) |dim| {
        p *= dim;
    }
    return p;
}

fn elem_sum(comptime shape: anytype) usize {
    comptime var p: usize = 0;
    inline for (shape) |dim| {
        p += dim;
    }
    return p;
}

fn rankOf(comptime shape: anytype) usize {
    return shape.len;
}

fn stridesOf(comptime shape: anytype) [shape.len]usize {
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

fn isShapeType(comptime shape: anytype) bool {
    return switch (@typeInfo(@TypeOf(shape))) {
        .type => @hasDecl(shape, "Dims"),
        else => false,
    };
}

fn shapeTypeOf(comptime shape: anytype) type {
    return switch (@typeInfo(@TypeOf(shape))) {
        .type => shape,
        else => makeShape(shape),
    };
}

pub fn Shape(comptime dims: anytype) type {
    return struct {
        pub const Dims = dims;
        pub const Rank = rankOf(dims);
        pub const Len = product(dims);
        pub const Strides = stridesOf(dims);
    };
}

const makeShape = Shape;

pub fn Tensor(comptime T: type, comptime shape: anytype) type {
    const ShapeInfo = shapeTypeOf(shape);
    const dims = ShapeInfo.Dims;
    const rank = ShapeInfo.Rank;
    const len = ShapeInfo.Len;
    const strides = ShapeInfo.Strides;

    return struct {
        const Self = @This();

        pub const Element = T;
        pub const Shape = ShapeInfo;
        pub const Dims = dims;
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
                if (v >= dims[i]) {
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

        // elementwise operation
        pub fn ew(self: Self, comptime op: fn (T) T) Self {
            var out: Self = undefined;
            inline for (0..len) |i| {
                out.data[i] = op(self.data[i]);
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
            const NewShape = shapeTypeOf(new_shape);
            comptime {
                if (NewShape.Len != len) {
                    @compileError("reshape requires equal element count");
                }
            }
            return .{ .data = self.data };
        }

        pub fn matmul(
            self: Self,
            other: anytype,
        ) Tensor(T, .{ dims[0], @TypeOf(other).Dims[1] }) {
            const Other = @TypeOf(other);

            if (rank != 2 or Other.Rank != 2) {
                @compileError("matmul currently supports rank-2 tensors only");
            }
            if (dims[1] != Other.Dims[0]) {
                @compileError("matmul shape mismatch");
            }
            if (T != Other.Element) {
                @compileError("matmul element types must match");
            }

            const M = dims[0];
            const K = dims[1];
            const N = Other.Dims[1];

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

        pub fn format(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("Tensor(", .{});
            inline for (dims, 0..) |dim, i| {
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

pub fn tensor(comptime T: type, comptime shape: anytype, comptime values: anytype) Tensor(T, shape) {
    return Tensor(T, shape).fromSlice(values);
}

fn relu(x: f32) f32 {
    return if (x > 0) x else 0;
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
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

test "tensor helper constructor" {
    const a = tensor(f32, .{ 2, 2 }, .{ 1, 2, 3, 4 });
    const b = tensor(f32, .{ 2, 2 }, .{ 10, 20, 30, 40 });

    try std.testing.expect(a.add(b).eql(tensor(f32, .{ 2, 2 }, .{ 11, 22, 33, 44 })));
}

test "tensor elementwise relu" {
    const T = Tensor(f32, .{ 2, 2 });
    const a = T.fromSlice(.{ -1, 2, -3, 4 });

    try std.testing.expect(a.ew(relu).eql(T.fromSlice(.{ 0, 2, 0, 4 })));
}

test "tensor elementwise sigmoid" {
    const T = Tensor(f32, .{2});
    const a = T.fromSlice(.{ 0, 1 });
    const actual = a.ew(sigmoid);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), actual.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), actual.data[1], 0.0001);
}

test "tensor format" {
    const a = tensor(i32, .{ 2, 2 }, .{ 1, 2, 3, 4 });
    var buf: [128]u8 = undefined;
    const actual = try std.fmt.bufPrint(&buf, "{f}", .{a});

    try std.testing.expectEqualStrings("Tensor(2x2)[1, 2, 3, 4]", actual);
}

test "named shape format" {
    const Matrix = Shape(.{ 2, 1 });
    const a = tensor(i32, Matrix, .{ 5, 6 });
    var buf: [128]u8 = undefined;
    const actual = try std.fmt.bufPrint(&buf, "{f}", .{a});

    try std.testing.expectEqualStrings("Tensor(2x1)[5, 6]", actual);
}

test "named shape type" {
    const Matrix = Shape(.{ 2, 2 });
    const T = Tensor(f32, Matrix);
    const a = tensor(f32, Matrix, .{ 1, 2, 3, 4 });

    comptime {
        if (T.Shape != Matrix) {
            @compileError("tensor should retain the named shape type");
        }
    }

    try std.testing.expect(a.eql(T.fromSlice(.{ 1, 2, 3, 4 })));
    try std.testing.expectEqual(@as(usize, 2), T.Shape.Rank);
    try std.testing.expectEqual(@as(usize, 4), T.Shape.Len);
}
