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

fn swapLastTwoDims(comptime shape: anytype) [shape.len]usize {
    comptime var out: [shape.len]usize = undefined;
    inline for (shape, 0..) |dim, i| {
        out[i] = dim;
    }
    if (shape.len >= 2) {
        const last = shape.len - 1;
        const prev = shape.len - 2;
        const tmp = out[prev];
        out[prev] = out[last];
        out[last] = tmp;
    }
    return out;
}

fn matmulResultShape(comptime lhs_shape: anytype, comptime rhs_shape: anytype) [lhs_shape.len]usize {
    comptime {
        if (lhs_shape.len < 2 or rhs_shape.len < 2) {
            @compileError("matmul requires rank >= 2");
        }
        if (lhs_shape[lhs_shape.len - 1] != rhs_shape[rhs_shape.len - 2]) {
            @compileError("matmul shape mismatch");
        }
        if (lhs_shape.len == 2 and rhs_shape.len > 2) {
            @compileError("matmul does not support broadcasting a higher-rank rhs over a rank-2 lhs");
        }
        if (lhs_shape.len > 2 and rhs_shape.len > 2) {
            if (lhs_shape.len != rhs_shape.len) {
                @compileError("matmul batched operands must have matching rank");
            }
            for (0..lhs_shape.len - 2) |i| {
                if (lhs_shape[i] != rhs_shape[i]) {
                    @compileError("matmul batched operands must have matching leading dims");
                }
            }
        }
    }

    comptime var out: [lhs_shape.len]usize = undefined;
    inline for (0..lhs_shape.len - 2) |i| {
        out[i] = lhs_shape[i];
    }
    out[lhs_shape.len - 2] = lhs_shape[lhs_shape.len - 2];
    out[lhs_shape.len - 1] = rhs_shape[rhs_shape.len - 1];
    return out;
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

pub fn Tensor(comptime Elem: type, comptime shape: anytype) type {
    const ShapeInfo = shapeTypeOf(shape);
    const dims = ShapeInfo.Dims;
    const rank = ShapeInfo.Rank;
    const len = ShapeInfo.Len;
    const strides = ShapeInfo.Strides;
    const last_axis = if (rank == 0) 0 else rank - 1;
    const row_axis = if (rank < 2) 0 else rank - 2;

    return struct {
        const Self = @This();

        pub const Element = Elem;
        pub const Shape = ShapeInfo;
        pub const Dims = dims;
        pub const Rank = rank;
        pub const Len = len;

        fn lastAxis() comptime_int {
            comptime {
                if (rank == 0) {
                    @compileError("tensor has no last axis");
                }
            }
            return last_axis;
        }

        fn rowAxis() comptime_int {
            comptime {
                if (rank < 2) {
                    @compileError("tensor has no row axis");
                }
            }
            return row_axis;
        }

        fn hasSameShape(comptime Other: type) bool {
            if (rank != Other.Rank or len != Other.Len) {
                return false;
            }
            inline for (0..rank) |axis| {
                if (dims[axis] != Other.Dims[axis]) {
                    return false;
                }
            }
            return true;
        }

        data: [len]Elem,

        pub fn zeros() Self {
            return .{ .data = [_]Elem{0} ** len };
        }

        pub fn fromSlice(comptime values: anytype) Self {
            if (values.len != len) {
                @compileError("fromSlice length does not match tensor size");
            }
            return .{ .data = values };
        }

        pub fn dimSize(comptime axis: usize) usize {
            comptime {
                if (axis >= rank) {
                    @compileError("tensor axis out of bounds");
                }
            }
            return dims[axis];
        }

        pub fn lastDimSize() usize {
            return dims[lastAxis()];
        }

        pub fn lastDimAsElement() Elem {
            comptime {
                switch (@typeInfo(Elem)) {
                    .float, .comptime_float => {},
                    else => @compileError("lastDimAsElement requires floating-point element type"),
                }
            }
            return @as(Elem, @floatFromInt(lastDimSize()));
        }

        pub fn lastDimValue(self: Self) Elem {
            _ = self;
            return lastDimAsElement();
        }

        pub fn rowCount() usize {
            return dims[rowAxis()];
        }

        pub fn matrixElementCount() usize {
            return rowCount() * lastDimSize();
        }

        pub fn matrixCount() usize {
            return len / matrixElementCount();
        }

        pub fn outerElementCount() usize {
            return len / lastDimSize();
        }

        pub fn strideSize(comptime axis: usize) usize {
            comptime {
                if (axis >= rank) {
                    @compileError("tensor stride axis out of bounds");
                }
            }
            return strides[axis];
        }

        pub fn matrixBase(batch_index: usize) usize {
            return batch_index * matrixElementCount();
        }

        pub fn at(self: *const Self, comptime idx: [rank]usize) Elem {
            return self.data[flatIndex(idx)];
        }

        pub fn set(self: *Self, comptime idx: [rank]usize, value: Elem) void {
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

        pub fn add(self: Self, other: anytype) Self {
            const Other = @TypeOf(other);

            comptime {
                if (Elem != Other.Element or !hasSameShape(Other)) {
                    @compileError("add requires tensors with matching element types and shapes");
                }
            }

            var out: Self = undefined;
            inline for (0..len) |i| {
                out.data[i] = self.data[i] + other.data[i];
            }
            return out;
        }

        // elementwise operation
        pub fn ew(self: Self, comptime op: fn (Elem) Elem) Self {
            var out: Self = undefined;
            inline for (0..len) |i| {
                out.data[i] = op(self.data[i]);
            }
            return out;
        }

        pub fn mulScalar(self: Self, scalar: Elem) Self {
            var out: Self = undefined;
            inline for (0..len) |i| {
                out.data[i] = self.data[i] * scalar;
            }
            return out;
        }

        pub fn divScalar(self: Self, scalar: Elem) Self {
            var out: Self = undefined;
            inline for (0..len) |i| {
                out.data[i] = self.data[i] / scalar;
            }
            return out;
        }

        pub fn softmax(self: Self) Self {
            comptime {
                if (rank == 0) {
                    @compileError("softmax requires rank >= 1");
                }
                switch (@typeInfo(Elem)) {
                    .float, .comptime_float => {},
                    else => @compileError("softmax requires floating-point element type"),
                }
            }

            const last_dim = lastDimSize();
            const outer = outerElementCount();
            var out: Self = undefined;

            var slice: usize = 0;
            while (slice < outer) : (slice += 1) {
                const base = slice * last_dim;
                var max_value = self.data[base];

                var i: usize = 1;
                while (i < last_dim) : (i += 1) {
                    const value = self.data[base + i];
                    if (value > max_value) {
                        max_value = value;
                    }
                }

                var sum: Elem = 0;
                i = 0;
                while (i < last_dim) : (i += 1) {
                    const exp_value = @exp(self.data[base + i] - max_value);
                    out.data[base + i] = exp_value;
                    sum += exp_value;
                }

                i = 0;
                while (i < last_dim) : (i += 1) {
                    out.data[base + i] /= sum;
                }
            }

            return out;
        }

        pub fn reshape(self: Self, comptime new_shape: anytype) Tensor(Elem, new_shape) {
            const NewShape = shapeTypeOf(new_shape);
            comptime {
                if (NewShape.Len != len) {
                    @compileError("reshape requires equal element count");
                }
            }
            return .{ .data = self.data };
        }

        pub fn T(self: Self) Tensor(Elem, swapLastTwoDims(dims)) {
            comptime {
                if (rank < 2) {
                    @compileError("T requires rank >= 2");
                }
            }

            const Transposed = Tensor(Elem, swapLastTwoDims(dims));
            const last = lastAxis();
            const prev = rowAxis();
            var out: Transposed = undefined;

            var out_flat: usize = 0;
            while (out_flat < len) : (out_flat += 1) {
                var rest = out_flat;
                var out_idx: [rank]usize = undefined;

                inline for (0..rank) |axis| {
                    out_idx[axis] = rest / Transposed.strideSize(axis);
                    rest %= Transposed.strideSize(axis);
                }

                var src_flat: usize = 0;
                inline for (0..rank) |axis| {
                    const src_idx = if (axis == prev)
                        out_idx[last]
                    else if (axis == last)
                        out_idx[prev]
                    else
                        out_idx[axis];
                    src_flat += src_idx * strides[axis];
                }

                out.data[out_flat] = self.data[src_flat];
            }

            return out;
        }

        pub fn matmul(
            self: Self,
            other: anytype,
        ) Tensor(Elem, matmulResultShape(dims, @TypeOf(other).Dims)) {
            const Other = @TypeOf(other);

            if (Elem != Other.Element) {
                @compileError("matmul element types must match");
            }

            const M = rowCount();
            const K = lastDimSize();
            const N = Other.lastDimSize();
            const lhs_batch_count = matrixCount();
            const out_batch_stride = M * N;

            var out = Tensor(Elem, matmulResultShape(dims, Other.Dims)).zeros();

            var batch: usize = 0;
            while (batch < lhs_batch_count) : (batch += 1) {
                const lhs_base = matrixBase(batch);
                const rhs_base = if (Other.Rank == 2) 0 else Other.matrixBase(batch);
                const out_base = batch * out_batch_stride;

                var i: usize = 0;
                while (i < M) : (i += 1) {
                    var j: usize = 0;
                    while (j < N) : (j += 1) {
                        var sum: Elem = 0;
                        var k: usize = 0;
                        while (k < K) : (k += 1) {
                            sum += self.data[lhs_base + i * K + k] * other.data[rhs_base + k * N + j];
                        }
                        out.data[out_base + i * N + j] = sum;
                    }
                }
            }
            return out;
        }

        pub fn eql(self: Self, other: anytype) bool {
            const Other = @TypeOf(other);

            if (Elem != Other.Element or !hasSameShape(Other)) {
                return false;
            }
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

pub fn tensor(comptime Elem: type, comptime shape: anytype, comptime values: anytype) Tensor(Elem, shape) {
    return Tensor(Elem, shape).fromSlice(values);
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

test "tensor transpose last two dims" {
    const A = Tensor(i32, .{ 2, 3 });

    const a = A.fromSlice(.{ 1, 2, 3, 4, 5, 6 });
    const b = a.T();
    const B = @TypeOf(b);

    try std.testing.expect(b.eql(B.fromSlice(.{ 1, 4, 2, 5, 3, 6 })));
}

test "tensor transpose batched last two dims" {
    const A = Tensor(i32, .{ 2, 2, 3 });

    const a = A.fromSlice(.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    const b = a.T();
    const B = @TypeOf(b);

    try std.testing.expect(b.eql(B.fromSlice(.{ 1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12 })));
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

test "tensor matmul batched" {
    const A = Tensor(f32, .{ 2, 2, 3 });
    const B = Tensor(f32, .{ 2, 3, 2 });
    const C = Tensor(f32, .{ 2, 2, 2 });

    const a = A.fromSlice(.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    const b = B.fromSlice(.{ 7, 8, 9, 10, 11, 12, 1, 2, 0, 1, 1, 0 });

    const c = a.matmul(b);

    try std.testing.expect(c.eql(C.fromSlice(.{
        58,  64,
        139, 154,
        16,  22,
        22,  31,
    })));
}

test "tensor matmul broadcast rhs matrix over batches" {
    const X = Tensor(f32, .{ 2, 2, 3 });
    const W = Tensor(f32, .{ 3, 2 });
    const Y = Tensor(f32, .{ 2, 2, 2 });

    const x = X.fromSlice(.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    const w = W.fromSlice(.{ 1, 2, 0, 1, 1, 0 });

    const y = x.matmul(w);

    try std.testing.expect(y.eql(Y.fromSlice(.{
        4,  4,
        10, 13,
        16, 22,
        22, 31,
    })));
}

test "tensor example sequence projection and row softmax" {
    const x = tensor(f32, .{ 1, 2, 3 }, .{
        1, 0, 1,
        0, 1, 1,
    });
    const w = tensor(f32, .{ 3, 2 }, .{
        1, 0,
        0, 1,
        1, 1,
    });

    const logits = x.matmul(w);
    const probs = logits.softmax();

    try std.testing.expectEqual(@as(usize, 2), @TypeOf(logits).lastDimSize());
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(logits).rowCount());
    try std.testing.expect(logits.eql(tensor(f32, .{ 1, 2, 2 }, .{
        2, 1,
        1, 2,
    })));
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), probs.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.26894143), probs.data[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.26894143), probs.data[2], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), probs.data[3], 0.0001);
}

test "tensor example transpose exposes swapped last dims" {
    const x = tensor(f32, .{ 2, 3 }, .{ 1, 2, 3, 4, 5, 6 });
    const x_t = x.T();

    try std.testing.expectEqual(@as(usize, 3), @TypeOf(x).lastDimSize());
    try std.testing.expectEqual(@as(usize, 2), @TypeOf(x_t).lastDimSize());
    try std.testing.expect(x_t.eql(tensor(f32, .{ 3, 2 }, .{ 1, 4, 2, 5, 3, 6 })));
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

test "tensor divide by scalar" {
    const T = Tensor(f32, .{ 2, 2 });
    const a = T.fromSlice(.{ 2, 4, 6, 8 });
    const actual = a.divScalar(2);

    try std.testing.expect(actual.eql(T.fromSlice(.{ 1, 2, 3, 4 })));
}

test "tensor softmax" {
    const T = Tensor(f32, .{ 2, 3 });
    const a = T.fromSlice(.{ 1, 2, 3, 1, 1, 1 });
    const actual = a.softmax();

    try std.testing.expectApproxEqAbs(@as(f32, 0.09003057), actual.data[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.24472848), actual.data[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.66524094), actual.data[2], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.33333334), actual.data[3], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.33333334), actual.data[4], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.33333334), actual.data[5], 0.0001);
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
