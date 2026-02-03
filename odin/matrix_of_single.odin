package main

// decompose gradient of any matrix operation into a sequence of single operations

// going to be row major for now
NodeMatrix :: struct {
	r:    int,
	c:    int,
	data: []Node,
}

binaryOpNodeMat :: proc(type: OpType, a: ^NodeMatrix, b: ^NodeMatrix) -> NodeMatrix {
	// shapes must match
	assert(a.r == b.r && a.c == b.c && len(a.data) == len(b.data))

	out_data: [dynamic]Node
	resize(&out_data, len(a.data))

	for i in 0 ..< len(a.data) {
		out_data[i] = Op{type, &a.data[i], &b.data[i]}
	}

	return NodeMatrix{a.r, a.c, out_data[:]}
}

// any other binary operation can be computed in this format
reduceNodeMat :: proc(type: OpType, mat: ^NodeMatrix) -> ^Node {
	// could switch to a tree format instead
	base := &mat.data[0]
	for i in 1 ..< len(mat.data) {
		new_parent := new(Node, context.temp_allocator)
		new_parent^ = Op{type, base, &mat.data[i]}
		base = new_parent
	}

	return base
}

// Extract row r_idx from matrix as a 1D NodeMatrix (1 x c)
getRow :: proc(mat: ^NodeMatrix, r_idx: int) -> NodeMatrix {
	row_data: [dynamic]Node
	resize(&row_data, mat.c)
	for c_idx in 0 ..< mat.c {
		row_data[c_idx] = mat.data[r_idx * mat.c + c_idx]
	}
	return NodeMatrix{1, mat.c, row_data[:]}
}

// Extract column c_idx from matrix as a 1D NodeMatrix (r x 1)
getCol :: proc(mat: ^NodeMatrix, c_idx: int) -> NodeMatrix {
	col_data: [dynamic]Node
	resize(&col_data, mat.r)
	for r_idx in 0 ..< mat.r {
		col_data[r_idx] = mat.data[r_idx * mat.c + c_idx]
	}
	return NodeMatrix{mat.r, 1, col_data[:]}
}

// Element-wise binary operation on vectors (only requires same data length, not shape)
// Used for dot product where row (1 x n) and column (n x 1) have different shapes
binaryOpNodeVec :: proc(type: OpType, a: ^NodeMatrix, b: ^NodeMatrix) -> NodeMatrix {
	// Only require same data length for vectors
	assert(len(a.data) == len(b.data))

	out_data: [dynamic]Node
	resize(&out_data, len(a.data))

	for i in 0 ..< len(a.data) {
		out_data[i] = Op{type, &a.data[i], &b.data[i]}
	}

	// Return as a flat vector (1 x len)
	return NodeMatrix{1, len(a.data), out_data[:]}
}

// Dot product of two vectors (as flat NodeMatrix arrays of same length)
dotNodeMat :: proc(a: ^NodeMatrix, b: ^NodeMatrix) -> ^Node {
	assert(len(a.data) == len(b.data))
	prod := binaryOpNodeVec(.Mul, a, b)
	return reduceNodeMat(.Add, &prod)
}

multNodeMat :: proc(a: ^NodeMatrix, b: ^NodeMatrix) -> NodeMatrix {
	// (a.r x a.c) x (b.r x b.c) -> a.r x b.c
	assert(a.c == b.r)

	out_data: [dynamic]Node
	resize(&out_data, a.r * b.c)

	for r_idx in 0 ..< a.r {
		row := getRow(a, r_idx)
		for c_idx in 0 ..< b.c {
			col := getCol(b, c_idx)
			// Dot product of row and column
			// row is 1 x a.c, col is b.r x 1, but they have same length (a.c == b.r)
			dot_result := dotNodeMat(&row, &col)
			out_data[r_idx * b.c + c_idx] = dot_result^
		}
	}

	return NodeMatrix{a.r, b.c, out_data[:]}
}
