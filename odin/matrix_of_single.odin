package main

// decompose gradient of any matrix operation into a sequence of single operations

// I don't feel like writing a parser, let's create ast parsers directly

NodeMatrix :: struct {
	r:    int,
	c:    int,
	data: [dynamic]Node,
}

binaryOpNodeMat :: proc(type: OpType, a: ^NodeMatrix, b: ^NodeMatrix) -> NodeMatrix {
	// shapes must match
	assert(a.r == b.r && a.c == b.c && len(a.data) == len(b.data))

	out_data: [dynamic]Node
	resize(&out_data, len(a.data))

	for i in 0 ..< len(a.data) {
		out_data[i] = Op{type, &a.data[i], &b.data[i]}
	}

	return NodeMatrix{a.r, a.c, out_data}
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

multNodeMat :: proc(a: ^NodeMatrix, b: ^NodeMatrix) {
	// (a.r x a.c) x (b.r x b.c) -> a.r x b.c
	assert(a.c == b.r)

	out_data: [dynamic]Node
	resize(&out_data, a.r * b.c)

	for r_idx in 0 ..< a.r {
		for c_idx in 0 ..< b.c {
		}
	}

}
