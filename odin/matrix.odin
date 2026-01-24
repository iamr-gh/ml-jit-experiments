package main

// eventually need
DynMatrix :: struct {
	r:    int,
	c:    int,
	data: [dynamic]f32,
}


// we need a dynamic matrix type
MatOpType :: enum {
	Add,
	MatMul,
}

MatOp :: struct {
	type: MatOpType,
	l:    ^MatNode,
	r:    ^MatNode,
}

// variables are now of matrix value
MatNode :: union {
	MatOp,
}

matrix_testing :: proc() {
	// what I want to do
	// linear regression y = Ax + B
	m: matrix[4, 2]f32


}
