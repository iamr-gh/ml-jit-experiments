package main

DynMatrix :: struct {
	r:    int,
	c:    int,
	data: [dynamic]f32,
}

matrix_testing :: proc() {
	m: matrix[4, 2]f32
	_ = m
}
