package main

main :: proc() {
	old_allocator := context.allocator
	context.allocator = context.temp_allocator // make arena alloc default
	defer context.allocator = old_allocator

	test_grad_matrix()
	trainLinearMatrixGrad()
}
