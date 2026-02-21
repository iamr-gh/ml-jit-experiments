package main

main :: proc() {
	old_allocator := context.allocator
	context.allocator = context.temp_allocator // make arena alloc default
	defer context.allocator = old_allocator

	test_grad_matrix()
	time_grad_matrix_vs_scalar(20, 20, 200, 10)
	time_grad_matrix_vs_scalar(50, 50, 500, 10)
}
