package main

main :: proc() {
	old_allocator := context.allocator
	context.allocator = context.temp_allocator // make arena alloc default
	defer context.allocator = old_allocator

	// basic_test()
	// eval_vm_basic_test()
	// time_base_test()
	// time_forward_vs_reverse()
	test_node_matrix_all()
}
