#+feature dynamic-literals
package main
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:time"

// running mean and variance(don't want to collect all values)
// uses welford's algorithm
next_mean :: proc(m: f64, n: int, x: f64) -> f64 {
	return m + ((x - m) / f64(n))
}

next_squares :: proc(squares: f64, m_old: f64, m_new: f64, x: f64) -> f64 {
	return squares + ((x - m_old) * (x - m_new))
}

stats :: struct {
	mean: f64,
	var:  f64,
	cv:   f64,
}

time_eval :: proc(ast: ^Node, vars: []f64, runs: int) -> stats {
	sw: time.Stopwatch

	m, squares: f64
	n: int

	inner_runs := 1000

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)
		for _ in 0 ..< inner_runs {
			eval(ast, nil)
		}
		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := next_mean(m, n, x)
			squares = next_squares(squares, m, m_new, x)
			m = m_new
		}
	}

	variance := squares / f64(n)

	return stats{m, variance, math.sqrt(variance) / m}
}

time_forward_vs_reverse :: proc() {
	ast, vars := parse("5*z+1-3+4*2/5+9-11*y/7")
	vals := map[string]f32 {
		"x" = 2,
		"y" = 6,
		"z" = 10,
	}

	bound := binding_to_arr(vals, vars)

	NUM_RUNS := 100
	forward_stats := time_grad_forward(ast, bound, NUM_RUNS)
	reverse_stats := time_grad_reverse(ast, bound, NUM_RUNS)

	fmt.printf("Forward:{}\n", forward_stats)
	fmt.printf("Reverse:{}\n", reverse_stats)
}


time_base_test :: proc() {
	fmt.println("Basic time test, forward eval vs sim")

	ast, vars := parse("5*3+1-3+4*2/5+9-11*16/7")
	instrs := compile_forward(ast)
	stack_instrs := compile_forward_stack(ast)
	tiny_instrs := compile_forward_tiny(ast)

	fmt.printf("instr_len(sim):{}\n", len(instrs))
	fmt.printf("instr_len(stack):{}\n", len(stack_instrs))

	eval_res := eval(ast, nil)
	sim_res := simulate(instrs[:])
	stack_res := simulate_stack(stack_instrs[:])
	tiny_res := simulate_stack_tiny(tiny_instrs[:])


	assert(eval_res == sim_res)
	assert(eval_res == stack_res)
	assert(eval_res == tiny_res)

	NUM_RUNS := 1000
	eval_stats := time_eval(ast, nil, NUM_RUNS)
	sim_stats := time_sim(instrs[:], NUM_RUNS)
	sim_stack_stats := time_sim_stack(stack_instrs[:], NUM_RUNS)
	sim_tiny_stats := time_sim_tiny(tiny_instrs[:], NUM_RUNS)

	fmt.printf("Tree Eval stats: {}\n", eval_stats)
	fmt.printf("Sim(register) stats: {}\n", sim_stats)
	fmt.printf("Sim(stack) stats: {}\n", sim_stack_stats)
	fmt.printf("Sim(tiny) stats: {}\n", sim_tiny_stats)
}

eval_vm_basic_test :: proc() {
	ast, vars := parse("5*3+1-3+4*2")
	print_ast(ast, vars)

	// don't optimize consts
	instrs := compile_forward(ast)

	basic_stack_instrs := compile_forward_stack(ast)
	tiny_instrs := compile_forward_tiny(ast)

	eval_res := eval(ast, nil)
	sim_res := simulate(instrs[:])
	sim_tiny_res := simulate_stack_tiny(tiny_instrs[:])

	fmt.printf("Eval res: %f, sim_res: %f, stack_res: %f\n", eval_res, sim_res, sim_tiny_res)

	assert(eval_res == sim_res)
	assert(eval_res == sim_tiny_res)
}

basic_test :: proc() {
	// ast, vars := parse("x*3/2+1")
	// fmt.printf("vars: %v\n", vars)
	// print_ast(ast, vars)
	//
	// constant_prop(ast)
	// print_ast(ast, vars)
	//
	// val := eval(ast, binding_to_arr(map[string]f32{"x" = 2, "y" = 2}, vars))
	// fmt.printf("%v\n", val)
	//
	// free_all(context.temp_allocator)

	ast2, vars2 := parse("x+y*y")
	print_ast(ast2, vars2)
	vals := map[string]f32 {
		"x" = 2,
		"y" = 6,
	}

	bound := binding_to_arr(vals, vars2)


	dfdx := eval_grad_forward(ast2, bound, vars2["x"])
	dfdy := eval_grad_forward(ast2, bound, vars2["y"])

	assert(dfdx.val == 38)
	assert(dfdy.val == 38)
	assert(dfdx.grad == 1)
	assert(dfdy.grad == 12)

	grads := make([]f32, len(bound))
	defer delete(grads)
	reverse_val := eval_grad_reverse(ast2, bound, grads)
	fmt.printf("reverse grads: {}, {}\n", grads, []f32{dfdx.grad, dfdy.grad})

	assert(reverse_val == dfdx.val)
	assert(dfdx.grad == grads[vars2["x"]])
	assert(dfdy.grad == grads[vars2["y"]])

	all_val, all_grads := eval_grad_forward_all(ast2, bound)
	fmt.printf("forward all grads: {}, {}\n", all_grads, []f32{dfdx.grad, dfdy.grad})

	assert(reverse_val == dfdx.val)
	assert(dfdx.grad == all_grads[vars2["x"]])
	assert(dfdy.grad == all_grads[vars2["y"]])

	free_all(context.temp_allocator)
}

time_sim :: proc(instrs: []VInstr, runs: int) -> stats {
	sw: time.Stopwatch

	m, squares: f64
	n: int

	inner_runs := 1000

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)
		for _ in 0 ..< inner_runs {
			simulate(instrs)
		}
		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := next_mean(m, n, x)
			squares = next_squares(squares, m, m_new, x)
			m = m_new
		}
	}

	variance := squares / f64(n)

	return stats{m, variance, math.sqrt(variance) / m}
}

time_sim_stack :: proc(instrs: []VInstrStack, runs: int) -> stats {
	sw: time.Stopwatch

	m, squares: f64
	n: int

	inner_runs := 1000

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)
		for _ in 0 ..< inner_runs {
			simulate_stack(instrs)
		}
		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := next_mean(m, n, x)
			squares = next_squares(squares, m, m_new, x)
			m = m_new
		}
	}

	variance := squares / f64(n)

	return stats{m, variance, math.sqrt(variance) / m}
}

time_sim_tiny :: proc(instrs: []VInstrStackTiny, runs: int) -> stats {
	sw: time.Stopwatch

	m, squares: f64
	n: int

	inner_runs := 1000

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)
		for _ in 0 ..< inner_runs {
			simulate_stack_tiny(instrs)
		}
		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := next_mean(m, n, x)
			squares = next_squares(squares, m, m_new, x)
			m = m_new
		}
	}

	variance := squares / f64(n)

	return stats{m, variance, math.sqrt(variance) / m}
}


time_grad_forward :: proc(node: ^Node, binding: []f32, runs: int) -> stats {
	sw: time.Stopwatch

	m, squares: f64
	n: int

	inner_runs := 1000

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)
		for _ in 0 ..< inner_runs {
			eval_grad_forward_all(node, binding)
		}
		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := next_mean(m, n, x)
			squares = next_squares(squares, m, m_new, x)
			m = m_new
		}
	}

	variance := squares / f64(n)

	return stats{m, variance, math.sqrt(variance) / m}
}

time_grad_reverse :: proc(node: ^Node, binding: []f32, runs: int) -> stats {
	sw: time.Stopwatch

	m, squares: f64
	n: int

	inner_runs := 1000

	grads := make([]f32, len(binding))
	defer delete(grads)

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)
		for _ in 0 ..< inner_runs {
			eval_grad_reverse(node, binding, grads)
		}
		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := next_mean(m, n, x)
			squares = next_squares(squares, m, m_new, x)
			m = m_new
		}
	}

	variance := squares / f64(n)

	return stats{m, variance, math.sqrt(variance) / m}
}

// Helper to create a NodeMatrix from constant f32 values
makeConstNodeMatrix :: proc(rows: int, cols: int, values: []f32) -> NodeMatrix {
	assert(len(values) == rows * cols)
	data: [dynamic]Node
	resize(&data, len(values))
	for i in 0 ..< len(values) {
		data[i] = values[i]
	}
	return NodeMatrix{rows, cols, data[:]}
}

// Test binaryOpNodeMat with element-wise addition
test_binaryOpNodeMat_add :: proc() {
	fmt.println("Testing binaryOpNodeMat (Add)...")

	// 2x2 matrices
	a := makeConstNodeMatrix(2, 2, []f32{1, 2, 3, 4})
	b := makeConstNodeMatrix(2, 2, []f32{5, 6, 7, 8})

	result := binaryOpNodeMat(.Add, &a, &b)

	// Expected: [6, 8, 10, 12]
	assert(result.r == 2 && result.c == 2)
	assert(eval(&result.data[0], nil) == 6)
	assert(eval(&result.data[1], nil) == 8)
	assert(eval(&result.data[2], nil) == 10)
	assert(eval(&result.data[3], nil) == 12)

	fmt.println("  binaryOpNodeMat (Add) passed!")
}

// Test binaryOpNodeMat with element-wise multiplication
test_binaryOpNodeMat_mul :: proc() {
	fmt.println("Testing binaryOpNodeMat (Mul)...")

	// 2x2 matrices
	a := makeConstNodeMatrix(2, 2, []f32{1, 2, 3, 4})
	b := makeConstNodeMatrix(2, 2, []f32{5, 6, 7, 8})

	result := binaryOpNodeMat(.Mul, &a, &b)

	// Expected: [5, 12, 21, 32]
	assert(result.r == 2 && result.c == 2)
	assert(eval(&result.data[0], nil) == 5)
	assert(eval(&result.data[1], nil) == 12)
	assert(eval(&result.data[2], nil) == 21)
	assert(eval(&result.data[3], nil) == 32)

	fmt.println("  binaryOpNodeMat (Mul) passed!")
}

// Test reduceNodeMat with addition (sum)
test_reduceNodeMat :: proc() {
	fmt.println("Testing reduceNodeMat (Add)...")

	// 1x4 matrix (vector)
	a := makeConstNodeMatrix(1, 4, []f32{1, 2, 3, 4})

	result := reduceNodeMat(.Add, &a)

	// Expected: 1 + 2 + 3 + 4 = 10
	assert(eval(result, nil) == 10)

	fmt.println("  reduceNodeMat (Add) passed!")
}

// Test getRow
test_getRow :: proc() {
	fmt.println("Testing getRow...")

	// 2x3 matrix
	a := makeConstNodeMatrix(2, 3, []f32{1, 2, 3, 4, 5, 6})

	row0 := getRow(&a, 0)
	row1 := getRow(&a, 1)

	// Row 0 should be [1, 2, 3]
	assert(row0.r == 1 && row0.c == 3)
	assert(eval(&row0.data[0], nil) == 1)
	assert(eval(&row0.data[1], nil) == 2)
	assert(eval(&row0.data[2], nil) == 3)

	// Row 1 should be [4, 5, 6]
	assert(row1.r == 1 && row1.c == 3)
	assert(eval(&row1.data[0], nil) == 4)
	assert(eval(&row1.data[1], nil) == 5)
	assert(eval(&row1.data[2], nil) == 6)

	fmt.println("  getRow passed!")
}

// Test getCol
test_getCol :: proc() {
	fmt.println("Testing getCol...")

	// 2x3 matrix
	a := makeConstNodeMatrix(2, 3, []f32{1, 2, 3, 4, 5, 6})

	col0 := getCol(&a, 0)
	col1 := getCol(&a, 1)
	col2 := getCol(&a, 2)

	// Col 0 should be [1, 4]
	assert(col0.r == 2 && col0.c == 1)
	assert(eval(&col0.data[0], nil) == 1)
	assert(eval(&col0.data[1], nil) == 4)

	// Col 1 should be [2, 5]
	assert(col1.r == 2 && col1.c == 1)
	assert(eval(&col1.data[0], nil) == 2)
	assert(eval(&col1.data[1], nil) == 5)

	// Col 2 should be [3, 6]
	assert(col2.r == 2 && col2.c == 1)
	assert(eval(&col2.data[0], nil) == 3)
	assert(eval(&col2.data[1], nil) == 6)

	fmt.println("  getCol passed!")
}

// Test dotNodeMat
test_dotNodeMat :: proc() {
	fmt.println("Testing dotNodeMat...")

	// Two vectors of length 3
	a := makeConstNodeMatrix(1, 3, []f32{1, 2, 3})
	b := makeConstNodeMatrix(1, 3, []f32{4, 5, 6})

	result := dotNodeMat(&a, &b)

	// Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	assert(eval(result, nil) == 32)

	fmt.println("  dotNodeMat passed!")
}

// Test multNodeMat - basic matrix multiplication
test_multNodeMat_basic :: proc() {
	fmt.println("Testing multNodeMat (basic)...")

	// A: 2x3 matrix
	// [1, 2, 3]
	// [4, 5, 6]
	a := makeConstNodeMatrix(2, 3, []f32{1, 2, 3, 4, 5, 6})

	// B: 3x2 matrix
	// [7,  8]
	// [9,  10]
	// [11, 12]
	b := makeConstNodeMatrix(3, 2, []f32{7, 8, 9, 10, 11, 12})

	result := multNodeMat(&a, &b)

	// Expected result: 2x2 matrix
	// [1*7+2*9+3*11,  1*8+2*10+3*12]  = [58, 64]
	// [4*7+5*9+6*11,  4*8+5*10+6*12]  = [139, 154]
	assert(result.r == 2 && result.c == 2)
	assert(eval(&result.data[0], nil) == 58) // [0,0]
	assert(eval(&result.data[1], nil) == 64) // [0,1]
	assert(eval(&result.data[2], nil) == 139) // [1,0]
	assert(eval(&result.data[3], nil) == 154) // [1,1]

	fmt.println("  multNodeMat (basic) passed!")
}

// Test multNodeMat - identity matrix
test_multNodeMat_identity :: proc() {
	fmt.println("Testing multNodeMat (identity)...")

	// A: 2x2 matrix
	// [1, 2]
	// [3, 4]
	a := makeConstNodeMatrix(2, 2, []f32{1, 2, 3, 4})

	// I: 2x2 identity matrix
	// [1, 0]
	// [0, 1]
	identity := makeConstNodeMatrix(2, 2, []f32{1, 0, 0, 1})

	result := multNodeMat(&a, &identity)

	// A * I = A
	assert(result.r == 2 && result.c == 2)
	assert(eval(&result.data[0], nil) == 1)
	assert(eval(&result.data[1], nil) == 2)
	assert(eval(&result.data[2], nil) == 3)
	assert(eval(&result.data[3], nil) == 4)

	fmt.println("  multNodeMat (identity) passed!")
}

// Test multNodeMat - vector-matrix multiplication
test_multNodeMat_vector :: proc() {
	fmt.println("Testing multNodeMat (vector)...")

	// Row vector: 1x3
	v := makeConstNodeMatrix(1, 3, []f32{1, 2, 3})

	// Matrix: 3x2
	m := makeConstNodeMatrix(3, 2, []f32{1, 2, 3, 4, 5, 6})

	result := multNodeMat(&v, &m)

	// Expected: 1x2 vector
	// [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
	assert(result.r == 1 && result.c == 2)
	assert(eval(&result.data[0], nil) == 22)
	assert(eval(&result.data[1], nil) == 28)

	fmt.println("  multNodeMat (vector) passed!")
}

// Run all NodeMatrix tests
test_node_matrix_all :: proc() {
	fmt.println("\n=== Running NodeMatrix Tests ===")

	test_binaryOpNodeMat_add()
	test_binaryOpNodeMat_mul()
	test_reduceNodeMat()
	test_getRow()
	test_getCol()
	test_dotNodeMat()
	test_multNodeMat_basic()
	test_multNodeMat_identity()
	test_multNodeMat_vector()

	fmt.println("=== All NodeMatrix Tests Passed! ===\n")
}

assert_mat_approx_equal :: proc(a, b: []f32, tol: f32 = 1e-5) {
	assert(len(a) == len(b), "length mismatch")
	for i in 0 ..< len(a) {
		diff := a[i] - b[i]
		if diff < 0 {diff = -diff}
		assert(diff < tol, "value mismatch")
	}
}

test_grad_matrix_elementwise :: proc() {
	fmt.println("Testing MatNode elementwise ops vs NodeMatrix...")

	a_data := []f32{1, 2, 3, 4}
	b_data := []f32{5, 6, 7, 8}

	var_shapes := []MatShape{{2, 2}, {2, 2}}
	binding := make([]f32, 8)
	defer delete(binding)
	for i in 0 ..< 4 {
		binding[i] = a_data[i]
		binding[4 + i] = b_data[i]
	}

	a_mat := make_mat_var(0)
	b_mat := make_mat_var(1)

	a_node_mat := makeConstNodeMatrix(2, 2, a_data)
	b_node_mat := makeConstNodeMatrix(2, 2, b_data)

	ops := []MatOpType{.Add, .Sub, .Mul, .Div}
	scalar_ops := []OpType{.Add, .Sub, .Mul, .Div}

	for op_idx in 0 ..< len(ops) {
		mat_node_result := make_mat_op(ops[op_idx], a_mat, b_mat)
		val, shape := eval_mat(mat_node_result, binding, var_shapes)
		defer delete(val)

		node_mat_result := binaryOpNodeMat(scalar_ops[op_idx], &a_node_mat, &b_node_mat)
		expected := make([]f32, 4)
		defer delete(expected)
		for i in 0 ..< 4 {
			expected[i] = eval(&node_mat_result.data[i], nil)
		}

		assert_mat_approx_equal(val, expected)
	}

	fmt.println("  MatNode elementwise ops passed!")
}

test_grad_matrix_matmul :: proc() {
	fmt.println("Testing MatNode matmul vs NodeMatrix...")

	a_data := []f32{1, 2, 3, 4, 5, 6}
	b_data := []f32{7, 8, 9, 10, 11, 12}

	var_shapes := []MatShape{{2, 3}, {3, 2}}
	binding := make([]f32, 12)
	defer delete(binding)
	for i in 0 ..< 6 {
		binding[i] = a_data[i]
		binding[6 + i] = b_data[i]
	}

	a_mat := make_mat_var(0)
	b_mat := make_mat_var(1)
	mat_node_result := make_mat_op(.MatMul, a_mat, b_mat)

	val, shape := eval_mat(mat_node_result, binding, var_shapes)
	defer delete(val)

	a_node_mat := makeConstNodeMatrix(2, 3, a_data)
	b_node_mat := makeConstNodeMatrix(3, 2, b_data)
	node_mat_result := multNodeMat(&a_node_mat, &b_node_mat)

	expected := make([]f32, 4)
	defer delete(expected)
	for i in 0 ..< 4 {
		expected[i] = eval(&node_mat_result.data[i], nil)
	}

	assert(shape.r == 2 && shape.c == 2)
	assert_mat_approx_equal(val, expected)

	fmt.println("  MatNode matmul passed!")
}

test_grad_matrix_reduce_sum :: proc() {
	fmt.println("Testing MatNode reduce_sum vs NodeMatrix...")

	a_data := []f32{1, 2, 3, 4, 5, 6}
	var_shapes := []MatShape{{2, 3}}
	binding := make([]f32, 6)
	defer delete(binding)
	for i in 0 ..< 6 {
		binding[i] = a_data[i]
	}

	a_mat := make_mat_var(0)
	reduce_node := make_mat_op(.ReduceSum, a_mat)

	val, shape := eval_mat(reduce_node, binding, var_shapes)
	defer delete(val)

	a_node_mat := makeConstNodeMatrix(2, 3, a_data)
	scalar_result := reduceNodeMat(.Add, &a_node_mat)
	expected := eval(scalar_result, nil)

	assert(shape.r == 1 && shape.c == 1)
	assert(len(val) == 1)
	diff := val[0] - expected
	if diff < 0 {diff = -diff}
	assert(diff < 1e-5)

	fmt.println("  MatNode reduce_sum passed!")
}

test_grad_matrix_forward_grad :: proc() {
	fmt.println("Testing MatNode forward grad vs NodeMatrix...")

	a_data := []f32{1, 2, 3, 4}
	b_data := []f32{5, 6, 7, 8}

	var_shapes := []MatShape{{2, 2}, {2, 2}}
	binding := make([]f32, 8)
	defer delete(binding)
	for i in 0 ..< 4 {
		binding[i] = a_data[i]
		binding[4 + i] = b_data[i]
	}

	a_mat := make_mat_var(0)
	b_mat := make_mat_var(1)
	mul_node := make_mat_op(.Mul, a_mat, b_mat)
	sum_node := make_mat_op(.ReduceSum, mul_node)

	vg := eval_mat_grad_forward(sum_node, binding, var_shapes, 0)
	defer delete(vg.val)
	defer delete(vg.grad)

	a_node_mat: NodeMatrix
	a_node_mat.r = 2
	a_node_mat.c = 2
	a_node_data := make([]Node, 4)
	for i in 0 ..< 4 {
		a_node_data[i] = i
	}
	a_node_mat.data = a_node_data

	b_node_mat: NodeMatrix
	b_node_mat.r = 2
	b_node_mat.c = 2
	b_node_data := make([]Node, 4)
	for i in 0 ..< 4 {
		b_node_data[i] = 4 + i
	}
	b_node_mat.data = b_node_data

	mul_result := binaryOpNodeMat(.Mul, &a_node_mat, &b_node_mat)
	scalar_sum := reduceNodeMat(.Add, &mul_result)

	scalar_grads := make([]f32, 8)
	defer delete(scalar_grads)
	scalar_val := eval_grad_reverse(scalar_sum, binding, scalar_grads)

	diff := vg.val[0] - scalar_val
	if diff < 0 {diff = -diff}
	assert(diff < 1e-5)

	expected_forward_grad: f32 = 0
	for i in 0 ..< 4 {
		expected_forward_grad += scalar_grads[i]
	}

	grad_diff := vg.grad[0] - expected_forward_grad
	if grad_diff < 0 {grad_diff = -grad_diff}
	assert(grad_diff < 1e-5)

	fmt.println("  MatNode forward grad passed!")
}

test_grad_matrix_reverse_grad :: proc() {
	fmt.println("Testing MatNode reverse grad vs NodeMatrix...")

	a_data := []f32{1, 2, 3, 4}
	b_data := []f32{5, 6, 7, 8}

	var_shapes := []MatShape{{2, 2}, {2, 2}}
	binding := make([]f32, 8)
	defer delete(binding)
	for i in 0 ..< 4 {
		binding[i] = a_data[i]
		binding[4 + i] = b_data[i]
	}

	a_mat := make_mat_var(0)
	b_mat := make_mat_var(1)
	mul_node := make_mat_op(.Mul, a_mat, b_mat)
	sum_node := make_mat_op(.ReduceSum, mul_node)

	mat_grads := make([]f32, 8)
	defer delete(mat_grads)
	mat_val := eval_mat_grad_reverse(sum_node, binding, var_shapes, mat_grads)
	defer delete(mat_val)

	a_node_mat: NodeMatrix
	a_node_mat.r = 2
	a_node_mat.c = 2
	a_node_data := make([]Node, 4)
	for i in 0 ..< 4 {
		a_node_data[i] = i
	}
	a_node_mat.data = a_node_data

	b_node_mat: NodeMatrix
	b_node_mat.r = 2
	b_node_mat.c = 2
	b_node_data := make([]Node, 4)
	for i in 0 ..< 4 {
		b_node_data[i] = 4 + i
	}
	b_node_mat.data = b_node_data

	mul_result := binaryOpNodeMat(.Mul, &a_node_mat, &b_node_mat)
	scalar_sum := reduceNodeMat(.Add, &mul_result)

	scalar_grads := make([]f32, 8)
	defer delete(scalar_grads)
	scalar_val := eval_grad_reverse(scalar_sum, binding, scalar_grads)

	diff := mat_val[0] - scalar_val
	if diff < 0 {diff = -diff}
	assert(diff < 1e-5)

	assert_mat_approx_equal(mat_grads, scalar_grads)

	fmt.println("  MatNode reverse grad passed!")
}

test_grad_matrix_matmul_grad :: proc() {
	fmt.println("Testing MatNode matmul gradients vs NodeMatrix...")

	A_data := []f32{1, 2, 3, 4, 5, 6}
	x_data := []f32{1, 2, 3}

	var_shapes := []MatShape{
		{2, 3},
		{3, 1},
	}

	total_size := 6 + 3
	binding := make([]f32, total_size)
	defer delete(binding)

	for i in 0 ..< 6 {
		binding[i] = A_data[i]
	}
	for i in 0 ..< 3 {
		binding[6 + i] = x_data[i]
	}

	A_mat := make_mat_var(0)
	x_mat := make_mat_var(1)

	Ax := make_mat_op(.MatMul, A_mat, x_mat)
	loss := make_mat_op(.ReduceSum, Ax)

	mat_grads := make([]f32, total_size)
	defer delete(mat_grads)
	mat_val := eval_mat_grad_reverse(loss, binding, var_shapes, mat_grads)
	defer delete(mat_val)

	A_node, next_idx := makeVarMat(2, 3, 0)
	x_node: NodeMatrix
	x_node, next_idx = makeVarMat(3, 1, next_idx)

	Ax_scalar := multNodeMat(&A_node, &x_node)
	loss_scalar := reduceNodeMat(.Add, &Ax_scalar)

	scalar_grads := make([]f32, total_size)
	defer delete(scalar_grads)
	scalar_val := eval_grad_reverse(loss_scalar, binding, scalar_grads)

	val_diff := mat_val[0] - scalar_val
	if val_diff < 0 {val_diff = -val_diff}
	assert(val_diff < 1e-4)

	assert_mat_approx_equal(mat_grads, scalar_grads, 1e-4)

	fmt.println("  MatNode matmul gradients passed!")
}

test_grad_matrix_squared :: proc() {
	fmt.println("Testing MatNode x*x gradients vs NodeMatrix...")

	a_data := []f32{2, 3, 4, 5}

	var_shapes := []MatShape{{2, 2}}
	binding := make([]f32, 4)
	defer delete(binding)
	for i in 0 ..< 4 {
		binding[i] = a_data[i]
	}

	a_mat := make_mat_var(0)
	sq := make_mat_op(.Mul, a_mat, a_mat)
	loss := make_mat_op(.ReduceSum, sq)

	mat_grads := make([]f32, 4)
	defer delete(mat_grads)
	mat_val := eval_mat_grad_reverse(loss, binding, var_shapes, mat_grads)
	defer delete(mat_val)

	a_node_mat: NodeMatrix
	a_node_mat.r = 2
	a_node_mat.c = 2
	a_node_data := make([]Node, 4)
	for i in 0 ..< 4 {
		a_node_data[i] = i
	}
	a_node_mat.data = a_node_data

	sq_scalar := binaryOpNodeMat(.Mul, &a_node_mat, &a_node_mat)
	loss_scalar := reduceNodeMat(.Add, &sq_scalar)

	scalar_grads := make([]f32, 4)
	defer delete(scalar_grads)
	scalar_val := eval_grad_reverse(loss_scalar, binding, scalar_grads)

	val_diff := mat_val[0] - scalar_val
	if val_diff < 0 {val_diff = -val_diff}
	assert(val_diff < 1e-4)

	assert_mat_approx_equal(mat_grads, scalar_grads, 1e-4)

	fmt.println("  MatNode x*x gradients passed!")
}

test_grad_matrix_exp_relu :: proc() {
	fmt.println("Testing MatNode exp/relu gradients...")

	var_shapes := []MatShape{{2, 2}}
	binding := []f32{-2, -0.5, 1, 3}

	x := make_mat_var(0)
	exp_node := make_mat_op(.Exp, x)
	relu_node := make_mat_op(.ReLU, x)

	exp_loss := make_mat_op(.ReduceSum, exp_node)
	relu_loss := make_mat_op(.ReduceSum, relu_node)

	exp_grads := make([]f32, len(binding))
	relu_grads := make([]f32, len(binding))
	defer {
		delete(exp_grads)
		delete(relu_grads)
	}

	exp_val := eval_mat_grad_reverse(exp_loss, binding, var_shapes, exp_grads)
	relu_val := eval_mat_grad_reverse(relu_loss, binding, var_shapes, relu_grads)
	defer {
		delete(exp_val)
		delete(relu_val)
	}

	tol: f32 = 1e-4
	expected_exp := []f32{
		f32(math.exp(f64(-2))),
		f32(math.exp(f64(-0.5))),
		f32(math.exp(f64(1))),
		f32(math.exp(f64(3))),
	}
	assert_mat_approx_equal(exp_grads, expected_exp, tol)

	expected_relu := []f32{0, 0, 1, 1}
	assert_mat_approx_equal(relu_grads, expected_relu, tol)

	expected_relu_sum: f32 = 4
	diff := relu_val[0] - expected_relu_sum
	if diff < 0 {diff = -diff}
	assert(diff < tol)

	assert(exp_val[0] > 0)
	fmt.println("  MatNode exp/relu gradients passed!")
}

test_grad_matrix_linear_model :: proc() {
	fmt.println("Testing MatNode linear model (Ax+b) vs NodeMatrix...")

	DIM_IN :: 3
	DIM_OUT :: 2

	A_data := []f32{1, 2, 3, 4, 5, 6}
	b_data := []f32{0.5, -0.5}
	x_data := []f32{1, 2, 3}
	y_data := []f32{10, 20}

	var_shapes := []MatShape{
		{DIM_OUT, DIM_IN},
		{DIM_OUT, 1},
		{DIM_IN, 1},
		{DIM_OUT, 1},
	}

	total_size := DIM_OUT * DIM_IN + DIM_OUT + DIM_IN + DIM_OUT
	binding := make([]f32, total_size)
	defer delete(binding)

	offset := 0
	for i in 0 ..< DIM_OUT * DIM_IN {
		binding[offset + i] = A_data[i]
	}
	offset += DIM_OUT * DIM_IN
	for i in 0 ..< DIM_OUT {
		binding[offset + i] = b_data[i]
	}
	offset += DIM_OUT
	for i in 0 ..< DIM_IN {
		binding[offset + i] = x_data[i]
	}
	offset += DIM_IN
	for i in 0 ..< DIM_OUT {
		binding[offset + i] = y_data[i]
	}

	A_mat := make_mat_var(0)
	b_mat := make_mat_var(1)
	x_mat := make_mat_var(2)
	y_mat := make_mat_var(3)

	Ax := make_mat_op(.MatMul, A_mat, x_mat)
	pred := make_mat_op(.Add, Ax, b_mat)
	diff := make_mat_op(.Sub, pred, y_mat)
	sq := make_mat_op(.Mul, diff, diff)
	loss := make_mat_op(.ReduceSum, sq)

	mat_grads := make([]f32, total_size)
	defer delete(mat_grads)
	mat_val := eval_mat_grad_reverse(loss, binding, var_shapes, mat_grads)
	defer delete(mat_val)

	A_node, next_idx := makeVarMat(DIM_OUT, DIM_IN, 0)
	b_node: NodeMatrix
	x_node: NodeMatrix
	y_node: NodeMatrix
	b_node, next_idx = makeVarMat(DIM_OUT, 1, next_idx)
	x_node, next_idx = makeVarMat(DIM_IN, 1, next_idx)
	y_node, next_idx = makeVarMat(DIM_OUT, 1, next_idx)

	Ax_scalar := multNodeMat(&A_node, &x_node)
	pred_scalar := binaryOpNodeMat(.Add, &Ax_scalar, &b_node)
	diff_scalar := binaryOpNodeMat(.Sub, &pred_scalar, &y_node)
	sq_scalar := binaryOpNodeMat(.Mul, &diff_scalar, &diff_scalar)
	loss_scalar := reduceNodeMat(.Add, &sq_scalar)

	scalar_grads := make([]f32, total_size)
	defer delete(scalar_grads)
	scalar_val := eval_grad_reverse(loss_scalar, binding, scalar_grads)

	val_diff := mat_val[0] - scalar_val
	if val_diff < 0 {val_diff = -val_diff}
	assert(val_diff < 1e-4)

	assert_mat_approx_equal(mat_grads, scalar_grads, 1e-4)

	fmt.println("  MatNode linear model passed!")
}

// Checks that grad_matrix and matrix_of_single produce identical gradients on a
// small fixed Ax+b MSE problem where exact values can be hand-verified.
//
// A (2x3):  [[1,2,3],[4,5,6]]
// b (2x1):  [0.5, -0.5]
// x (3x1):  [1, 2, 3]
// y (2x1):  [10, 20]
//
// pred = Ax+b = [1+4+9+0.5, 4+10+18-0.5] = [14.5, 31.5]
// diff = pred-y = [4.5, 11.5]
// loss = sum(diff^2) = 4.5^2 + 11.5^2 = 20.25 + 132.25 = 152.5
//
// d(loss)/d(diff_i) = 2*diff_i  =>  [9, 23]
// d(loss)/d(b_i)    = 2*diff_i  =>  [9, 23]
// d(loss)/d(x_j)    = sum_i 2*diff_i * A_ij
//   j=0: 2*4.5*1 + 2*11.5*4 = 9 + 92 = 101
//   j=1: 2*4.5*2 + 2*11.5*5 = 18 + 115 = 133
//   j=2: 2*4.5*3 + 2*11.5*6 = 27 + 138 = 165
// d(loss)/d(A_ij) = 2*diff_i * x_j
//   row0: [9*1, 9*2, 9*3]     = [9,  18, 27]
//   row1: [23*1, 23*2, 23*3]  = [23, 46, 69]
test_grad_matrix_vs_scalar_gradients :: proc() {
	fmt.println("Testing grad_matrix vs matrix_of_single gradients (fixed 2x3 Ax+b)...")

	A_data := []f32{1, 2, 3, 4, 5, 6}
	b_data := []f32{0.5, -0.5}
	x_data := []f32{1, 2, 3}
	y_data := []f32{10, 20}

	// layout: [A(6), b(2), x(3), y(2)]  total=13
	binding := []f32{1, 2, 3, 4, 5, 6,  0.5, -0.5,  1, 2, 3,  10, 20}

	expected_loss: f32 = 152.5
	expected_grads := []f32{
		9, 18, 27, 23, 46, 69,  // dA
		9, 23,                  // db
		101, 133, 165,          // dx
		0, 0,                   // dy (not checked — target is fixed)
	}

	var_shapes := []MatShape{{2, 3}, {2, 1}, {3, 1}, {2, 1}}
	tol: f32 = 1e-4

	// --- grad_matrix ---
	A_mat  := make_mat_var(0); b_mat := make_mat_var(1)
	x_mat  := make_mat_var(2); y_mat := make_mat_var(3)
	diff_m := make_mat_op(.Sub, make_mat_op(.Add, make_mat_op(.MatMul, A_mat, x_mat), b_mat), y_mat)
	loss_m := make_mat_op(.ReduceSum, make_mat_op(.Mul, diff_m, diff_m))

	mat_grads := make([]f32, len(binding)); defer delete(mat_grads)
	mat_val   := eval_mat_grad_reverse(loss_m, binding, var_shapes, mat_grads)
	defer delete(mat_val)

	loss_diff := mat_val[0] - expected_loss
	if loss_diff < 0 { loss_diff = -loss_diff }
	assert(loss_diff < tol, "grad_matrix: loss mismatch")
	assert_mat_approx_equal(mat_grads[:11], expected_grads[:11], tol)

	// --- matrix_of_single ---
	A_node, next := makeVarMat(2, 3, 0)
	b_node: NodeMatrix; b_node, next = makeVarMat(2, 1, next)
	x_node: NodeMatrix; x_node, next = makeVarMat(3, 1, next)
	y_node: NodeMatrix; y_node, next = makeVarMat(2, 1, next)

	Ax     := multNodeMat(&A_node, &x_node)
	pred   := binaryOpNodeMat(.Add, &Ax, &b_node)
	diff   := binaryOpNodeMat(.Sub, &pred, &y_node)
	sq     := binaryOpNodeMat(.Mul, &diff, &diff)
	loss_s := reduceNodeMat(.Add, &sq)

	scalar_grads := make([]f32, len(binding)); defer delete(scalar_grads)
	scalar_loss  := eval_grad_reverse(loss_s, binding, scalar_grads)

	loss_diff = scalar_loss - expected_loss
	if loss_diff < 0 { loss_diff = -loss_diff }
	assert(loss_diff < tol, "matrix_of_single: loss mismatch")
	assert_mat_approx_equal(scalar_grads[:11], expected_grads[:11], tol)

	// --- the two approaches must agree with each other ---
	assert_mat_approx_equal(mat_grads, scalar_grads, tol)

	fmt.println("  grad_matrix vs matrix_of_single gradients passed!")
}

test_grad_matrix :: proc() {
	fmt.println("\n=== Running MatNode vs NodeMatrix Comparison Tests ===")

	test_grad_matrix_elementwise()
	test_grad_matrix_matmul()
	test_grad_matrix_reduce_sum()
	test_grad_matrix_forward_grad()
	test_grad_matrix_reverse_grad()
	test_grad_matrix_matmul_grad()
	test_grad_matrix_squared()
	test_grad_matrix_exp_relu()
	test_grad_matrix_linear_model()
	test_grad_matrix_vs_scalar_gradients()

	fmt.println("=== All MatNode Tests Passed! ===\n")
}

// Timing comparison: grad_matrix (MatNode reverse) vs matrix_of_single (scalar Node reverse)
// Both evaluate Ax+b MSE loss on the same linear model.
time_grad_matrix_vs_scalar :: proc(dim_in, dim_out, num_points, runs: int) {
	xs := make([dynamic][]f32, num_points)
	ys := make([dynamic][]f32, num_points)
	defer {
		for i in 0 ..< num_points { delete(xs[i]); delete(ys[i]) }
		delete(xs); delete(ys)
	}

	true_A := make([]f32, dim_out * dim_in)
	true_b := make([]f32, dim_out)
	defer { delete(true_A); delete(true_b) }

	for i in 0 ..< dim_out * dim_in { true_A[i] = rand.float32_range(-2, 2) }
	for i in 0 ..< dim_out         { true_b[i] = rand.float32_range(-1, 1) }

	for i in 0 ..< num_points {
		x := make([]f32, dim_in)
		y := make([]f32, dim_out)
		for j in 0 ..< dim_in { x[j] = rand.float32_range(-5, 5) }
		for i2 in 0 ..< dim_out {
			sum := true_b[i2]
			for j in 0 ..< dim_in { sum += true_A[i2 * dim_in + j] * x[j] }
			y[i2] = sum
		}
		xs[i] = x
		ys[i] = y
	}

	num_trainable := dim_out * dim_in + dim_out
	total_size    := num_trainable + dim_in + dim_out
	x_start       := num_trainable
	y_start        := x_start + dim_in

	// --- matrix_of_single (scalar) setup ---
	A_node, next_idx := makeVarMat(dim_out, dim_in, 0)
	b_node: NodeMatrix
	x_node_s: NodeMatrix
	y_node_s: NodeMatrix
	b_node,    next_idx = makeVarMat(dim_out, 1, next_idx)
	x_node_s,  next_idx = makeVarMat(dim_in, 1, next_idx)
	y_node_s,  next_idx = makeVarMat(dim_out, 1, next_idx)

	Ax_s    := multNodeMat(&A_node, &x_node_s)
	pred_s  := binaryOpNodeMat(.Add, &Ax_s, &b_node)
	diff_s  := binaryOpNodeMat(.Sub, &pred_s, &y_node_s)
	sq_s    := binaryOpNodeMat(.Mul, &diff_s, &diff_s)
	loss_s  := reduceNodeMat(.Add, &sq_s)

	binding_s := make([]f32, total_size)
	defer delete(binding_s)

	scalar_stats := time_train_epoch(loss_s, binding_s, xs[:], ys[:], x_start, y_start, num_trainable, 0, runs)

	// --- grad_matrix (MatNode) setup ---
	var_shapes := []MatShape{{dim_out, dim_in}, {dim_out, 1}, {dim_in, 1}, {dim_out, 1}}

	A_mat  := make_mat_var(0)
	b_mat  := make_mat_var(1)
	x_mat  := make_mat_var(2)
	y_mat  := make_mat_var(3)

	Ax_m   := make_mat_op(.MatMul, A_mat, x_mat)
	pred_m := make_mat_op(.Add, Ax_m, b_mat)
	diff_m := make_mat_op(.Sub, pred_m, y_mat)
	sq_m   := make_mat_op(.Mul, diff_m, diff_m)
	loss_m := make_mat_op(.ReduceSum, sq_m)

	binding_m := make([]f32, total_size)
	defer delete(binding_m)

	// --- verify outputs match on first data point ---
	for j in 0 ..< dim_in  { binding_s[x_start + j]   = xs[0][j]; binding_m[x_start + j]   = xs[0][j] }
	for j in 0 ..< dim_out { binding_s[y_start + j]   = ys[0][j]; binding_m[y_start + j]   = ys[0][j] }

	scalar_grads := make([]f32, total_size); defer delete(scalar_grads)
	mat_grads    := make([]f32, total_size); defer delete(mat_grads)

	scalar_loss := eval_grad_reverse(loss_s, binding_s, scalar_grads)
	mat_loss_v  := eval_mat_grad_reverse(loss_m, binding_m, var_shapes, mat_grads)
	defer delete(mat_loss_v)

	tol: f32 = 1e-4
	loss_diff := scalar_loss - mat_loss_v[0]
	if loss_diff < 0 { loss_diff = -loss_diff }
	assert(loss_diff < tol, "loss value mismatch between scalar and mat approaches")
	assert_mat_approx_equal(scalar_grads, mat_grads, tol)
	fmt.printf("  [verify] outputs match (tol=%.0e)\n", tol)

	mat_stats := time_train_epoch_mat(loss_m, binding_m, var_shapes, xs[:], ys[:], 2, 3, num_trainable, 0, runs)

	fmt.printf(
		"\n=== grad_matrix vs matrix_of_single (%dx%d, %d pts, %d runs) ===\n",
		dim_out, dim_in, num_points, runs,
	)
	fmt.printf("  matrix_of_single: mean=%.3fms  cv=%.4f\n", scalar_stats.mean_ms, scalar_stats.cv)
	fmt.printf("  grad_matrix:      mean=%.3fms  cv=%.4f\n", mat_stats.mean_ms, mat_stats.cv)
	fmt.printf("  speedup (grad_matrix / scalar): %.2fx\n\n", scalar_stats.mean_ms / mat_stats.mean_ms)
}

time_diff_vm :: proc(instrs: []DiffInstr, binding: []f32, runs: int) -> stats {
	sw: time.Stopwatch

	m, squares: f64
	n: int

	inner_runs := 1000

	grads := make([]f32, len(binding))
	defer delete(grads)

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)
		for _ in 0 ..< inner_runs {
			mem := make([]f32, len(binding))
			copy(mem, binding)
			for i in 0 ..< len(grads) { grads[i] = 0 }
			_, tape := diff_sim_forward(instrs, mem)
			diff_sim_backward(tape, grads)
			delete(mem)
			delete(tape)
		}
		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := next_mean(m, n, x)
			squares = next_squares(squares, m, m_new, x)
			m = m_new
		}
	}

	variance := squares / f64(n)
	return stats{m, variance, math.sqrt(variance) / m}
}

time_sim_reverse :: proc(compiled: CompiledReverse, binding: []f32, runs: int) -> stats {
	sw: time.Stopwatch

	m, squares: f64
	n: int

	inner_runs := 1000

	mem := make([]f32, compiled.total_mem_size)
	defer delete(mem)

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)
		for _ in 0 ..< inner_runs {
			copy(mem[:compiled.num_bindings], binding)
			for i in compiled.num_bindings ..< compiled.total_mem_size { mem[i] = 0 }
			simulate(compiled.instrs[:], mem)
		}
		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := next_mean(m, n, x)
			squares = next_squares(squares, m, m_new, x)
			m = m_new
		}
	}

	variance := squares / f64(n)
	return stats{m, variance, math.sqrt(variance) / m}
}

time_diff_vm_vs_compiled_reverse :: proc() {
	ast, vars := parse("5*z+1-3+4*2/5+9-11*y/7")
	vals := map[string]f32{"x" = 2, "y" = 6, "z" = 10}
	binding := binding_to_arr(vals, vars)

	diff_instrs := compile_diff_vm(ast)
	compiled := compile_reverse(ast, len(binding))

	// correctness: use eval_grad_forward_all as reference (handles all ops including div)
	ref_val, ref_grads := eval_grad_forward_all(ast, binding)
	defer delete(ref_grads)

	diff_mem := make([]f32, len(binding))
	copy(diff_mem, binding)
	diff_grads := make([]f32, len(binding))
	defer delete(diff_grads)
	defer delete(diff_mem)
	diff_val, tape := diff_sim_forward(diff_instrs[:], diff_mem)
	diff_sim_backward(tape, diff_grads)
	delete(tape)

	comp_mem := make([]f32, compiled.total_mem_size)
	defer delete(comp_mem)
	copy(comp_mem[:compiled.num_bindings], binding)
	simulate(compiled.instrs[:], comp_mem)

	tol: f32 = 1e-4
	assert(abs(diff_val - ref_val) < tol, "diff_vm value mismatch")
	for i in 0 ..< len(binding) {
		assert(abs(diff_grads[i] - ref_grads[i]) < tol, "diff_vm grad mismatch")
	}

	NUM_RUNS := 100
	interp_stats := time_grad_reverse(ast, binding, NUM_RUNS)
	diff_vm_stats := time_diff_vm(diff_instrs[:], binding, NUM_RUNS)
	compiled_stats := time_sim_reverse(compiled, binding, NUM_RUNS)

	fmt.printf("\n=== diff_vm vs compiled_reverse vs interp_reverse (%s) ===\n", "5*z+1-3+4*2/5+9-11*y/7")
	fmt.printf("  interp_reverse:    mean=%.3fms  cv=%.4f\n", interp_stats.mean, interp_stats.cv)
	fmt.printf("  diff_vm:           mean=%.3fms  cv=%.4f\n", diff_vm_stats.mean, diff_vm_stats.cv)
	fmt.printf("  compiled_reverse:  mean=%.3fms  cv=%.4f\n", compiled_stats.mean, compiled_stats.cv)
}
