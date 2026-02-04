#+feature dynamic-literals
package main
import "core:fmt"
import "core:math"
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

	reverse_val, grads := eval_grad_reverse(ast2, bound)
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

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)
		for _ in 0 ..< inner_runs {
			eval_grad_reverse(node, binding)
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

test_grad_matrix :: proc() {

}
