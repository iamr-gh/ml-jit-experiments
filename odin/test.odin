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
