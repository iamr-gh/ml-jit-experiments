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


time_base_test :: proc() {
	fmt.println("Basic time test, forward eval vs sim")

	ast, vars := parse("5*3+1-3+4*2/5+9-11*16/7")
	instrs := compile_forward(ast)
	stack_instrs := compile_forward_stack(ast)

	eval_res := eval(ast, nil)
	sim_res := simulate(instrs[:])
	assert(eval_res == sim_res)

	NUM_RUNS := 1000
	eval_stats := time_eval(ast, nil, NUM_RUNS)
	sim_stats := time_sim(instrs[:], NUM_RUNS)
	sim_stack_stats := time_sim_stack(stack_instrs[:], NUM_RUNS)

	fmt.printf("Tree Eval stats: {}\n", eval_stats)
	fmt.printf("Sim(register) stats: {}\n", sim_stats)
	fmt.printf("Sim(stack) stats: {}\n", sim_stack_stats)


}

eval_vm_basic_test :: proc() {
	ast, vars := parse("5*3+1-3+4*2")
	print_ast(ast, vars)

	// don't optimize consts
	instrs := compile_forward(ast)
	fmt.printf("Instrs: %#v\n", instrs)

	stack_instrs := compile_forward_stack(ast)

	eval_res := eval(ast, nil)
	sim_res := simulate(instrs[:])
	sim_stack_res := simulate_stack(stack_instrs[:])

	fmt.printf("Eval res: %f, sim_res: %f, stack_res: %f\n", eval_res, sim_res, sim_stack_res)

	assert(eval_res == sim_res)
	assert(eval_res == sim_stack_res)
}

basic_test :: proc() {
	ast, vars := parse("x*3/2+1")
	fmt.printf("vars: %v\n", vars)
	print_ast(ast, vars)

	constant_prop(ast)
	print_ast(ast, vars)

	val := eval(ast, binding_to_arr(map[string]f32{"x" = 2, "y" = 2}, vars))
	fmt.printf("%v\n", val)

	free_all(context.temp_allocator)

	ast2, vars2 := parse("x+y*y")
	print_ast(ast2, vars2)
	vals := map[string]f32 {
		"x" = 2,
		"y" = 6,
	}

	bound := binding_to_arr(vals, vars2)

	dfdx := eval_grad(ast2, bound, vars["x"])
	dfdy := eval_grad(ast2, bound, vars["y"])

	assert(dfdx.val == 38)
	assert(dfdy.val == 38)
	assert(dfdx.grad == 1)
	assert(dfdy.grad == 12)

	free_all(context.temp_allocator)
}
