package main
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:time"

// let's train something from scratch

// say I have the expression y = Ax + b
// and then I want to train on A and b

makeVarMat :: proc(r: int, c: int, start_idx: int) -> (NodeMatrix, int) {
	out_idx := (r * c) + start_idx
	out_data: [dynamic]Node

	resize(&out_data, r * c)
	for i in 0 ..< (r * c) {
		out_data[i] = start_idx + i
	}

	return NodeMatrix{r, c, out_data[:]}, out_idx
}

TrainStats :: struct {
	mean_ms: f64,
	var_ms:  f64,
	cv:      f64,
}

time_train_epoch :: proc(
	loss_node: ^Node,
	binding: []f32,
	xs: [][]f32,
	ys: [][]f32,
	x_start: int,
	y_start: int,
	num_trainable: int,
	lr: f32,
	runs: int,
) -> TrainStats {
	sw: time.Stopwatch
	m, squares: f64
	n: int

	grads := make([]f32, len(binding))
	defer delete(grads)

	activation_cache: map[^Node]f32
	grad_cache: map[^Node]f32
	defer delete(activation_cache)
	defer delete(grad_cache)

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)

		for i in 0 ..< len(xs) {
			for j in 0 ..< len(xs[i]) {
				binding[x_start + j] = xs[i][j]
			}
			for j in 0 ..< len(ys[i]) {
				binding[y_start + j] = ys[i][j]
			}

			eval_grad_reverse(loss_node, binding, grads, &activation_cache, &grad_cache)

			for j in 0 ..< num_trainable {
				binding[j] -= lr * grads[j]
			}
		}

		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := m + ((x - m) / f64(n))
			squares = squares + ((x - m) * (x - m_new))
			m = m_new
		}
	}

	variance := squares / f64(n)
	return TrainStats{m, variance, math.sqrt(variance) / m}
}

time_train_epoch_compiled :: proc(
	compiled: CompiledReverse,
	mem: []f32,
	xs: [][]f32,
	ys: [][]f32,
	x_start: int,
	y_start: int,
	num_trainable: int,
	lr: f32,
	runs: int,
) -> TrainStats {
	sw: time.Stopwatch
	m, squares: f64
	n: int

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)

		for i in 0 ..< len(xs) {
			for j in 0 ..< len(xs[i]) {
				mem[x_start + j] = xs[i][j]
			}
			for j in 0 ..< len(ys[i]) {
				mem[y_start + j] = ys[i][j]
			}

			for j in 0 ..< compiled.total_mem_size - compiled.grad_offset {
				mem[compiled.grad_offset + j] = 0
			}

			simulate(compiled.instrs[:], mem)

			for j in 0 ..< num_trainable {
				mem[j] -= lr * mem[compiled.grad_offset + j]
			}
		}

		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := m + ((x - m) / f64(n))
			squares = squares + ((x - m) * (x - m_new))
			m = m_new
		}
	}

	variance := squares / f64(n)
	return TrainStats{m, variance, math.sqrt(variance) / m}
}

trainLinear :: proc() {
	DIM_IN :: 50
	DIM_OUT :: 50
	NUM_POINTS :: 1000

	true_A: [DIM_OUT][DIM_IN]f32
	true_b: [DIM_OUT]f32

	for i in 0 ..< DIM_OUT {
		for j in 0 ..< DIM_IN {
			true_A[i][j] = rand.float32_range(-2, 2)
		}
		true_b[i] = rand.float32_range(-1, 1)
	}

	xs: [dynamic][]f32
	ys: [dynamic][]f32

	for _ in 0 ..< NUM_POINTS {
		x := make([]f32, DIM_IN)
		for j in 0 ..< DIM_IN {
			x[j] = rand.float32_range(-5, 5)
		}

		y := make([]f32, DIM_OUT)
		for i in 0 ..< DIM_OUT {
			sum: f32 = true_b[i]
			for j in 0 ..< DIM_IN {
				sum += true_A[i][j] * x[j]
			}
			y[i] = sum
		}

		append(&xs, x)
		append(&ys, y)
	}

	lr: f32 = 0.0001
	epochs := 50
	print_every := 10

	A_node, next_idx := makeVarMat(DIM_OUT, DIM_IN, 0)
	b_node: NodeMatrix
	x_node: NodeMatrix
	y_node: NodeMatrix
	b_node, next_idx = makeVarMat(DIM_OUT, 1, next_idx)
	x_node, next_idx = makeVarMat(DIM_IN, 1, next_idx)
	y_node, next_idx = makeVarMat(DIM_OUT, 1, next_idx)

	num_trainable := DIM_OUT * DIM_IN + DIM_OUT
	x_start := num_trainable
	y_start := x_start + DIM_IN

	Ax := multNodeMat(&A_node, &x_node)
	y_pred := binaryOpNodeMat(.Add, &Ax, &b_node)

	diff := binaryOpNodeMat(.Sub, &y_pred, &y_node)
	sq := binaryOpNodeMat(.Mul, &diff, &diff)
	loss_node := reduceNodeMat(.Add, &sq)

	binding := make([]f32, next_idx)

	fmt.printf(
		"Training linear model: %dx%d matrix, %d bias, %d datapoints\n",
		DIM_OUT,
		DIM_IN,
		DIM_OUT,
		NUM_POINTS,
	)
	fmt.printf("Trainable parameters: %d\n", num_trainable)

	timing_runs := 10
	train_stats := time_train_epoch(
		loss_node,
		binding,
		xs[:],
		ys[:],
		x_start,
		y_start,
		num_trainable,
		lr,
		timing_runs,
	)

	fmt.printf(
		"Interpreted (%d epochs): mean=%.3fms, var=%.6f, cv=%.4f\n",
		timing_runs,
		train_stats.mean_ms,
		train_stats.var_ms,
		train_stats.cv,
	)

	compiled := compile_reverse(loss_node, next_idx)
	compiled_mem := make([]f32, compiled.total_mem_size)
	defer delete(compiled_mem)

	fmt.printf(
		"Compiled: %d instructions, %d mem slots\n",
		len(compiled.instrs),
		compiled.total_mem_size,
	)

	compiled_stats := time_train_epoch_compiled(
		compiled,
		compiled_mem,
		xs[:],
		ys[:],
		x_start,
		y_start,
		num_trainable,
		lr,
		timing_runs,
	)

	fmt.printf(
		"Compiled (%d epochs): mean=%.3fms, var=%.6f, cv=%.4f\n",
		timing_runs,
		compiled_stats.mean_ms,
		compiled_stats.var_ms,
		compiled_stats.cv,
	)

	speedup := train_stats.mean_ms / compiled_stats.mean_ms
	fmt.printf("Speedup: %.2fx\n", speedup)

	for j in 0 ..< next_idx {
		binding[j] = 0
	}

	grads := make([]f32, next_idx)
	defer delete(grads)

	activation_cache: map[^Node]f32
	grad_cache: map[^Node]f32
	defer delete(activation_cache)
	defer delete(grad_cache)

	for epoch in 0 ..< epochs {
		total_loss: f32 = 0

		for i in 0 ..< len(xs) {
			for j in 0 ..< DIM_IN {
				binding[x_start + j] = xs[i][j]
			}
			for j in 0 ..< DIM_OUT {
				binding[y_start + j] = ys[i][j]
			}

			loss_val := eval_grad_reverse(loss_node, binding, grads, &activation_cache, &grad_cache)
			total_loss += loss_val

			for j in 0 ..< num_trainable {
				binding[j] -= lr * grads[j]
			}
		}

		if epoch % print_every == 0 {
			fmt.printf("epoch %d: avg_loss = %f\n", epoch, total_loss / f32(len(xs)))
		}
	}

	fmt.println("\nLearned A:")
	for i in 0 ..< DIM_OUT {
		fmt.printf("  [")
		for j in 0 ..< DIM_IN {
			fmt.printf("%.3f", binding[i * DIM_IN + j])
			if j < DIM_IN - 1 {fmt.printf(", ")}
		}
		fmt.printf("]\n")
	}

	fmt.println("True A:")
	for i in 0 ..< DIM_OUT {
		fmt.printf("  [")
		for j in 0 ..< DIM_IN {
			fmt.printf("%.3f", true_A[i][j])
			if j < DIM_IN - 1 {fmt.printf(", ")}
		}
		fmt.printf("]\n")
	}

	fmt.println("\nLearned b:")
	fmt.printf("  [")
	for i in 0 ..< DIM_OUT {
		fmt.printf("%.3f", binding[DIM_OUT * DIM_IN + i])
		if i < DIM_OUT - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")

	fmt.println("True b:")
	fmt.printf("  [")
	for i in 0 ..< DIM_OUT {
		fmt.printf("%.3f", true_b[i])
		if i < DIM_OUT - 1 {fmt.printf(", ")}
	}
	fmt.printf("]\n")
}

time_train_epoch_diff_vm :: proc(
	instrs: []DiffInstr,
	mem_size: int,
	xs: [][]f32,
	ys: [][]f32,
	x_start: int,
	y_start: int,
	num_trainable: int,
	lr: f32,
	runs: int,
) -> TrainStats {
	sw: time.Stopwatch
	m, squares: f64
	n: int

	mem := make([]f32, mem_size)
	defer delete(mem)
	grads := make([]f32, mem_size)
	defer delete(grads)

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)

		for i in 0 ..< len(xs) {
			for j in 0 ..< len(xs[i]) {
				mem[x_start + j] = xs[i][j]
			}
			for j in 0 ..< len(ys[i]) {
				mem[y_start + j] = ys[i][j]
			}

			for j in 0 ..< mem_size {
				grads[j] = 0
			}

			_, tape := diff_sim_forward(instrs, mem)
			diff_sim_backward(tape, grads)
			delete(tape)

			for j in 0 ..< num_trainable {
				mem[j] -= lr * grads[j]
			}
		}

		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := m + ((x - m) / f64(n))
			squares = squares + ((x - m) * (x - m_new))
			m = m_new
		}
	}

	variance := squares / f64(n)
	return TrainStats{m, variance, math.sqrt(variance) / m}
}

time_train_epoch_mat :: proc(
	loss_node: ^MatNode,
	binding: []f32,
	var_shapes: []MatShape,
	xs: [][]f32,
	ys: [][]f32,
	x_idx: int,
	y_idx: int,
	num_trainable: int,
	lr: f32,
	runs: int,
) -> TrainStats {
	sw: time.Stopwatch
	m, squares: f64
	n: int

	grads := make([]f32, len(binding))
	defer delete(grads)

	activation_cache: map[^MatNode][]f32
	grad_cache: map[^MatNode][]f32
	defer delete(activation_cache)
	defer delete(grad_cache)

	x_offset := mat_var_offset(x_idx, var_shapes)
	y_offset := mat_var_offset(y_idx, var_shapes)
	x_size := mat_size(var_shapes[x_idx])
	y_size := mat_size(var_shapes[y_idx])

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)

		for i in 0 ..< len(xs) {
			for j in 0 ..< x_size {
				binding[x_offset + j] = xs[i][j]
			}
			for j in 0 ..< y_size {
				binding[y_offset + j] = ys[i][j]
			}

			val := eval_mat_grad_reverse(
				loss_node,
				binding,
				var_shapes,
				grads,
				&activation_cache,
				&grad_cache,
			)
			delete(val)

			for j in 0 ..< num_trainable {
				binding[j] -= lr * grads[j]
			}
		}

		time.stopwatch_stop(&sw)
		x := time.duration_milliseconds(time.stopwatch_duration(sw))

		if n == 1 {
			m = x
		} else {
			m_new := m + ((x - m) / f64(n))
			squares = squares + ((x - m) * (x - m_new))
			m = m_new
		}
	}

	delete_mat_cache_contents(&activation_cache)
	delete_mat_cache_contents(&grad_cache)

	variance := squares / f64(n)
	return TrainStats{m, variance, math.sqrt(variance) / m}
}

trainLinearMatrixGrad :: proc() {
	DIM_IN :: 50
	DIM_OUT :: 50
	NUM_POINTS :: 1000

	true_A: [DIM_OUT][DIM_IN]f32
	true_b: [DIM_OUT]f32

	for i in 0 ..< DIM_OUT {
		for j in 0 ..< DIM_IN {
			true_A[i][j] = rand.float32_range(-2, 2)
		}
		true_b[i] = rand.float32_range(-1, 1)
	}

	xs: [dynamic][]f32
	ys: [dynamic][]f32

	for _ in 0 ..< NUM_POINTS {
		x := make([]f32, DIM_IN)
		for j in 0 ..< DIM_IN {
			x[j] = rand.float32_range(-5, 5)
		}

		y := make([]f32, DIM_OUT)
		for i in 0 ..< DIM_OUT {
			sum: f32 = true_b[i]
			for j in 0 ..< DIM_IN {
				sum += true_A[i][j] * x[j]
			}
			y[i] = sum
		}

		append(&xs, x)
		append(&ys, y)
	}

	defer {
		for i in 0 ..< len(xs) {
			delete(xs[i])
			delete(ys[i])
		}
		delete(xs)
		delete(ys)
	}

	lr: f32 = 0.0001
	epochs := 50
	print_every := 10

	var_shapes := []MatShape{
		{DIM_OUT, DIM_IN},
		{DIM_OUT, 1},
		{DIM_IN, 1},
		{DIM_OUT, 1},
	}

	num_trainable := DIM_OUT * DIM_IN + DIM_OUT
	total_size := num_trainable + DIM_IN + DIM_OUT

	A_node := make_mat_var(0)
	b_node := make_mat_var(1)
	x_node := make_mat_var(2)
	y_node := make_mat_var(3)

	Ax := make_mat_op(.MatMul, A_node, x_node)
	pred := make_mat_op(.Add, Ax, b_node)
	diff := make_mat_op(.Sub, pred, y_node)
	sq := make_mat_op(.Mul, diff, diff)
	loss_node := make_mat_op(.ReduceSum, sq)

	binding := make([]f32, total_size)
	defer delete(binding)

	fmt.printf(
		"Training linear model (MatNode): %dx%d matrix, %d bias, %d datapoints\n",
		DIM_OUT,
		DIM_IN,
		DIM_OUT,
		NUM_POINTS,
	)
	fmt.printf("Trainable parameters: %d\n", num_trainable)

	timing_runs := 10
	train_stats := time_train_epoch_mat(
		loss_node,
		binding,
		var_shapes,
		xs[:],
		ys[:],
		2,
		3,
		num_trainable,
		lr,
		timing_runs,
	)

	fmt.printf(
		"Timing (%d epochs): mean=%.3fms, var=%.6f, cv=%.4f\n",
		timing_runs,
		train_stats.mean_ms,
		train_stats.var_ms,
		train_stats.cv,
	)

	for j in 0 ..< total_size {
		binding[j] = 0
	}

	grads := make([]f32, total_size)
	defer delete(grads)

	activation_cache: map[^MatNode][]f32
	grad_cache: map[^MatNode][]f32
	defer {
		delete_mat_cache_contents(&activation_cache)
		delete_mat_cache_contents(&grad_cache)
		delete(activation_cache)
		delete(grad_cache)
	}

	x_offset := mat_var_offset(2, var_shapes)
	y_offset := mat_var_offset(3, var_shapes)

	for epoch in 0 ..< epochs {
		total_loss: f32 = 0

		for i in 0 ..< len(xs) {
			for j in 0 ..< DIM_IN {
				binding[x_offset + j] = xs[i][j]
			}
			for j in 0 ..< DIM_OUT {
				binding[y_offset + j] = ys[i][j]
			}

			loss_val := eval_mat_grad_reverse(
				loss_node,
				binding,
				var_shapes,
				grads,
				&activation_cache,
				&grad_cache,
			)
			total_loss += loss_val[0]
			delete(loss_val)

			for j in 0 ..< num_trainable {
				binding[j] -= lr * grads[j]
			}
		}

		if epoch % print_every == 0 {
			fmt.printf("epoch %d: avg_loss = %f\n", epoch, total_loss / f32(len(xs)))
		}
	}

	fmt.println("\nLearned A (first 5x5):")
	for i in 0 ..< min(5, DIM_OUT) {
		fmt.printf("  [")
		for j in 0 ..< min(5, DIM_IN) {
			fmt.printf("%.3f", binding[i * DIM_IN + j])
			if j < min(5, DIM_IN) - 1 {fmt.printf(", ")}
		}
		fmt.printf("...]\n")
	}

	fmt.println("True A (first 5x5):")
	for i in 0 ..< min(5, DIM_OUT) {
		fmt.printf("  [")
		for j in 0 ..< min(5, DIM_IN) {
			fmt.printf("%.3f", true_A[i][j])
			if j < min(5, DIM_IN) - 1 {fmt.printf(", ")}
		}
		fmt.printf("...]\n")
	}

	b_offset := DIM_OUT * DIM_IN
	fmt.println("\nLearned b (first 5):")
	fmt.printf("  [")
	for i in 0 ..< min(5, DIM_OUT) {
		fmt.printf("%.3f", binding[b_offset + i])
		if i < min(5, DIM_OUT) - 1 {fmt.printf(", ")}
	}
	fmt.printf("...]\n")

	fmt.println("True b (first 5):")
	fmt.printf("  [")
	for i in 0 ..< min(5, DIM_OUT) {
		fmt.printf("%.3f", true_b[i])
		if i < min(5, DIM_OUT) - 1 {fmt.printf(", ")}
	}
	fmt.printf("...]\n")
}
