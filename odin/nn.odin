package main
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:time"

generateNNData :: proc(
	input_dim: int,
	hidden_dim: int,
	output_dim: int,
	num_points: int,
) -> (
	xs: [dynamic][]f32,
	ys: [dynamic][]f32,
) {
	true_W1 := make([]f32, hidden_dim * input_dim)
	true_b1 := make([]f32, hidden_dim)
	true_W2 := make([]f32, output_dim * hidden_dim)
	true_b2 := make([]f32, output_dim)
	defer delete(true_W1)
	defer delete(true_b1)
	defer delete(true_W2)
	defer delete(true_b2)

	for i in 0 ..< len(true_W1) do true_W1[i] = rand.float32_range(-1, 1)
	for i in 0 ..< len(true_b1) do true_b1[i] = rand.float32_range(-0.5, 0.5)
	for i in 0 ..< len(true_W2) do true_W2[i] = rand.float32_range(-1, 1)
	for i in 0 ..< len(true_b2) do true_b2[i] = rand.float32_range(-0.5, 0.5)

	xs = make([dynamic][]f32)
	ys = make([dynamic][]f32)

	for _ in 0 ..< num_points {
		x := make([]f32, input_dim)
		for j in 0 ..< input_dim do x[j] = rand.float32_range(-2, 2)

		h := make([]f32, hidden_dim)
		defer delete(h)
		for i in 0 ..< hidden_dim {
			s: f32 = true_b1[i]
			for j in 0 ..< input_dim do s += true_W1[i * input_dim + j] * x[j]
			h[i] = max(0, s)
		}

		y := make([]f32, output_dim)
		for i in 0 ..< output_dim {
			s: f32 = true_b2[i]
			for j in 0 ..< hidden_dim do s += true_W2[i * hidden_dim + j] * h[j]
			y[i] = s
		}

		append(&xs, x)
		append(&ys, y)
	}
	return
}

// buildNNGraph constructs the computation graph for:
//   x (input_dim) -> W1*x+b1 -> ReLU -> W2*h1+b2 -> ReLU -> W3*h2+b3 -> MSE vs y
//
// var_shapes must be: [W1, b1, W2, b2, W3, b3, x, y]
// Returns the loss node, indices of x and y vars, and count of trainable parameters.
buildNNGraph :: proc(
	var_shapes: []MatShape,
) -> (
	loss_node: ^MatNode,
	x_idx: int,
	y_idx: int,
	num_trainable: int,
) {
	x_idx = len(var_shapes) - 2
	y_idx = len(var_shapes) - 1
	for i in 0 ..< x_idx do num_trainable += mat_size(var_shapes[i])

	W1 := make_mat_var(0)
	b1 := make_mat_var(1)
	W2 := make_mat_var(2)
	b2 := make_mat_var(3)
	W3 := make_mat_var(4)
	b3 := make_mat_var(5)
	x  := make_mat_var(6)
	y  := make_mat_var(7)

	h1 := make_mat_op(.ReLU, make_mat_op(.Add, make_mat_op(.MatMul, W1, x), b1))
	h2 := make_mat_op(.ReLU, make_mat_op(.Add, make_mat_op(.MatMul, W2, h1), b2))
	pred := make_mat_op(.Add, make_mat_op(.MatMul, W3, h2), b3)
	diff := make_mat_op(.Sub, pred, y)
	loss_node = make_mat_op(.ReduceSum, make_mat_op(.Mul, diff, diff))
	return
}

nn_var_shapes :: proc(input_dim, hidden_dim, output_dim: int) -> []MatShape {
	s := make([]MatShape, 8)
	s[0] = {hidden_dim, input_dim}  // W1
	s[1] = {hidden_dim, 1}          // b1
	s[2] = {hidden_dim, hidden_dim} // W2
	s[3] = {hidden_dim, 1}          // b2
	s[4] = {output_dim, hidden_dim} // W3
	s[5] = {output_dim, 1}          // b3
	s[6] = {input_dim, 1}           // x
	s[7] = {output_dim, 1}          // y
	return s
}

time_train_epoch_nn :: proc(
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
	defer {
		delete_mat_cache_contents(&activation_cache)
		delete_mat_cache_contents(&grad_cache)
		delete(activation_cache)
		delete(grad_cache)
	}

	x_offset := mat_var_offset(x_idx, var_shapes)
	y_offset := mat_var_offset(y_idx, var_shapes)
	x_size := mat_size(var_shapes[x_idx])
	y_size := mat_size(var_shapes[y_idx])

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)

		for i in 0 ..< len(xs) {
			for j in 0 ..< x_size do binding[x_offset + j] = xs[i][j]
			for j in 0 ..< y_size do binding[y_offset + j] = ys[i][j]

			val := eval_mat_grad_reverse(loss_node, binding, var_shapes, grads, &activation_cache, &grad_cache)
			delete(val)

			for j in 0 ..< num_trainable do binding[j] -= lr * grads[j]
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

time_train_epoch_nn_compiled :: proc(
	compiled: CompiledReverse,
	mem: []f32,
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

	x_offset := mat_var_offset(x_idx, var_shapes)
	y_offset := mat_var_offset(y_idx, var_shapes)
	x_size := mat_size(var_shapes[x_idx])
	y_size := mat_size(var_shapes[y_idx])

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)

		for i in 0 ..< len(xs) {
			for j in 0 ..< x_size do mem[x_offset + j] = xs[i][j]
			for j in 0 ..< y_size do mem[y_offset + j] = ys[i][j]

			for j in 0 ..< compiled.total_mem_size - compiled.grad_offset {
				mem[compiled.grad_offset + j] = 0
			}

			simulate(compiled.instrs[:], mem)

			for j in 0 ..< num_trainable do mem[j] -= lr * mem[compiled.grad_offset + j]
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

// time_train_epoch_nn_diff_vm benchmarks the DiffInstr tape-based compiled path.
// grad_size must be >= total_mem_size so diff_sim_backward can write to all addresses.
// Param gradients are read from [0..num_trainable) of the grads array.
time_train_epoch_nn_diff_vm :: proc(
	instrs: []DiffInstr,
	mem: []f32,
	var_shapes: []MatShape,
	xs: [][]f32,
	ys: [][]f32,
	x_idx: int,
	y_idx: int,
	num_trainable: int,
	grad_size: int,
	lr: f32,
	runs: int,
) -> TrainStats {
	sw: time.Stopwatch
	m, squares: f64
	n: int

	x_offset := mat_var_offset(x_idx, var_shapes)
	y_offset := mat_var_offset(y_idx, var_shapes)
	x_size := mat_size(var_shapes[x_idx])
	y_size := mat_size(var_shapes[y_idx])

	grads := make([]f32, grad_size)
	defer delete(grads)

	for _ in 0 ..< runs {
		n += 1
		time.stopwatch_reset(&sw)
		time.stopwatch_start(&sw)

		for i in 0 ..< len(xs) {
			for j in 0 ..< x_size do mem[x_offset + j] = xs[i][j]
			for j in 0 ..< y_size do mem[y_offset + j] = ys[i][j]

			for j in 0 ..< grad_size do grads[j] = 0

			_, tape := diff_sim_forward(instrs, mem)
			diff_sim_backward(tape, grads)
			delete(tape)

			for j in 0 ..< num_trainable do mem[j] -= lr * grads[j]
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

trainNNMatrixGrad :: proc() {
	INPUT_DIM  :: 50
	HIDDEN_DIM :: 64
	OUTPUT_DIM :: 50
	NUM_POINTS :: 1000

	xs, ys := generateNNData(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_POINTS)
	defer {
		for i in 0 ..< len(xs) {delete(xs[i]); delete(ys[i])}
		delete(xs); delete(ys)
	}

	var_shapes := nn_var_shapes(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
	defer delete(var_shapes)
	loss_node, x_idx, y_idx, num_trainable := buildNNGraph(var_shapes)

	total_size := 0
	for shape in var_shapes do total_size += mat_size(shape)

	binding := make([]f32, total_size)
	defer delete(binding)
	for j in 0 ..< num_trainable do binding[j] = rand.float32_range(-0.1, 0.1)

	fmt.printf(
		"Training NN: %d→%d(ReLU)→%d(ReLU)→%d, %d datapoints\n",
		INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_POINTS,
	)
	fmt.printf("Trainable parameters: %d\n", num_trainable)

	lr: f32 = 0.0001
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

	x_offset := mat_var_offset(x_idx, var_shapes)
	y_offset := mat_var_offset(y_idx, var_shapes)

	for epoch in 0 ..< 50 {
		total_loss: f32 = 0
		for i in 0 ..< len(xs) {
			for j in 0 ..< len(xs[i]) do binding[x_offset + j] = xs[i][j]
			for j in 0 ..< len(ys[i]) do binding[y_offset + j] = ys[i][j]

			loss_val := eval_mat_grad_reverse(loss_node, binding, var_shapes, grads, &activation_cache, &grad_cache)
			total_loss += loss_val[0]
			delete(loss_val)

			for j in 0 ..< num_trainable do binding[j] -= lr * grads[j]
		}
		if epoch % 10 == 0 {
			fmt.printf("epoch %d: avg_loss = %f\n", epoch, total_loss / f32(len(xs)))
		}
	}
}

timeNNMatrixGrad :: proc() {
	INPUT_DIM  :: 50
	HIDDEN_DIM :: 64
	OUTPUT_DIM :: 50
	NUM_POINTS :: 1000

	xs, ys := generateNNData(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_POINTS)
	defer {
		for i in 0 ..< len(xs) {delete(xs[i]); delete(ys[i])}
		delete(xs); delete(ys)
	}

	var_shapes := nn_var_shapes(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
	defer delete(var_shapes)
	loss_node, x_idx, y_idx, num_trainable := buildNNGraph(var_shapes)

	total_size := 0
	for shape in var_shapes do total_size += mat_size(shape)

	binding := make([]f32, total_size)
	defer delete(binding)
	for j in 0 ..< num_trainable do binding[j] = rand.float32_range(-0.1, 0.1)

	fmt.printf(
		"\nBenchmarking NN (interpreted): %d→%d(ReLU)→%d(ReLU)→%d, %d datapoints\n",
		INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_POINTS,
	)
	fmt.printf("Trainable parameters: %d\n", num_trainable)

	stats := time_train_epoch_nn(loss_node, binding, var_shapes, xs[:], ys[:], x_idx, y_idx, num_trainable, 0.0001, 10)
	fmt.printf("Interpreted (10 epochs): mean=%.3fms, var=%.6f, cv=%.4f\n", stats.mean_ms, stats.var_ms, stats.cv)
}

timeNNCompilationMethods :: proc() {
	INPUT_DIM  :: 50
	HIDDEN_DIM :: 64
	OUTPUT_DIM :: 50
	NUM_POINTS :: 1000

	xs, ys := generateNNData(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_POINTS)
	defer {
		for i in 0 ..< len(xs) {delete(xs[i]); delete(ys[i])}
		delete(xs); delete(ys)
	}

	var_shapes := nn_var_shapes(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
	defer delete(var_shapes)
	loss_node, x_idx, y_idx, num_trainable := buildNNGraph(var_shapes)

	total_size := 0
	for shape in var_shapes do total_size += mat_size(shape)

	base_binding := make([]f32, total_size)
	defer delete(base_binding)
	for j in 0 ..< num_trainable do base_binding[j] = rand.float32_range(-0.1, 0.1)

	fmt.printf(
		"\nBenchmarking NN compilation methods: %d→%d(ReLU)→%d(ReLU)→%d, %d datapoints\n",
		INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_POINTS,
	)
	fmt.printf("Trainable parameters: %d\n", num_trainable)

	timing_runs := 10

	// interpreted baseline
	interp_binding := make([]f32, total_size)
	defer delete(interp_binding)
	copy(interp_binding, base_binding)
	interp_stats := time_train_epoch_nn(loss_node, interp_binding, var_shapes, xs[:], ys[:], x_idx, y_idx, num_trainable, 0.0001, timing_runs)

	// compile_reverse_mat → simulate (VInstr path)
	compiled_v := compile_reverse_mat(loss_node, var_shapes, num_trainable)
	defer delete(compiled_v.instrs)
	compiled_v_mem := make([]f32, compiled_v.total_mem_size)
	defer delete(compiled_v_mem)
	copy(compiled_v_mem, base_binding)

	compiled_v_stats := time_train_epoch_nn_compiled(compiled_v, compiled_v_mem, var_shapes, xs[:], ys[:], x_idx, y_idx, num_trainable, 0.0001, timing_runs)

	// compile_diff_vm_mat → diff_sim_forward + diff_sim_backward (DiffInstr tape path)
	diff_vm_instrs, diff_vm_total := compile_diff_vm_mat(loss_node, var_shapes)
	defer delete(diff_vm_instrs)
	diff_vm_mem := make([]f32, diff_vm_total)
	defer delete(diff_vm_mem)
	copy(diff_vm_mem, base_binding)

	diff_vm_stats := time_train_epoch_nn_diff_vm(diff_vm_instrs[:], diff_vm_mem, var_shapes, xs[:], ys[:], x_idx, y_idx, num_trainable, diff_vm_total, 0.0001, timing_runs)

	fmt.printf("Compiled: %d VInstrs, %d mem slots\n", len(compiled_v.instrs), compiled_v.total_mem_size)
	fmt.printf("Diff-VM:  %d DInstrs, %d mem slots\n", len(diff_vm_instrs), diff_vm_total)
	fmt.printf("\n")
	fmt.printf("Interpreted  (%d epochs): mean=%.3fms, var=%.6f, cv=%.4f\n", timing_runs, interp_stats.mean_ms, interp_stats.var_ms, interp_stats.cv)
	fmt.printf("Compiled     (%d epochs): mean=%.3fms, var=%.6f, cv=%.4f\n", timing_runs, compiled_v_stats.mean_ms, compiled_v_stats.var_ms, compiled_v_stats.cv)
	fmt.printf("Diff-VM      (%d epochs): mean=%.3fms, var=%.6f, cv=%.4f\n", timing_runs, diff_vm_stats.mean_ms, diff_vm_stats.var_ms, diff_vm_stats.cv)
	fmt.printf("Speedup (interp/compiled): %.2fx\n", interp_stats.mean_ms / compiled_v_stats.mean_ms)
	fmt.printf("Speedup (interp/diff-vm):  %.2fx\n", interp_stats.mean_ms / diff_vm_stats.mean_ms)
}
