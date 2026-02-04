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

			_, grads := eval_grad_reverse(loss_node, binding)

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

trainLinear :: proc() {
	DIM_IN :: 4
	DIM_OUT :: 3
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

	lr: f32 = 0.00001
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
		"Timing (%d epochs): mean=%.3fms, var=%.6f, cv=%.4f\n",
		timing_runs,
		train_stats.mean_ms,
		train_stats.var_ms,
		train_stats.cv,
	)

	for j in 0 ..< next_idx {
		binding[j] = 0
	}

	for epoch in 0 ..< epochs {
		total_loss: f32 = 0

		for i in 0 ..< len(xs) {
			for j in 0 ..< DIM_IN {
				binding[x_start + j] = xs[i][j]
			}
			for j in 0 ..< DIM_OUT {
				binding[y_start + j] = ys[i][j]
			}

			loss_val, grads := eval_grad_reverse(loss_node, binding)
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
