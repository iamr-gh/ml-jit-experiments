package main

import "core:fmt"
import "core:math"
import "core:math/rand"

NNActivation :: enum {
	Sigmoid,
	ReLU,
}

nn_activation_name :: proc(a: NNActivation) -> string {
	switch a {
	case .Sigmoid:
		return "sigmoid"
	case .ReLU:
		return "relu"
	}
	return "unknown"
}

make_mat_const_fill :: proc(shape: MatShape, val: f32) -> ^MatNode {
	data := make([]f32, mat_size(shape))
	for i in 0 ..< len(data) {
		data[i] = val
	}
	return make_mat_const(shape.r, shape.c, data)
}

dense_mat :: proc(input, w, b: ^MatNode) -> ^MatNode {
	return make_mat_op(.Add, make_mat_op(.MatMul, w, input), b)
}

relu_mat :: proc(input: ^MatNode) -> ^MatNode {
	return make_mat_op(.ReLU, input)
}

sigmoid_mat :: proc(input: ^MatNode, shape: MatShape) -> ^MatNode {
	neg_one := make_mat_const_fill(shape, -1)
	one := make_mat_const_fill(shape, 1)
	neg := make_mat_op(.Mul, input, neg_one)
	denom := make_mat_op(.Add, one, make_mat_op(.Exp, neg))
	return make_mat_op(.Div, one, denom)
}

sigmoid_scalar :: proc(x: f32) -> f32 {
	return 1 / (1 + f32(math.exp(-f64(x))))
}

relu_scalar :: proc(x: f32) -> f32 {
	if x > 0 {
		return x
	}
	return 0
}

activate_mat :: proc(input: ^MatNode, shape: MatShape, activation: NNActivation) -> ^MatNode {
	switch activation {
	case .Sigmoid:
		return sigmoid_mat(input, shape)
	case .ReLU:
		return relu_mat(input)
	}
	return input
}

activate_scalar :: proc(x: f32, activation: NNActivation) -> f32 {
	switch activation {
	case .Sigmoid:
		return sigmoid_scalar(x)
	case .ReLU:
		return relu_scalar(x)
	}
	return x
}

init_dense_params :: proc(binding: []f32, offset, out_dim, in_dim: int, scale: f32) -> int {
	idx := offset
	for i in 0 ..< out_dim * in_dim {
		binding[idx + i] = rand.float32_range(-scale, scale)
	}
	idx += out_dim * in_dim
	for i in 0 ..< out_dim {
		binding[idx + i] = 0
	}
	return idx + out_dim
}

fill_synthetic_nn_dataset :: proc(
	xs: ^[dynamic][]f32,
	ys: ^[dynamic][]f32,
	num_points, dim_in, dim_hidden, dim_out: int,
	hidden_activation: NNActivation = .Sigmoid,
	output_activation: NNActivation = .Sigmoid,
) {
	teacher_w1 := make([]f32, dim_hidden * dim_in)
	teacher_b1 := make([]f32, dim_hidden)
	teacher_w2 := make([]f32, dim_out * dim_hidden)
	teacher_b2 := make([]f32, dim_out)
	defer {
		delete(teacher_w1)
		delete(teacher_b1)
		delete(teacher_w2)
		delete(teacher_b2)
	}

	for i in 0 ..< len(teacher_w1) {
		teacher_w1[i] = rand.float32_range(-1.5, 1.5)
	}
	for i in 0 ..< len(teacher_b1) {
		teacher_b1[i] = rand.float32_range(-0.5, 0.5)
	}
	for i in 0 ..< len(teacher_w2) {
		teacher_w2[i] = rand.float32_range(-1.5, 1.5)
	}
	for i in 0 ..< len(teacher_b2) {
		teacher_b2[i] = rand.float32_range(-0.5, 0.5)
	}

	for _ in 0 ..< num_points {
		x := make([]f32, dim_in)
		h := make([]f32, dim_hidden)
		y := make([]f32, dim_out)

		for j in 0 ..< dim_in {
			x[j] = rand.float32_range(-1, 1)
		}

		for i in 0 ..< dim_hidden {
			sum := teacher_b1[i]
			for j in 0 ..< dim_in {
				sum += teacher_w1[i * dim_in + j] * x[j]
			}
			h[i] = activate_scalar(sum, hidden_activation)
		}

		for i in 0 ..< dim_out {
			sum := teacher_b2[i]
			for j in 0 ..< dim_hidden {
				sum += teacher_w2[i * dim_hidden + j] * h[j]
			}
			y[i] = activate_scalar(sum, output_activation)
		}

		append(xs, x)
		append(ys, y)
		delete(h)
	}
}

build_two_layer_nn_loss :: proc(
	dim_in, dim_hidden, dim_out: int,
	hidden_activation: NNActivation,
	output_activation: NNActivation = .Sigmoid,
) -> (
	^MatNode,
	^MatNode,
	[6]MatShape,
	int,
	int,
	int,
) {
	var_shapes := [6]MatShape {
		{dim_hidden, dim_in},
		{dim_hidden, 1},
		{dim_out, dim_hidden},
		{dim_out, 1},
		{dim_in, 1},
		{dim_out, 1},
	}

	w1 := make_mat_var(0)
	b1 := make_mat_var(1)
	w2 := make_mat_var(2)
	b2 := make_mat_var(3)
	x := make_mat_var(4)
	y := make_mat_var(5)

	h_pre := dense_mat(x, w1, b1)
	h := activate_mat(h_pre, MatShape{dim_hidden, 1}, hidden_activation)
	pred_pre := dense_mat(h, w2, b2)
	pred := activate_mat(pred_pre, MatShape{dim_out, 1}, output_activation)
	diff := make_mat_op(.Sub, pred, y)
	loss_node := make_mat_op(.ReduceSum, make_mat_op(.Mul, diff, diff))

	num_trainable := dim_hidden * dim_in + dim_hidden + dim_out * dim_hidden + dim_out
	return loss_node, pred, var_shapes, num_trainable, 4, 5
}

init_two_layer_params :: proc(binding: []f32, dim_in, dim_hidden, dim_out: int) {
	offset := 0
	offset = init_dense_params(binding, offset, dim_hidden, dim_in, 0.4)
	offset = init_dense_params(binding, offset, dim_out, dim_hidden, 0.4)
}

trainNNMatrixGrad :: proc() {
	DIM_IN :: 4
	DIM_HIDDEN :: 8
	DIM_OUT :: 2
	NUM_POINTS :: 600
	loss_node, pred, var_shapes_arr, num_trainable, x_idx, y_idx := build_two_layer_nn_loss(
		DIM_IN,
		DIM_HIDDEN,
		DIM_OUT,
		.Sigmoid,
		.Sigmoid,
	)
	var_shapes := var_shapes_arr[:]
	total_size := num_trainable + DIM_IN + DIM_OUT

	xs: [dynamic][]f32
	ys: [dynamic][]f32
	fill_synthetic_nn_dataset(
		&xs,
		&ys,
		NUM_POINTS,
		DIM_IN,
		DIM_HIDDEN,
		DIM_OUT,
		.Sigmoid,
		.Sigmoid,
	)
	defer {
		for i in 0 ..< len(xs) {
			delete(xs[i])
			delete(ys[i])
		}
		delete(xs)
		delete(ys)
	}

	binding := make([]f32, total_size)
	defer delete(binding)

	init_two_layer_params(binding, DIM_IN, DIM_HIDDEN, DIM_OUT)

	x_offset := mat_var_offset(x_idx, var_shapes)
	y_offset := mat_var_offset(y_idx, var_shapes)

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

	fmt.printf(
		"Training NN (MatNode): %d -> %d -> %d, datapoints=%d, trainable=%d\n",
		DIM_IN,
		DIM_HIDDEN,
		DIM_OUT,
		NUM_POINTS,
		num_trainable,
	)

	for j in 0 ..< DIM_IN {
		binding[x_offset + j] = xs[0][j]
	}
	delete_mat_cache_contents(&activation_cache)
	before := eval_mat_grad_reverse_forward(pred, binding, var_shapes, &activation_cache)
	fmt.printf(
		"sample pred before: [%.4f, %.4f], target: [%.4f, %.4f]\n",
		before[0],
		before[1],
		ys[0][0],
		ys[0][1],
	)
	delete(before)

	lr: f32 = 0.08
	epochs := 300
	print_every := 30

	for epoch in 0 ..< epochs {
		total_loss: f32 = 0

		for i in 0 ..< len(xs) {
			for j in 0 ..< DIM_IN {
				binding[x_offset + j] = xs[i][j]
			}
			for j in 0 ..< DIM_OUT {
				binding[y_offset + j] = ys[i][j]
			}

			loss := eval_mat_grad_reverse(
				loss_node,
				binding,
				var_shapes,
				grads,
				&activation_cache,
				&grad_cache,
			)
			total_loss += loss[0]
			delete(loss)

			for j in 0 ..< num_trainable {
				binding[j] -= lr * grads[j]
			}
		}

		if epoch % print_every == 0 {
			fmt.printf("epoch %d: avg_loss=%f\n", epoch, total_loss / f32(len(xs)))
		}
	}

	for j in 0 ..< DIM_IN {
		binding[x_offset + j] = xs[0][j]
	}
	delete_mat_cache_contents(&activation_cache)
	after := eval_mat_grad_reverse_forward(pred, binding, var_shapes, &activation_cache)
	fmt.printf(
		"sample pred after:  [%.4f, %.4f], target: [%.4f, %.4f]\n",
		after[0],
		after[1],
		ys[0][0],
		ys[0][1],
	)
	delete(after)
}

timeNNMatrixGrad :: proc() {
	DIM_IN :: 32
	DIM_HIDDEN :: 64
	DIM_OUT :: 8
	NUM_POINTS :: 1000
	TIMING_RUNS :: 8
	LR: f32 = 0.001

	xs: [dynamic][]f32
	ys: [dynamic][]f32
	fill_synthetic_nn_dataset(
		&xs,
		&ys,
		NUM_POINTS,
		DIM_IN,
		DIM_HIDDEN,
		DIM_OUT,
		.Sigmoid,
		.Sigmoid,
	)
	defer {
		for i in 0 ..< len(xs) {
			delete(xs[i])
			delete(ys[i])
		}
		delete(xs)
		delete(ys)
	}

	fmt.printf(
		"\nNN timing benchmark (MatNode): %d -> %d -> %d, datapoints=%d\n",
		DIM_IN,
		DIM_HIDDEN,
		DIM_OUT,
		NUM_POINTS,
	)

	activations: [2]NNActivation = {.Sigmoid, .ReLU}
	for activation in activations {
		loss_node, _, var_shapes_arr, num_trainable, x_idx, y_idx := build_two_layer_nn_loss(
			DIM_IN,
			DIM_HIDDEN,
			DIM_OUT,
			activation,
			.Sigmoid,
		)
		var_shapes := var_shapes_arr[:]
		total_size := num_trainable + DIM_IN + DIM_OUT

		binding_eval := make([]f32, total_size)
		defer delete(binding_eval)
		init_two_layer_params(binding_eval, DIM_IN, DIM_HIDDEN, DIM_OUT)

		eval_stats := time_train_epoch_mat(
			loss_node,
			binding_eval,
			var_shapes,
			xs[:],
			ys[:],
			x_idx,
			y_idx,
			num_trainable,
			0,
			TIMING_RUNS,
		)

		binding_sgd := make([]f32, total_size)
		defer delete(binding_sgd)
		init_two_layer_params(binding_sgd, DIM_IN, DIM_HIDDEN, DIM_OUT)

		sgd_stats := time_train_epoch_mat(
			loss_node,
			binding_sgd,
			var_shapes,
			xs[:],
			ys[:],
			x_idx,
			y_idx,
			num_trainable,
			LR,
			TIMING_RUNS,
		)

		fmt.printf(
			"  hidden=%-7s eval-only mean=%.3fms cv=%.4f | eval+sgd mean=%.3fms cv=%.4f\n",
			nn_activation_name(activation),
			eval_stats.mean_ms,
			eval_stats.cv,
			sgd_stats.mean_ms,
			sgd_stats.cv,
		)
	}

	fmt.println("")
}
