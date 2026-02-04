package main
import "core:fmt"

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

trainLinear :: proc() {
	// make fake linear data
	mat_A := matrix[2, 2]f32{
		2, 1,
		3, 4,
	}

	b := matrix[2, 1]f32{
		0,
		1,
	}

	xs: [dynamic]matrix[2, 1]f32
	for i in ([4]f32{2, 3, 4, 5}) {
		for j in ([4]f32{1, 5, 6, 1}) {
			append(&xs, matrix[2, 1]f32{
				i,
				j,
			})
		}
	}

	// ideally broadcasting would be a thing
	// then all of this is one line
	ys: [dynamic]matrix[2, 1]f32
	for x in xs {
		append(&ys, mat_A * x + b)
	}

	// using gradient descent, train a new mat_A and b based on these xs and ys
	lr: f32 = 0.01
	epochs := 100
	print_every := 10

	// build computation graph once
	// binding layout: A (0-3), b (4-5), x (6-7), y (8-9)
	A_node, next_idx := makeVarMat(2, 2, 0)
	b_node: NodeMatrix
	x_node: NodeMatrix
	y_node: NodeMatrix
	b_node, next_idx = makeVarMat(2, 1, next_idx)
	x_node, next_idx = makeVarMat(2, 1, next_idx)
	y_node, next_idx = makeVarMat(2, 1, next_idx)

	// y_pred = A @ x + b
	Ax := multNodeMat(&A_node, &x_node)
	y_pred := binaryOpNodeMat(.Add, &Ax, &b_node)

	// loss = sum((y_pred - y)^2)
	diff := binaryOpNodeMat(.Sub, &y_pred, &y_node)
	sq := binaryOpNodeMat(.Mul, &diff, &diff)
	loss_node := reduceNodeMat(.Add, &sq)

	// initialize binding (all zeros)
	binding := make([]f32, next_idx)

	// training loop
	for epoch in 0 ..< epochs {
		total_loss: f32 = 0

		for i in 0 ..< len(xs) {
			// load x into binding
			binding[6] = xs[i][0, 0]
			binding[7] = xs[i][1, 0]

			// load y into binding
			binding[8] = ys[i][0, 0]
			binding[9] = ys[i][1, 0]

			loss_val, grads := eval_grad_reverse(loss_node, binding)
			total_loss += loss_val

			// update only trainable params (A and b, indices 0-5)
			for j in 0 ..< 6 {
				binding[j] -= lr * grads[j]
			}
		}

		if epoch % print_every == 0 {
			fmt.printf("epoch %d: avg_loss = %f\n", epoch, total_loss / f32(len(xs)))
		}
	}

	fmt.println("Learned A:")
	fmt.printf("  [%f, %f]\n", binding[0], binding[1])
	fmt.printf("  [%f, %f]\n", binding[2], binding[3])
	fmt.println("True A:")
	fmt.println("  [2, 1]")
	fmt.println("  [3, 4]")

	fmt.println("Learned b:")
	fmt.printf("  [%f, %f]\n", binding[4], binding[5])
	fmt.println("True b:")
	fmt.println("  [0, 1]")
}
