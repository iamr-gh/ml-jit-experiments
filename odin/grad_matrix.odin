package main

MatShape :: struct {
	r, c: int,
}

MatOpType :: enum {
	Add,
	Sub,
	Mul,
	Div,
	MatMul,
	ReduceSum,
}

MatOp :: struct {
	type: MatOpType,
	l:    ^MatNode,
	r:    ^MatNode,
}

MatConst :: struct {
	shape: MatShape,
	data:  []f32,
}

MatNode :: union {
	MatOp,
	^MatConst,
	int,
}

MatValGrad :: struct {
	val:   []f32,
	grad:  []f32,
	shape: MatShape,
}

mat_size :: proc(shape: MatShape) -> int {
	return shape.r * shape.c
}

mat_var_offset :: proc(idx: int, var_shapes: []MatShape) -> int {
	offset := 0
	for i in 0 ..< idx {
		offset += mat_size(var_shapes[i])
	}
	return offset
}

mat_var_data :: proc(idx: int, binding: []f32, var_shapes: []MatShape) -> []f32 {
	offset := mat_var_offset(idx, var_shapes)
	size := mat_size(var_shapes[idx])
	return binding[offset:offset + size]
}

mat_node_shape :: proc(node: ^MatNode, var_shapes: []MatShape) -> MatShape {
	switch n in node {
	case int:
		return var_shapes[n]
	case ^MatConst:
		return n.shape
	case MatOp:
		l_shape := mat_node_shape(n.l, var_shapes)
		switch n.type {
		case .Add, .Sub, .Mul, .Div:
			return l_shape
		case .MatMul:
			r_shape := mat_node_shape(n.r, var_shapes)
			return MatShape{l_shape.r, r_shape.c}
		case .ReduceSum:
			return MatShape{1, 1}
		}
	}
	return MatShape{0, 0}
}

mat_transpose :: proc(data: []f32, shape: MatShape) -> []f32 {
	result := make([]f32, len(data))
	for r in 0 ..< shape.r {
		for c in 0 ..< shape.c {
			result[c * shape.r + r] = data[r * shape.c + c]
		}
	}
	return result
}

mat_clone :: proc(data: []f32) -> []f32 {
	result := make([]f32, len(data))
	for i in 0 ..< len(data) {
		result[i] = data[i]
	}
	return result
}

mat_zeros :: proc(size: int) -> []f32 {
	result := make([]f32, size)
	for i in 0 ..< size {
		result[i] = 0
	}
	return result
}

mat_negate :: proc(data: []f32) -> []f32 {
	result := make([]f32, len(data))
	for i in 0 ..< len(data) {
		result[i] = -data[i]
	}
	return result
}

mat_broadcast :: proc(scalar: f32, size: int) -> []f32 {
	result := make([]f32, size)
	for i in 0 ..< size {
		result[i] = scalar
	}
	return result
}

make_mat_var :: proc(idx: int) -> ^MatNode {
	node := new(MatNode, context.temp_allocator)
	node^ = idx
	return node
}

make_mat_const :: proc(r, c: int, data: []f32) -> ^MatNode {
	mc := new(MatConst, context.temp_allocator)
	mc.shape = MatShape{r, c}
	mc.data = data
	node := new(MatNode, context.temp_allocator)
	node^ = mc
	return node
}

make_mat_op :: proc(type: MatOpType, l: ^MatNode, r: ^MatNode = nil) -> ^MatNode {
	node := new(MatNode, context.temp_allocator)
	node^ = MatOp{type, l, r}
	return node
}

mat_elementwise_op :: proc(type: MatOpType, l, r: []f32) -> []f32 {
	assert(len(l) == len(r))
	result := make([]f32, len(l))
	for i in 0 ..< len(l) {
		switch type {
		case .Add:
			result[i] = l[i] + r[i]
		case .Sub:
			result[i] = l[i] - r[i]
		case .Mul:
			result[i] = l[i] * r[i]
		case .Div:
			result[i] = l[i] / r[i]
		case .MatMul, .ReduceSum:
			panic("not an elementwise op")
		}
	}
	return result
}

mat_multiply :: proc(l: []f32, l_shape: MatShape, r: []f32, r_shape: MatShape) -> []f32 {
	assert(l_shape.c == r_shape.r)
	out_shape := MatShape{l_shape.r, r_shape.c}
	result := make([]f32, mat_size(out_shape))

	for i in 0 ..< l_shape.r {
		for j in 0 ..< r_shape.c {
			sum: f32 = 0
			for k in 0 ..< l_shape.c {
				sum += l[i * l_shape.c + k] * r[k * r_shape.c + j]
			}
			result[i * r_shape.c + j] = sum
		}
	}
	return result
}

mat_reduce_sum :: proc(data: []f32) -> []f32 {
	result := make([]f32, 1)
	sum: f32 = 0
	for i in 0 ..< len(data) {
		sum += data[i]
	}
	result[0] = sum
	return result
}

eval_mat :: proc(node: ^MatNode, binding: []f32, var_shapes: []MatShape) -> ([]f32, MatShape) {
	switch n in node {
	case int:
		data := mat_var_data(n, binding, var_shapes)
		return mat_clone(data), var_shapes[n]
	case ^MatConst:
		return mat_clone(n.data), n.shape
	case MatOp:
		l_val, l_shape := eval_mat(n.l, binding, var_shapes)
		defer delete(l_val)

		switch n.type {
		case .Add, .Sub, .Mul, .Div:
			r_val, _ := eval_mat(n.r, binding, var_shapes)
			defer delete(r_val)
			return mat_elementwise_op(n.type, l_val, r_val), l_shape
		case .MatMul:
			r_val, r_shape := eval_mat(n.r, binding, var_shapes)
			defer delete(r_val)
			out_shape := MatShape{l_shape.r, r_shape.c}
			return mat_multiply(l_val, l_shape, r_val, r_shape), out_shape
		case .ReduceSum:
			return mat_reduce_sum(l_val), MatShape{1, 1}
		}
	}
	return nil, MatShape{0, 0}
}

eval_mat_grad_forward :: proc(
	node: ^MatNode,
	binding: []f32,
	var_shapes: []MatShape,
	respect: int,
) -> MatValGrad {
	switch n in node {
	case int:
		data := mat_var_data(n, binding, var_shapes)
		shape := var_shapes[n]
		size := mat_size(shape)
		val := mat_clone(data)
		grad: []f32
		if n == respect {
			grad = mat_broadcast(1, size)
		} else {
			grad = mat_zeros(size)
		}
		return MatValGrad{val, grad, shape}

	case ^MatConst:
		size := mat_size(n.shape)
		return MatValGrad{mat_clone(n.data), mat_zeros(size), n.shape}

	case MatOp:
		lvg := eval_mat_grad_forward(n.l, binding, var_shapes, respect)
		defer delete(lvg.val)
		defer delete(lvg.grad)

		switch n.type {
		case .Add:
			rvg := eval_mat_grad_forward(n.r, binding, var_shapes, respect)
			defer delete(rvg.val)
			defer delete(rvg.grad)
			val := mat_elementwise_op(.Add, lvg.val, rvg.val)
			grad := mat_elementwise_op(.Add, lvg.grad, rvg.grad)
			return MatValGrad{val, grad, lvg.shape}

		case .Sub:
			rvg := eval_mat_grad_forward(n.r, binding, var_shapes, respect)
			defer delete(rvg.val)
			defer delete(rvg.grad)
			val := mat_elementwise_op(.Sub, lvg.val, rvg.val)
			grad := mat_elementwise_op(.Sub, lvg.grad, rvg.grad)
			return MatValGrad{val, grad, lvg.shape}

		case .Mul:
			rvg := eval_mat_grad_forward(n.r, binding, var_shapes, respect)
			defer delete(rvg.val)
			defer delete(rvg.grad)
			val := mat_elementwise_op(.Mul, lvg.val, rvg.val)
			// grad = dL * R + L * dR
			term1 := mat_elementwise_op(.Mul, lvg.grad, rvg.val)
			defer delete(term1)
			term2 := mat_elementwise_op(.Mul, lvg.val, rvg.grad)
			defer delete(term2)
			grad := mat_elementwise_op(.Add, term1, term2)
			return MatValGrad{val, grad, lvg.shape}

		case .Div:
			rvg := eval_mat_grad_forward(n.r, binding, var_shapes, respect)
			defer delete(rvg.val)
			defer delete(rvg.grad)
			val := mat_elementwise_op(.Div, lvg.val, rvg.val)
			// grad = (dL * R - L * dR) / R^2
			term1 := mat_elementwise_op(.Mul, lvg.grad, rvg.val)
			defer delete(term1)
			term2 := mat_elementwise_op(.Mul, lvg.val, rvg.grad)
			defer delete(term2)
			numer := mat_elementwise_op(.Sub, term1, term2)
			defer delete(numer)
			denom := mat_elementwise_op(.Mul, rvg.val, rvg.val)
			defer delete(denom)
			grad := mat_elementwise_op(.Div, numer, denom)
			return MatValGrad{val, grad, lvg.shape}

		case .MatMul:
			rvg := eval_mat_grad_forward(n.r, binding, var_shapes, respect)
			defer delete(rvg.val)
			defer delete(rvg.grad)
			val := mat_multiply(lvg.val, lvg.shape, rvg.val, rvg.shape)
			// grad = dL @ R + L @ dR
			term1 := mat_multiply(lvg.grad, lvg.shape, rvg.val, rvg.shape)
			defer delete(term1)
			term2 := mat_multiply(lvg.val, lvg.shape, rvg.grad, rvg.shape)
			defer delete(term2)
			out_shape := MatShape{lvg.shape.r, rvg.shape.c}
			grad := mat_elementwise_op(.Add, term1, term2)
			return MatValGrad{val, grad, out_shape}

		case .ReduceSum:
			val := mat_reduce_sum(lvg.val)
			grad := mat_reduce_sum(lvg.grad)
			return MatValGrad{val, grad, MatShape{1, 1}}
		}
	}
	return MatValGrad{nil, nil, MatShape{0, 0}}
}

eval_mat_grad_forward_all :: proc(
	node: ^MatNode,
	binding: []f32,
	var_shapes: []MatShape,
) -> ([]f32, MatShape, [][]f32) {
	num_vars := len(var_shapes)
	grads := make([][]f32, num_vars)

	val: []f32
	shape: MatShape

	for i in 0 ..< num_vars {
		vg := eval_mat_grad_forward(node, binding, var_shapes, i)
		grads[i] = vg.grad
		if i == 0 {
			val = vg.val
			shape = vg.shape
		} else {
			delete(vg.val)
		}
	}

	return val, shape, grads
}

eval_mat_grad_reverse :: proc(
	node: ^MatNode,
	binding: []f32,
	var_shapes: []MatShape,
	out_grads: []f32,
	activation_cache: ^map[^MatNode][]f32 = nil,
	grad_cache: ^map[^MatNode][]f32 = nil,
) -> []f32 {
	for i in 0 ..< len(out_grads) {
		out_grads[i] = 0
	}

	local_activation_cache: map[^MatNode][]f32
	local_grad_cache: map[^MatNode][]f32

	act_cache := activation_cache if activation_cache != nil else &local_activation_cache
	grd_cache := grad_cache if grad_cache != nil else &local_grad_cache

	delete_mat_cache_contents(act_cache)
	delete_mat_cache_contents(grd_cache)

	val := eval_mat_grad_reverse_forward(node, binding, var_shapes, act_cache)

	out_shape := mat_node_shape(node, var_shapes)
	grd_cache[node] = mat_broadcast(1, mat_size(out_shape))

	eval_mat_grad_reverse_backward(node, var_shapes, out_grads, act_cache, grd_cache)

	if activation_cache == nil {
		delete_mat_cache_contents(&local_activation_cache)
		delete(local_activation_cache)
	}
	if grad_cache == nil {
		delete_mat_cache_contents(&local_grad_cache)
		delete(local_grad_cache)
	}

	return val
}

eval_mat_grad_reverse_forward :: proc(
	node: ^MatNode,
	binding: []f32,
	var_shapes: []MatShape,
	activation_cache: ^map[^MatNode][]f32,
) -> []f32 {
	if node in activation_cache {
		return mat_clone(activation_cache[node])
	}

	val: []f32
	switch n in node {
	case int:
		data := mat_var_data(n, binding, var_shapes)
		val = mat_clone(data)
	case ^MatConst:
		val = mat_clone(n.data)
	case MatOp:
		l_val := eval_mat_grad_reverse_forward(n.l, binding, var_shapes, activation_cache)
		l_shape := mat_node_shape(n.l, var_shapes)

		switch n.type {
		case .Add, .Sub, .Mul, .Div:
			r_val := eval_mat_grad_reverse_forward(n.r, binding, var_shapes, activation_cache)
			val = mat_elementwise_op(n.type, l_val, r_val)
		case .MatMul:
			r_val := eval_mat_grad_reverse_forward(n.r, binding, var_shapes, activation_cache)
			r_shape := mat_node_shape(n.r, var_shapes)
			val = mat_multiply(l_val, l_shape, r_val, r_shape)
		case .ReduceSum:
			val = mat_reduce_sum(l_val)
		}
	}

	activation_cache[node] = val
	return mat_clone(val)
}

eval_mat_grad_reverse_backward :: proc(
	node: ^MatNode,
	var_shapes: []MatShape,
	out_grads: []f32,
	activation_cache: ^map[^MatNode][]f32,
	grad_cache: ^map[^MatNode][]f32,
) {
	node_grad := grad_cache[node]

	switch n in node {
	case int:
		offset := mat_var_offset(n, var_shapes)
		size := mat_size(var_shapes[n])
		for i in 0 ..< size {
			out_grads[offset + i] += node_grad[i]
		}
	case ^MatConst:
		return
	case MatOp:
		l_shape := mat_node_shape(n.l, var_shapes)
		l_size := mat_size(l_shape)

		switch n.type {
		case .Add:
			if n.l in grad_cache {
				existing := grad_cache[n.l]
				new_grad := mat_elementwise_op(.Add, existing, node_grad)
				delete(existing)
				grad_cache[n.l] = new_grad
			} else {
				grad_cache[n.l] = mat_clone(node_grad)
			}
			if n.r in grad_cache {
				existing := grad_cache[n.r]
				new_grad := mat_elementwise_op(.Add, existing, node_grad)
				delete(existing)
				grad_cache[n.r] = new_grad
			} else {
				grad_cache[n.r] = mat_clone(node_grad)
			}

		case .Sub:
			if n.l in grad_cache {
				existing := grad_cache[n.l]
				new_grad := mat_elementwise_op(.Add, existing, node_grad)
				delete(existing)
				grad_cache[n.l] = new_grad
			} else {
				grad_cache[n.l] = mat_clone(node_grad)
			}
			neg_grad := mat_negate(node_grad)
			if n.r in grad_cache {
				existing := grad_cache[n.r]
				new_grad := mat_elementwise_op(.Add, existing, neg_grad)
				delete(existing)
				delete(neg_grad)
				grad_cache[n.r] = new_grad
			} else {
				grad_cache[n.r] = neg_grad
			}

		case .Mul:
			l_val := activation_cache[n.l]
			r_val := activation_cache[n.r]
			l_grad := mat_elementwise_op(.Mul, node_grad, r_val)
			r_grad := mat_elementwise_op(.Mul, node_grad, l_val)
			if n.l in grad_cache {
				existing := grad_cache[n.l]
				new_grad := mat_elementwise_op(.Add, existing, l_grad)
				delete(existing)
				delete(l_grad)
				grad_cache[n.l] = new_grad
			} else {
				grad_cache[n.l] = l_grad
			}
			if n.r in grad_cache {
				existing := grad_cache[n.r]
				new_grad := mat_elementwise_op(.Add, existing, r_grad)
				delete(existing)
				delete(r_grad)
				grad_cache[n.r] = new_grad
			} else {
				grad_cache[n.r] = r_grad
			}

		case .Div:
			l_val := activation_cache[n.l]
			r_val := activation_cache[n.r]
			// grad_L = grad_out / R
			l_grad := mat_elementwise_op(.Div, node_grad, r_val)
			// grad_R = -grad_out * L / R^2
			neg_grad := mat_negate(node_grad)
			defer delete(neg_grad)
			numer := mat_elementwise_op(.Mul, neg_grad, l_val)
			defer delete(numer)
			denom := mat_elementwise_op(.Mul, r_val, r_val)
			defer delete(denom)
			r_grad := mat_elementwise_op(.Div, numer, denom)
			if n.l in grad_cache {
				existing := grad_cache[n.l]
				new_grad := mat_elementwise_op(.Add, existing, l_grad)
				delete(existing)
				delete(l_grad)
				grad_cache[n.l] = new_grad
			} else {
				grad_cache[n.l] = l_grad
			}
			if n.r in grad_cache {
				existing := grad_cache[n.r]
				new_grad := mat_elementwise_op(.Add, existing, r_grad)
				delete(existing)
				delete(r_grad)
				grad_cache[n.r] = new_grad
			} else {
				grad_cache[n.r] = r_grad
			}

		case .MatMul:
			l_val := activation_cache[n.l]
			r_val := activation_cache[n.r]
			r_shape := mat_node_shape(n.r, var_shapes)
			// grad_L = grad_out @ R^T
			r_t := mat_transpose(r_val, r_shape)
			defer delete(r_t)
			out_shape := mat_node_shape(node, var_shapes)
			r_t_shape := MatShape{r_shape.c, r_shape.r}
			l_grad := mat_multiply(node_grad, out_shape, r_t, r_t_shape)
			// grad_R = L^T @ grad_out
			l_t := mat_transpose(l_val, l_shape)
			defer delete(l_t)
			l_t_shape := MatShape{l_shape.c, l_shape.r}
			r_grad := mat_multiply(l_t, l_t_shape, node_grad, out_shape)
			if n.l in grad_cache {
				existing := grad_cache[n.l]
				new_grad := mat_elementwise_op(.Add, existing, l_grad)
				delete(existing)
				delete(l_grad)
				grad_cache[n.l] = new_grad
			} else {
				grad_cache[n.l] = l_grad
			}
			if n.r in grad_cache {
				existing := grad_cache[n.r]
				new_grad := mat_elementwise_op(.Add, existing, r_grad)
				delete(existing)
				delete(r_grad)
				grad_cache[n.r] = new_grad
			} else {
				grad_cache[n.r] = r_grad
			}

		case .ReduceSum:
			broadcast_grad := mat_broadcast(node_grad[0], l_size)
			if n.l in grad_cache {
				existing := grad_cache[n.l]
				new_grad := mat_elementwise_op(.Add, existing, broadcast_grad)
				delete(existing)
				delete(broadcast_grad)
				grad_cache[n.l] = new_grad
			} else {
				grad_cache[n.l] = broadcast_grad
			}
		}

		eval_mat_grad_reverse_backward(n.l, var_shapes, out_grads, activation_cache, grad_cache)
		if n.r != nil && n.r != n.l {
			eval_mat_grad_reverse_backward(n.r, var_shapes, out_grads, activation_cache, grad_cache)
		}
	}
}

delete_mat_cache_contents :: proc(cache: ^map[^MatNode][]f32) {
	for _, v in cache {
		delete(v)
	}
	clear(cache)
}
