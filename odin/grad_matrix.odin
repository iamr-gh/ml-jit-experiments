package main

import "core:mem"
import "core:math"

MatShape :: struct {
	r, c: int,
}

MatOpType :: enum {
	Add,
	Sub,
	Mul,
	Div,
	Exp,
	ReLU,
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
		case .Add, .Sub, .Mul, .Div, .Exp, .ReLU:
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

mat_clone :: proc(data: []f32) -> []f32 {
	result := make([]f32, len(data))
	mem.copy(raw_data(result), raw_data(data), len(data) * size_of(f32))
	return result
}

mat_fill :: proc(val: f32, size: int) -> []f32 {
	result := make([]f32, size)
	for i in 0 ..< size {
		result[i] = val
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

mat_ew :: proc(type: MatOpType, l, r: []f32) -> []f32 {
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
		case .Exp, .ReLU, .MatMul, .ReduceSum:
			panic("not an elementwise op")
		}
	}
	return result
}

mat_exp :: proc(data: []f32) -> []f32 {
	result := make([]f32, len(data))
	for i in 0 ..< len(data) {
		result[i] = f32(math.exp(f64(data[i])))
	}
	return result
}

mat_relu :: proc(data: []f32) -> []f32 {
	result := make([]f32, len(data))
	for i in 0 ..< len(data) {
		if data[i] > 0 {
			result[i] = data[i]
		} else {
			result[i] = 0
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

mat_transpose :: proc(data: []f32, shape: MatShape) -> []f32 {
	result := make([]f32, len(data))
	for r in 0 ..< shape.r {
		for c in 0 ..< shape.c {
			result[c * shape.r + r] = data[r * shape.c + c]
		}
	}
	return result
}

mat_reduce_sum :: proc(data: []f32) -> []f32 {
	result := make([]f32, 1)
	for v in data {
		result[0] += v
	}
	return result
}

eval_mat :: proc(node: ^MatNode, binding: []f32, var_shapes: []MatShape) -> ([]f32, MatShape) {
	switch n in node {
	case int:
		return mat_clone(mat_var_data(n, binding, var_shapes)), var_shapes[n]
	case ^MatConst:
		return mat_clone(n.data), n.shape
	case MatOp:
		l_val, l_shape := eval_mat(n.l, binding, var_shapes)
		defer delete(l_val)
		switch n.type {
		case .Add, .Sub, .Mul, .Div:
			r_val, _ := eval_mat(n.r, binding, var_shapes)
			defer delete(r_val)
			return mat_ew(n.type, l_val, r_val), l_shape
		case .Exp:
			return mat_exp(l_val), l_shape
		case .ReLU:
			return mat_relu(l_val), l_shape
		case .MatMul:
			r_val, r_shape := eval_mat(n.r, binding, var_shapes)
			defer delete(r_val)
			return mat_multiply(l_val, l_shape, r_val, r_shape), MatShape{l_shape.r, r_shape.c}
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
		shape := var_shapes[n]
		size := mat_size(shape)
		return MatValGrad{mat_clone(mat_var_data(n, binding, var_shapes)), mat_fill(n == respect ? 1 : 0, size), shape}
	case ^MatConst:
		return MatValGrad{mat_clone(n.data), make([]f32, mat_size(n.shape)), n.shape}
	case MatOp:
		lvg := eval_mat_grad_forward(n.l, binding, var_shapes, respect)
		defer delete(lvg.val)
		defer delete(lvg.grad)

		switch n.type {
		case .Add, .Sub:
			rvg := eval_mat_grad_forward(n.r, binding, var_shapes, respect)
			defer delete(rvg.val)
			defer delete(rvg.grad)
			return MatValGrad{mat_ew(n.type, lvg.val, rvg.val), mat_ew(n.type, lvg.grad, rvg.grad), lvg.shape}

		case .Mul:
			rvg := eval_mat_grad_forward(n.r, binding, var_shapes, respect)
			defer delete(rvg.val)
			defer delete(rvg.grad)
			val := mat_ew(.Mul, lvg.val, rvg.val)
			t1 := mat_ew(.Mul, lvg.grad, rvg.val); defer delete(t1)
			t2 := mat_ew(.Mul, lvg.val, rvg.grad); defer delete(t2)
			return MatValGrad{val, mat_ew(.Add, t1, t2), lvg.shape}

		case .Div:
			rvg := eval_mat_grad_forward(n.r, binding, var_shapes, respect)
			defer delete(rvg.val)
			defer delete(rvg.grad)
			val := mat_ew(.Div, lvg.val, rvg.val)
			t1 := mat_ew(.Mul, lvg.grad, rvg.val); defer delete(t1)
			t2 := mat_ew(.Mul, lvg.val, rvg.grad); defer delete(t2)
			numer := mat_ew(.Sub, t1, t2); defer delete(numer)
			denom := mat_ew(.Mul, rvg.val, rvg.val); defer delete(denom)
			return MatValGrad{val, mat_ew(.Div, numer, denom), lvg.shape}

		case .Exp:
			val := mat_exp(lvg.val)
			grad := mat_ew(.Mul, val, lvg.grad)
			return MatValGrad{val, grad, lvg.shape}

		case .ReLU:
			val := mat_relu(lvg.val)
			grad := make([]f32, len(lvg.grad))
			for i in 0 ..< len(grad) {
				if lvg.val[i] > 0 {
					grad[i] = lvg.grad[i]
				}
			}
			return MatValGrad{val, grad, lvg.shape}

		case .MatMul:
			rvg := eval_mat_grad_forward(n.r, binding, var_shapes, respect)
			defer delete(rvg.val)
			defer delete(rvg.grad)
			val := mat_multiply(lvg.val, lvg.shape, rvg.val, rvg.shape)
			t1 := mat_multiply(lvg.grad, lvg.shape, rvg.val, rvg.shape); defer delete(t1)
			t2 := mat_multiply(lvg.val, lvg.shape, rvg.grad, rvg.shape); defer delete(t2)
			return MatValGrad{val, mat_ew(.Add, t1, t2), MatShape{lvg.shape.r, rvg.shape.c}}

		case .ReduceSum:
			return MatValGrad{mat_reduce_sum(lvg.val), mat_reduce_sum(lvg.grad), MatShape{1, 1}}
		}
	}
	return MatValGrad{nil, nil, MatShape{0, 0}}
}

eval_mat_grad_forward_all :: proc(
	node: ^MatNode,
	binding: []f32,
	var_shapes: []MatShape,
) -> ([]f32, MatShape, [][]f32) {
	grads := make([][]f32, len(var_shapes))
	val: []f32
	shape: MatShape
	for i in 0 ..< len(var_shapes) {
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

accumulate_grad :: proc(child: ^MatNode, grad: []f32, grad_cache: ^map[^MatNode][]f32) {
	if child in grad_cache {
		existing := grad_cache[child]
		grad_cache[child] = mat_ew(.Add, existing, grad)
		delete(existing)
		delete(grad)
	} else {
		grad_cache[child] = grad
	}
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
	grd_cache[node] = mat_fill(1, mat_size(out_shape))

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
		val = mat_clone(mat_var_data(n, binding, var_shapes))
	case ^MatConst:
		val = mat_clone(n.data)
	case MatOp:
		l_val := eval_mat_grad_reverse_forward(n.l, binding, var_shapes, activation_cache)
		l_shape := mat_node_shape(n.l, var_shapes)
		switch n.type {
		case .Add, .Sub, .Mul, .Div:
			r_val := eval_mat_grad_reverse_forward(n.r, binding, var_shapes, activation_cache)
			val = mat_ew(n.type, l_val, r_val)
		case .MatMul:
			r_val := eval_mat_grad_reverse_forward(n.r, binding, var_shapes, activation_cache)
			r_shape := mat_node_shape(n.r, var_shapes)
			val = mat_multiply(l_val, l_shape, r_val, r_shape)
		case .ReduceSum:
			val = mat_reduce_sum(l_val)
		case .Exp:
			val = mat_exp(l_val)
		case .ReLU:
			val = mat_relu(l_val)
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
		for i in 0 ..< mat_size(var_shapes[n]) {
			out_grads[offset + i] += node_grad[i]
		}
	case ^MatConst:
		return
	case MatOp:
		l_shape := mat_node_shape(n.l, var_shapes)

		switch n.type {
		case .Add:
			accumulate_grad(n.l, mat_clone(node_grad), grad_cache)
			accumulate_grad(n.r, mat_clone(node_grad), grad_cache)

		case .Sub:
			accumulate_grad(n.l, mat_clone(node_grad), grad_cache)
			neg := make([]f32, len(node_grad))
			for i in 0 ..< len(neg) { neg[i] = -node_grad[i] }
			accumulate_grad(n.r, neg, grad_cache)

		case .Mul:
			accumulate_grad(n.l, mat_ew(.Mul, node_grad, activation_cache[n.r]), grad_cache)
			accumulate_grad(n.r, mat_ew(.Mul, node_grad, activation_cache[n.l]), grad_cache)

		case .Div:
			r_val := activation_cache[n.r]
			l_grad := mat_ew(.Div, node_grad, r_val)
			neg := make([]f32, len(node_grad))
			for i in 0 ..< len(neg) { neg[i] = -node_grad[i] }
			defer delete(neg)
			numer := mat_ew(.Mul, neg, activation_cache[n.l]); defer delete(numer)
			denom := mat_ew(.Mul, r_val, r_val); defer delete(denom)
			accumulate_grad(n.l, l_grad, grad_cache)
			accumulate_grad(n.r, mat_ew(.Div, numer, denom), grad_cache)

		case .MatMul:
			r_shape := mat_node_shape(n.r, var_shapes)
			out_shape := mat_node_shape(node, var_shapes)
			r_t := mat_transpose(activation_cache[n.r], r_shape); defer delete(r_t)
			l_t := mat_transpose(activation_cache[n.l], l_shape); defer delete(l_t)
			accumulate_grad(n.l, mat_multiply(node_grad, out_shape, r_t, MatShape{r_shape.c, r_shape.r}), grad_cache)
			accumulate_grad(n.r, mat_multiply(l_t, MatShape{l_shape.c, l_shape.r}, node_grad, out_shape), grad_cache)

		case .ReduceSum:
			accumulate_grad(n.l, mat_fill(node_grad[0], mat_size(l_shape)), grad_cache)

		case .Exp:
			accumulate_grad(n.l, mat_ew(.Mul, node_grad, activation_cache[node]), grad_cache)

		case .ReLU:
			relu_grad := make([]f32, len(node_grad))
			for i in 0 ..< len(relu_grad) {
				if activation_cache[n.l][i] > 0 {
					relu_grad[i] = node_grad[i]
				}
			}
			accumulate_grad(n.l, relu_grad, grad_cache)
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
