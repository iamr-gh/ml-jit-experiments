package main

import "base:intrinsics"

MatNodeInfo :: struct {
	act_offset: int,
	grad_offset: int,
	shape:      MatShape,
}

linearize_mat_tree :: proc(node: ^MatNode, out: ^[dynamic]^MatNode, visited: ^map[^MatNode]bool) {
	if node in visited {
		return
	}
	switch n in node {
	case MatOp:
		linearize_mat_tree(n.l, out, visited)
		if n.r != nil {
			linearize_mat_tree(n.r, out, visited)
		}
	case ^MatConst:
	case int:
	}
	visited[node] = true
	append(out, node)
}

compile_mat_reverse :: proc(node: ^MatNode, var_shapes: []MatShape) -> CompiledReverse {
	var_total := 0
	for s in var_shapes {
		var_total += mat_size(s)
	}

	order := [dynamic]^MatNode{}
	visited := map[^MatNode]bool{}
	linearize_mat_tree(node, &order, &visited)

	node_info := map[^MatNode]MatNodeInfo{}

	next_act := var_total
	for n in order {
		shape := mat_node_shape(n, var_shapes)
		sz := mat_size(shape)
		node_info[n] = MatNodeInfo{act_offset = next_act, grad_offset = -1, shape = shape}
		next_act += sz
	}

	act_total := next_act - var_total
	grad_offset := next_act
	// [grad_offset .. grad_offset+var_total)          = binding grads (output)
	// [grad_offset+var_total .. grad_offset+var_total+act_total) = node activation grads
	total_mem_size := grad_offset + var_total + act_total

	for n, info in node_info {
		rel := info.act_offset - var_total
		node_info[n] = MatNodeInfo{
			act_offset  = info.act_offset,
			grad_offset = grad_offset + var_total + rel,
			shape       = info.shape,
		}
	}

	instrs := [dynamic]VInstr{}

	for n in order {
		info := node_info[n]
		switch nd in n {
		case int:
			src_off := mat_var_offset(nd, var_shapes)
			sz := mat_size(var_shapes[nd])
			for i in 0 ..< sz {
				append(&instrs, FLoad{addr = src_off + i, dst = .Ret})
				append(&instrs, FStore{addr = info.act_offset + i, src = .Ret})
			}
		case ^MatConst:
			for i in 0 ..< mat_size(nd.shape) {
				append(&instrs, FMovI{dst = .Ret, imm = nd.data[i]})
				append(&instrs, FStore{addr = info.act_offset + i, src = .Ret})
			}
		case MatOp:
			l_info := node_info[nd.l]
			switch nd.type {
			case .Add, .Sub, .Mul, .Div:
				r_info := node_info[nd.r]
				sz := mat_size(l_info.shape)
				for i in 0 ..< sz {
					append(&instrs, FLoad{addr = l_info.act_offset + i, dst = .Ret})
					append(&instrs, FLoad{addr = r_info.act_offset + i, dst = .R1})
					switch nd.type {
					case .Add:
						append(&instrs, FAdd{src1 = .Ret, src2 = .R1, dst = .Ret})
					case .Sub:
						append(&instrs, FSub{src1 = .Ret, src2 = .R1, dst = .Ret})
					case .Mul:
						append(&instrs, FMul{src1 = .Ret, src2 = .R1, dst = .Ret})
					case .Div:
						append(&instrs, FDiv{src1 = .Ret, src2 = .R1, dst = .Ret})
					case .MatMul, .ReduceSum:
					}
					append(&instrs, FStore{addr = info.act_offset + i, src = .Ret})
				}
			case .MatMul:
				r_info := node_info[nd.r]
				ls := l_info.shape
				rs := r_info.shape
				for i in 0 ..< ls.r {
					for j in 0 ..< rs.c {
						append(&instrs, FMovI{dst = .Ret, imm = 0})
						for k in 0 ..< ls.c {
							append(&instrs, FLoad{addr = l_info.act_offset + i * ls.c + k, dst = .R1})
							append(&instrs, FLoad{addr = r_info.act_offset + k * rs.c + j, dst = .R2})
							append(&instrs, FMul{src1 = .R1, src2 = .R2, dst = .R1})
							append(&instrs, FAdd{src1 = .Ret, src2 = .R1, dst = .Ret})
						}
						append(&instrs, FStore{addr = info.act_offset + i * rs.c + j, src = .Ret})
					}
				}
			case .ReduceSum:
				sz := mat_size(l_info.shape)
				append(&instrs, FMovI{dst = .Ret, imm = 0})
				for i in 0 ..< sz {
					append(&instrs, FLoad{addr = l_info.act_offset + i, dst = .R1})
					append(&instrs, FAdd{src1 = .Ret, src2 = .R1, dst = .Ret})
				}
				append(&instrs, FStore{addr = info.act_offset, src = .Ret})
			}
		}
	}

	root_info := node_info[node]
	append(&instrs, FMovI{dst = .Ret, imm = 1.0})
	append(&instrs, FStore{addr = root_info.grad_offset, src = .Ret})

	#reverse for n in order {
		info := node_info[n]
		switch nd in n {
		case int:
			var_grad_base := grad_offset + mat_var_offset(nd, var_shapes)
			sz := mat_size(var_shapes[nd])
			for i in 0 ..< sz {
				append(&instrs, FLoad{addr = info.grad_offset + i, dst = .Ret})
				append(&instrs, FLoad{addr = var_grad_base + i, dst = .R1})
				append(&instrs, FAdd{src1 = .R1, src2 = .Ret, dst = .R1})
				append(&instrs, FStore{addr = var_grad_base + i, src = .R1})
			}
		case ^MatConst:
		case MatOp:
			l_info := node_info[nd.l]
			switch nd.type {
			case .Add:
				sz := mat_size(info.shape)
				for i in 0 ..< sz {
					append(&instrs, FLoad{addr = info.grad_offset + i, dst = .Ret})
					append(&instrs, FLoad{addr = l_info.grad_offset + i, dst = .R1})
					append(&instrs, FAdd{src1 = .R1, src2 = .Ret, dst = .R1})
					append(&instrs, FStore{addr = l_info.grad_offset + i, src = .R1})
				}
				r_info := node_info[nd.r]
				for i in 0 ..< sz {
					append(&instrs, FLoad{addr = info.grad_offset + i, dst = .Ret})
					append(&instrs, FLoad{addr = r_info.grad_offset + i, dst = .R1})
					append(&instrs, FAdd{src1 = .R1, src2 = .Ret, dst = .R1})
					append(&instrs, FStore{addr = r_info.grad_offset + i, src = .R1})
				}
			case .Sub:
				sz := mat_size(info.shape)
				for i in 0 ..< sz {
					append(&instrs, FLoad{addr = info.grad_offset + i, dst = .Ret})
					append(&instrs, FLoad{addr = l_info.grad_offset + i, dst = .R1})
					append(&instrs, FAdd{src1 = .R1, src2 = .Ret, dst = .R1})
					append(&instrs, FStore{addr = l_info.grad_offset + i, src = .R1})
				}
				r_info := node_info[nd.r]
				for i in 0 ..< sz {
					append(&instrs, FLoad{addr = info.grad_offset + i, dst = .Ret})
					append(&instrs, FLoad{addr = r_info.grad_offset + i, dst = .R1})
					append(&instrs, FSub{src1 = .R1, src2 = .Ret, dst = .R1})
					append(&instrs, FStore{addr = r_info.grad_offset + i, src = .R1})
				}
			case .Mul:
				r_info := node_info[nd.r]
				sz := mat_size(info.shape)
				for i in 0 ..< sz {
					append(&instrs, FLoad{addr = info.grad_offset + i, dst = .Ret})
					append(&instrs, FLoad{addr = r_info.act_offset + i, dst = .R1})
					append(&instrs, FMul{src1 = .Ret, src2 = .R1, dst = .R2})
					append(&instrs, FLoad{addr = l_info.grad_offset + i, dst = .R1})
					append(&instrs, FAdd{src1 = .R1, src2 = .R2, dst = .R1})
					append(&instrs, FStore{addr = l_info.grad_offset + i, src = .R1})
				}
				for i in 0 ..< sz {
					append(&instrs, FLoad{addr = info.grad_offset + i, dst = .Ret})
					append(&instrs, FLoad{addr = l_info.act_offset + i, dst = .R1})
					append(&instrs, FMul{src1 = .Ret, src2 = .R1, dst = .R2})
					append(&instrs, FLoad{addr = r_info.grad_offset + i, dst = .R1})
					append(&instrs, FAdd{src1 = .R1, src2 = .R2, dst = .R1})
					append(&instrs, FStore{addr = r_info.grad_offset + i, src = .R1})
				}
			case .Div:
				r_info := node_info[nd.r]
				sz := mat_size(info.shape)
				for i in 0 ..< sz {
					append(&instrs, FLoad{addr = info.grad_offset + i, dst = .Ret})
					append(&instrs, FLoad{addr = r_info.act_offset + i, dst = .R1})
					append(&instrs, FDiv{src1 = .Ret, src2 = .R1, dst = .R2})
					append(&instrs, FLoad{addr = l_info.grad_offset + i, dst = .R1})
					append(&instrs, FAdd{src1 = .R1, src2 = .R2, dst = .R1})
					append(&instrs, FStore{addr = l_info.grad_offset + i, src = .R1})

					append(&instrs, FLoad{addr = info.grad_offset + i, dst = .Ret})
					append(&instrs, FLoad{addr = l_info.act_offset + i, dst = .R1})
					append(&instrs, FLoad{addr = r_info.act_offset + i, dst = .R2})
					append(&instrs, FMul{src1 = .Ret, src2 = .R1, dst = .R3})
					append(&instrs, FMul{src1 = .R2, src2 = .R2, dst = .R4})
					append(&instrs, FDiv{src1 = .R3, src2 = .R4, dst = .R3})
					append(&instrs, FLoad{addr = r_info.grad_offset + i, dst = .R1})
					append(&instrs, FSub{src1 = .R1, src2 = .R3, dst = .R1})
					append(&instrs, FStore{addr = r_info.grad_offset + i, src = .R1})
				}
			case .MatMul:
				r_info := node_info[nd.r]
				ls := l_info.shape
				rs := r_info.shape
				os := info.shape
				for i in 0 ..< ls.r {
					for k in 0 ..< ls.c {
						for j in 0 ..< rs.c {
							append(&instrs, FLoad{addr = info.grad_offset + i * os.c + j, dst = .Ret})
							append(&instrs, FLoad{addr = r_info.act_offset + k * rs.c + j, dst = .R1})
							append(&instrs, FMul{src1 = .Ret, src2 = .R1, dst = .R2})
							append(&instrs, FLoad{addr = l_info.grad_offset + i * ls.c + k, dst = .R1})
							append(&instrs, FAdd{src1 = .R1, src2 = .R2, dst = .R1})
							append(&instrs, FStore{addr = l_info.grad_offset + i * ls.c + k, src = .R1})
						}
					}
				}
				for k in 0 ..< ls.c {
					for j in 0 ..< rs.c {
						for i in 0 ..< ls.r {
							append(&instrs, FLoad{addr = info.grad_offset + i * os.c + j, dst = .Ret})
							append(&instrs, FLoad{addr = l_info.act_offset + i * ls.c + k, dst = .R1})
							append(&instrs, FMul{src1 = .Ret, src2 = .R1, dst = .R2})
							append(&instrs, FLoad{addr = r_info.grad_offset + k * rs.c + j, dst = .R1})
							append(&instrs, FAdd{src1 = .R1, src2 = .R2, dst = .R1})
							append(&instrs, FStore{addr = r_info.grad_offset + k * rs.c + j, src = .R1})
						}
					}
				}
			case .ReduceSum:
				sz := mat_size(l_info.shape)
				for i in 0 ..< sz {
					append(&instrs, FLoad{addr = info.grad_offset, dst = .Ret})
					append(&instrs, FLoad{addr = l_info.grad_offset + i, dst = .R1})
					append(&instrs, FAdd{src1 = .R1, src2 = .Ret, dst = .R1})
					append(&instrs, FStore{addr = l_info.grad_offset + i, src = .R1})
				}
			}
		}
	}

	return CompiledReverse {
		instrs          = instrs,
		num_bindings    = var_total,
		num_activations = act_total,
		grad_offset     = grad_offset,
		total_mem_size  = total_mem_size,
	}
}

compile_mat_diff_vm :: proc(node: ^MatNode, var_shapes: []MatShape) -> [dynamic]DiffInstr {
	var_total := 0
	for s in var_shapes {
		var_total += mat_size(s)
	}

	order := [dynamic]^MatNode{}
	visited := map[^MatNode]bool{}
	linearize_mat_tree(node, &order, &visited)

	act_offsets := map[^MatNode]int{}
	act_shapes  := map[^MatNode]MatShape{}

	next_act := var_total
	for n in order {
		shape := mat_node_shape(n, var_shapes)
		act_offsets[n] = next_act
		act_shapes[n]  = shape
		next_act += mat_size(shape)
	}

	out := [dynamic]DiffInstr{}

	for n in order {
		off := act_offsets[n]
		switch nd in n {
		case int:
			src_off := mat_var_offset(nd, var_shapes)
			sz := mat_size(var_shapes[nd])
			for i in 0 ..< sz {
				append(&out, DLoad{addr = src_off + i, dst = .Ret})
				append(&out, DStore{addr = off + i, src = .Ret})
			}
		case ^MatConst:
			for i in 0 ..< mat_size(nd.shape) {
				append(&out, DMovI{dst = .Ret, imm = nd.data[i]})
				append(&out, DStore{addr = off + i, src = .Ret})
			}
		case MatOp:
			l_off   := act_offsets[nd.l]
			l_shape := act_shapes[nd.l]
			switch nd.type {
			case .Add, .Sub, .Mul, .Div:
				r_off := act_offsets[nd.r]
				sz := mat_size(l_shape)
				for i in 0 ..< sz {
					append(&out, DLoad{addr = l_off + i, dst = .Ret})
					append(&out, DMov{src = .Ret, dst = .R1})
					append(&out, DLoad{addr = r_off + i, dst = .Ret})
					switch nd.type {
					case .Add:
						append(&out, DAdd{src1 = .R1, src2 = .Ret, dst = .Ret})
					case .Sub:
						append(&out, DSub{src1 = .R1, src2 = .Ret, dst = .Ret})
					case .Mul:
						append(&out, DMul{src1 = .R1, src2 = .Ret, dst = .Ret})
					case .Div:
						append(&out, DDiv{src1 = .R1, src2 = .Ret, dst = .Ret})
					case .MatMul, .ReduceSum:
					}
					append(&out, DStore{addr = off + i, src = .Ret})
				}
			case .MatMul:
				r_off   := act_offsets[nd.r]
				r_shape := act_shapes[nd.r]
				out_shape := act_shapes[n]
				for i in 0 ..< l_shape.r {
					for j in 0 ..< r_shape.c {
						append(&out, DMovI{dst = .Ret, imm = 0})
						for k in 0 ..< l_shape.c {
							append(&out, DLoad{addr = l_off + i * l_shape.c + k, dst = .R1})
							append(&out, DMov{src = .Ret, dst = .R2})
							append(&out, DLoad{addr = r_off + k * r_shape.c + j, dst = .Ret})
							append(&out, DMul{src1 = .R1, src2 = .Ret, dst = .R1})
							append(&out, DAdd{src1 = .R2, src2 = .R1, dst = .Ret})
						}
						append(&out, DStore{addr = off + i * out_shape.c + j, src = .Ret})
					}
				}
			case .ReduceSum:
				sz := mat_size(l_shape)
				append(&out, DMovI{dst = .Ret, imm = 0})
				for i in 0 ..< sz {
					append(&out, DMov{src = .Ret, dst = .R1})
					append(&out, DLoad{addr = l_off + i, dst = .Ret})
					append(&out, DAdd{src1 = .R1, src2 = .Ret, dst = .Ret})
				}
				append(&out, DStore{addr = off, src = .Ret})
			}
		}
	}

	return out
}

CompiledReverse :: struct {
	instrs:          [dynamic]VInstr,
	num_bindings:    int,
	num_activations: int,
	grad_offset:     int,
	total_mem_size:  int,
}

NodeInfo :: struct {
	node:            ^Node,
	activation_addr: int,
	grad_addr:       int,
}

linearize_tree :: proc(tree: ^Node, out: ^[dynamic]NodeInfo, next_addr: ^int) {
	switch n in tree {
	case Op:
		linearize_tree(n.l, out, next_addr)
		linearize_tree(n.r, out, next_addr)
	case f32:
	case int:
	}
	addr := next_addr^
	next_addr^ += 1
	append(out, NodeInfo{node = tree, activation_addr = addr, grad_addr = -1})
}

compile_reverse :: proc(tree: ^Node, num_bindings: int) -> CompiledReverse {
	nodes := [dynamic]NodeInfo{}
	next_addr := num_bindings

	linearize_tree(tree, &nodes, &next_addr)

	num_activations := next_addr - num_bindings
	grad_offset := next_addr
	total_mem_size := grad_offset + num_bindings

	for i in 0 ..< len(nodes) {
		nodes[i].grad_addr = grad_offset + num_bindings + i
	}
	total_mem_size = grad_offset + num_bindings + len(nodes)

	node_to_idx := map[^Node]int{}
	for info, idx in nodes {
		node_to_idx[info.node] = idx
	}

	instrs := [dynamic]VInstr{}

	for info in nodes {
		switch n in info.node {
		case Op:
			l_idx := node_to_idx[n.l]
			r_idx := node_to_idx[n.r]
			append(&instrs, FLoad{addr = nodes[l_idx].activation_addr, dst = .Ret})
			append(&instrs, FLoad{addr = nodes[r_idx].activation_addr, dst = .R1})
			switch n.type {
			case .Add:
				append(&instrs, FAdd{src1 = .Ret, src2 = .R1, dst = .Ret})
			case .Sub:
				append(&instrs, FSub{src1 = .Ret, src2 = .R1, dst = .Ret})
			case .Mul:
				append(&instrs, FMul{src1 = .Ret, src2 = .R1, dst = .Ret})
			case .Div:
				append(&instrs, FDiv{src1 = .Ret, src2 = .R1, dst = .Ret})
			}
			append(&instrs, FStore{addr = info.activation_addr, src = .Ret})
		case f32:
			append(&instrs, FMovI{dst = .Ret, imm = n})
			append(&instrs, FStore{addr = info.activation_addr, src = .Ret})
		case int:
			append(&instrs, FLoad{addr = n, dst = .Ret})
			append(&instrs, FStore{addr = info.activation_addr, src = .Ret})
		}
	}

	root_idx := len(nodes) - 1
	append(&instrs, FMovI{dst = .Ret, imm = 1.0})
	append(&instrs, FStore{addr = nodes[root_idx].grad_addr, src = .Ret})

	#reverse for info in nodes {
		switch n in info.node {
		case Op:
			l_idx := node_to_idx[n.l]
			r_idx := node_to_idx[n.r]

			append(&instrs, FLoad{addr = info.grad_addr, dst = .Ret})

			switch n.type {
			case .Add:
				append(&instrs, FLoad{addr = nodes[l_idx].grad_addr, dst = .R1})
				append(&instrs, FAdd{src1 = .R1, src2 = .Ret, dst = .R1})
				append(&instrs, FStore{addr = nodes[l_idx].grad_addr, src = .R1})

				append(&instrs, FLoad{addr = nodes[r_idx].grad_addr, dst = .R1})
				append(&instrs, FAdd{src1 = .R1, src2 = .Ret, dst = .R1})
				append(&instrs, FStore{addr = nodes[r_idx].grad_addr, src = .R1})

			case .Sub:
				append(&instrs, FLoad{addr = nodes[l_idx].grad_addr, dst = .R1})
				append(&instrs, FAdd{src1 = .R1, src2 = .Ret, dst = .R1})
				append(&instrs, FStore{addr = nodes[l_idx].grad_addr, src = .R1})

				append(&instrs, FLoad{addr = nodes[r_idx].grad_addr, dst = .R1})
				append(&instrs, FSub{src1 = .R1, src2 = .Ret, dst = .R1})
				append(&instrs, FStore{addr = nodes[r_idx].grad_addr, src = .R1})

			case .Mul:
				append(&instrs, FLoad{addr = nodes[r_idx].activation_addr, dst = .R1})
				append(&instrs, FMul{src1 = .Ret, src2 = .R1, dst = .R2})
				append(&instrs, FLoad{addr = nodes[l_idx].grad_addr, dst = .R1})
				append(&instrs, FAdd{src1 = .R1, src2 = .R2, dst = .R1})
				append(&instrs, FStore{addr = nodes[l_idx].grad_addr, src = .R1})

				append(&instrs, FLoad{addr = nodes[l_idx].activation_addr, dst = .R1})
				append(&instrs, FMul{src1 = .Ret, src2 = .R1, dst = .R2})
				append(&instrs, FLoad{addr = nodes[r_idx].grad_addr, dst = .R1})
				append(&instrs, FAdd{src1 = .R1, src2 = .R2, dst = .R1})
				append(&instrs, FStore{addr = nodes[r_idx].grad_addr, src = .R1})

			case .Div:
			}

		case f32:
		case int:
			append(&instrs, FLoad{addr = info.grad_addr, dst = .Ret})
			append(&instrs, FLoad{addr = grad_offset + n, dst = .R1})
			append(&instrs, FAdd{src1 = .R1, src2 = .Ret, dst = .R1})
			append(&instrs, FStore{addr = grad_offset + n, src = .R1})
		}
	}

	return CompiledReverse {
		instrs          = instrs,
		num_bindings    = num_bindings,
		num_activations = num_activations,
		grad_offset     = grad_offset,
		total_mem_size  = total_mem_size,
	}
}

compile_diff_vm :: proc(tree: ^Node) -> [dynamic]DiffInstr {
	out := [dynamic]DiffInstr{}
	compile_diff_vm_helper(tree, &out)
	return out
}

linearize_scalar_tree :: proc(
	node: ^Node,
	out: ^[dynamic]^Node,
	visited: ^map[^Node]bool,
) {
	if node in visited {
		return
	}
	switch n in node {
	case Op:
		linearize_scalar_tree(n.l, out, visited)
		linearize_scalar_tree(n.r, out, visited)
	case f32:
	case int:
	}
	visited[node] = true
	append(out, node)
}

compile_diff_vm_mem :: proc(tree: ^Node, num_bindings: int) -> ([dynamic]DiffInstr, int) {
	order := [dynamic]^Node{}
	visited := map[^Node]bool{}
	linearize_scalar_tree(tree, &order, &visited)

	act_offsets := map[^Node]int{}
	next_act := num_bindings
	for n in order {
		act_offsets[n] = next_act
		next_act += 1
	}
	mem_size := next_act

	out := [dynamic]DiffInstr{}
	for n in order {
		off := act_offsets[n]
		switch nd in n {
		case int:
			append(&out, DLoad{addr = nd, dst = .Ret})
			append(&out, DStore{addr = off, src = .Ret})
		case f32:
			append(&out, DMovI{dst = .Ret, imm = nd})
			append(&out, DStore{addr = off, src = .Ret})
		case Op:
			l_off := act_offsets[nd.l]
			r_off := act_offsets[nd.r]
			append(&out, DLoad{addr = l_off, dst = .Ret})
			append(&out, DMov{src = .Ret, dst = .R1})
			append(&out, DLoad{addr = r_off, dst = .Ret})
			switch nd.type {
			case .Add:
				append(&out, DAdd{src1 = .R1, src2 = .Ret, dst = .Ret})
			case .Sub:
				append(&out, DSub{src1 = .R1, src2 = .Ret, dst = .Ret})
			case .Mul:
				append(&out, DMul{src1 = .R1, src2 = .Ret, dst = .Ret})
			case .Div:
				append(&out, DDiv{src1 = .R1, src2 = .Ret, dst = .Ret})
			}
			append(&out, DStore{addr = off, src = .Ret})
		}
	}

	return out, mem_size
}

compile_diff_vm_helper :: proc(tree: ^Node, out: ^[dynamic]DiffInstr) {
	switch n in tree {
	case Op:
		compile_diff_vm_helper(n.r, out)
		append(out, DMov{src = .Ret, dst = .R1})
		compile_diff_vm_helper(n.l, out)
		switch n.type {
		case .Add:
			append(out, DAdd{src1 = .Ret, src2 = .R1, dst = .Ret})
		case .Sub:
			append(out, DSub{src1 = .Ret, src2 = .R1, dst = .Ret})
		case .Mul:
			append(out, DMul{src1 = .Ret, src2 = .R1, dst = .Ret})
		case .Div:
			append(out, DDiv{src1 = .Ret, src2 = .R1, dst = .Ret})
		}
	case f32:
		append(out, DMovI{dst = .Ret, imm = n})
	case int:
		append(out, DLoad{addr = n, dst = .Ret})
	}
}

compile_forward :: proc(tree: ^Node) -> [dynamic]VInstr {
	out := [dynamic]VInstr{}
	compile_forward_helper(tree, &out)
	return out
}

compile_forward_helper :: proc(tree: ^Node, out: ^[dynamic]VInstr) {
	switch n in tree {
	case Op:
		// works if l is always constant or var
		// if it's not, we'll need to add stack
		compile_forward_helper(n.r, out)
		append(out, FMov{src = .Ret, dst = .R1})
		compile_forward_helper(n.l, out)
		switch n.type {
		case .Add:
			append(out, FAdd{src1 = .Ret, src2 = .R1, dst = .Ret})
		case .Sub:
			append(out, FSub{src1 = .Ret, src2 = .R1, dst = .Ret})
		case .Mul:
			append(out, FMul{src1 = .Ret, src2 = .R1, dst = .Ret})
		case .Div:
			append(out, FDiv{src1 = .Ret, src2 = .R1, dst = .Ret})
		}
	case f32:
		append(out, FMovI{dst = .Ret, imm = n})
	case int:
	// need to read from inputted mem location
	}
}

compile_forward_stack :: proc(tree: ^Node) -> [dynamic]VInstrStack {
	out := [dynamic]VInstrStack{}
	compile_forward_stack_helper(tree, &out)
	return out
}
compile_forward_stack_helper :: proc(tree: ^Node, out: ^[dynamic]VInstrStack) {
	switch n in tree {
	case Op:
		// works if l is always constant or var
		// if it's not, we'll need to add stack
		compile_forward_stack_helper(n.l, out)
		compile_forward_stack_helper(n.r, out)
		append(out, n.type)
	case f32:
		append(out, Push{imm = n})
	case int:
	// need to read from inputted mem location
	}

}

compile_forward_tiny :: proc(tree: ^Node) -> [dynamic]VInstrStackTiny {
	out := [dynamic]VInstrStackTiny{}
	compile_forward_tiny_helper(tree, &out)
	return out
}
compile_forward_tiny_helper :: proc(tree: ^Node, out: ^[dynamic]VInstrStackTiny) {
	switch n in tree {
	case Op:
		// works if l is always constant or var
		// if it's not, we'll need to add stack
		compile_forward_tiny_helper(n.l, out)
		compile_forward_tiny_helper(n.r, out)
		val: i64
		switch n.type {
		case .Add:
			val = 0b1
		case .Sub:
			val = 0b10
		case .Mul:
			val = 0b11
		case .Div:
			val = 0b100
		}
		append(out, val)
	case f32:
		bits := transmute(u32)n
		append(out, i64(bits) << 32)
	case int:
	// need to read from inputted mem location
	}

}
