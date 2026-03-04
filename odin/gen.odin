package main

import "base:intrinsics"

emit :: proc(out: ^[dynamic]VInstr, instrs: ..VInstr) {
	append(out, ..instrs)
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
			emit(&instrs, FLoad{addr = nodes[l_idx].activation_addr, dst = .Ret}, FLoad{addr = nodes[r_idx].activation_addr, dst = .R1})
			switch n.type {
			case .Add:
				emit(&instrs, FAdd{src1 = .Ret, src2 = .R1, dst = .Ret})
			case .Sub:
				emit(&instrs, FSub{src1 = .Ret, src2 = .R1, dst = .Ret})
			case .Mul:
				emit(&instrs, FMul{src1 = .Ret, src2 = .R1, dst = .Ret})
			case .Div:
				emit(&instrs, FDiv{src1 = .Ret, src2 = .R1, dst = .Ret})
			}
			emit(&instrs, FStore{addr = info.activation_addr, src = .Ret})
		case f32:
			emit(&instrs, FMovI{dst = .Ret, imm = n}, FStore{addr = info.activation_addr, src = .Ret})
		case int:
			emit(&instrs, FLoad{addr = n, dst = .Ret}, FStore{addr = info.activation_addr, src = .Ret})
		}
	}

	root_idx := len(nodes) - 1
	emit(&instrs, FMovI{dst = .Ret, imm = 1.0}, FStore{addr = nodes[root_idx].grad_addr, src = .Ret})

	#reverse for info in nodes {
		switch n in info.node {
		case Op:
			l_idx := node_to_idx[n.l]
			r_idx := node_to_idx[n.r]

			emit(&instrs, FLoad{addr = info.grad_addr, dst = .Ret})

			switch n.type {
			case .Add:
				emit(&instrs,
					FLoad{addr = nodes[l_idx].grad_addr, dst = .R1},
					FAdd{src1 = .R1, src2 = .Ret, dst = .R1},
					FStore{addr = nodes[l_idx].grad_addr, src = .R1},
					FLoad{addr = nodes[r_idx].grad_addr, dst = .R1},
					FAdd{src1 = .R1, src2 = .Ret, dst = .R1},
					FStore{addr = nodes[r_idx].grad_addr, src = .R1},
				)
			case .Sub:
				emit(&instrs,
					FLoad{addr = nodes[l_idx].grad_addr, dst = .R1},
					FAdd{src1 = .R1, src2 = .Ret, dst = .R1},
					FStore{addr = nodes[l_idx].grad_addr, src = .R1},
					FLoad{addr = nodes[r_idx].grad_addr, dst = .R1},
					FSub{src1 = .R1, src2 = .Ret, dst = .R1},
					FStore{addr = nodes[r_idx].grad_addr, src = .R1},
				)
			case .Mul:
				emit(&instrs,
					FLoad{addr = nodes[r_idx].activation_addr, dst = .R1},
					FMul{src1 = .Ret, src2 = .R1, dst = .R2},
					FLoad{addr = nodes[l_idx].grad_addr, dst = .R1},
					FAdd{src1 = .R1, src2 = .R2, dst = .R1},
					FStore{addr = nodes[l_idx].grad_addr, src = .R1},
					FLoad{addr = nodes[l_idx].activation_addr, dst = .R1},
					FMul{src1 = .Ret, src2 = .R1, dst = .R2},
					FLoad{addr = nodes[r_idx].grad_addr, dst = .R1},
					FAdd{src1 = .R1, src2 = .R2, dst = .R1},
					FStore{addr = nodes[r_idx].grad_addr, src = .R1},
				)
			case .Div:
			}

		case f32:
		case int:
			emit(&instrs,
				FLoad{addr = info.grad_addr, dst = .Ret},
				FLoad{addr = grad_offset + n, dst = .R1},
				FAdd{src1 = .R1, src2 = .Ret, dst = .R1},
				FStore{addr = grad_offset + n, src = .R1},
			)
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
		compile_forward_stack_helper(n.l, out)
		compile_forward_stack_helper(n.r, out)
		append(out, n.type)
	case f32:
		append(out, Push{imm = n})
	case int:
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
	}
}

// MatNode compilation to flat scalar instruction streams.
//
// Memory layout (shared by both compile_reverse_mat and compile_diff_vm_mat):
//
//   [0 .. vars_end)         variable storage (params then data vars), packed by var_shapes
//   [vars_end .. act_end)   activation storage in topo order; ReLU nodes store a 0/1 mask
//                           in mask_addr immediately before their output activation
//   [act_end .. total)      gradient storage parallel to [0 .. vars_end)
//
// The ReLU mask (1.0 if pre-activation > 0, else 0.0) is written during the forward pass
// so the backward pass can gate gradients without any branching.

MatNodeInfo :: struct {
	node:      ^MatNode,
	act_addr:  int,
	mask_addr: int, // ReLU nodes only; -1 otherwise
}

linearize_mat_tree :: proc(
	tree: ^MatNode,
	out: ^[dynamic]MatNodeInfo,
	visited: ^map[^MatNode]bool,
	addr: ^int,
	var_shapes: []MatShape,
) {
	if tree in visited {
		return
	}
	visited[tree] = true

	if op, ok := tree.(MatOp); ok {
		linearize_mat_tree(op.l, out, visited, addr, var_shapes)
		if op.r != nil {
			linearize_mat_tree(op.r, out, visited, addr, var_shapes)
		}
	}

	size := mat_size(mat_node_shape(tree, var_shapes))
	mask_addr := -1

	if op, ok := tree.(MatOp); ok && op.type == .ReLU {
		mask_addr = addr^
		addr^ += size
	}

	act_addr := addr^
	addr^ += size

	append(out, MatNodeInfo{node = tree, act_addr = act_addr, mask_addr = mask_addr})
}

mat_act_addr :: proc(node: ^MatNode, infos: []MatNodeInfo, node_to_idx: map[^MatNode]int) -> int {
	return infos[node_to_idx[node]].act_addr
}

// compile_reverse_mat compiles a MatNode graph to scalar VInstr (same type as compile_reverse).
// The result is executed by simulate() from asm.odin.
//
// Memory layout:
//   [0 .. vars_end)            variable storage (params then data vars)
//   [vars_end .. act_end)      activation storage (includes ReLU mask slots)
//   [act_end .. grad_end)      gradient storage — one slot per activation node element,
//                              followed by one slot per variable element
//                              Node grads at: act_end + (act_addr - vars_end)
//                              Var  grads at: act_end + (act_end - vars_end) + var_offset
compile_reverse_mat :: proc(tree: ^MatNode, var_shapes: []MatShape, num_trainable: int) -> CompiledReverse {
	visited := make(map[^MatNode]bool)
	defer delete(visited)

	vars_end := 0
	for shape in var_shapes {
		vars_end += mat_size(shape)
	}

	addr := vars_end
	nodes := [dynamic]MatNodeInfo{}
	linearize_mat_tree(tree, &nodes, &visited, &addr, var_shapes)

	act_end := addr
	// grad region: one slot per activation element + one slot per variable element
	act_region_size := act_end - vars_end
	grad_offset := act_end
	// node grads are at grad_offset + (node.act_addr - vars_end)
	// var grads  are at grad_offset + act_region_size + var_offset
	var_grad_base := grad_offset + act_region_size
	total_mem_size := var_grad_base + vars_end

	node_to_idx := make(map[^MatNode]int)
	defer delete(node_to_idx)
	for info, idx in nodes {
		node_to_idx[info.node] = idx
	}

	instrs := [dynamic]VInstr{}
	emit_mat_forward_vinstr(nodes[:], node_to_idx, var_shapes, &instrs)

	// seed root gradient
	root_node_grad := grad_offset + (nodes[len(nodes) - 1].act_addr - vars_end)
	append(&instrs, FMovI{imm = 1.0, dst = .Ret}, FStore{addr = root_node_grad, src = .Ret})

	emit_mat_backward_vinstr(nodes[:], node_to_idx, var_shapes, grad_offset, vars_end, var_grad_base, &instrs)

	return CompiledReverse {
		instrs          = instrs,
		num_bindings    = vars_end,
		num_activations = act_end - vars_end,
		grad_offset     = var_grad_base,
		total_mem_size  = total_mem_size,
	}
}

// compile_diff_vm_mat compiles a MatNode graph to scalar DiffInstr (forward pass only).
// The result is executed by diff_sim_forward; diff_sim_backward traverses the recorded tape
// to propagate gradients — no explicit backward instructions are needed.
//
// Memory layout:
//   [0 .. vars_end)        variable storage (params, then data vars)
//   [vars_end .. total)    activation storage (includes ReLU mask slots)
//
// diff_sim_backward writes into out_grads indexed by address. DLoad instructions in the
// tape have addresses spanning the full memory range (both vars and activations), so the
// caller must pass out_grads of size >= total_mem_size. Param gradients are at [0..vars_end).
compile_diff_vm_mat :: proc(
	tree: ^MatNode,
	var_shapes: []MatShape,
) -> (
	instrs: [dynamic]DiffInstr,
	total_mem_size: int,
) {
	visited := make(map[^MatNode]bool)
	defer delete(visited)

	vars_end := 0
	for shape in var_shapes {
		vars_end += mat_size(shape)
	}

	addr := vars_end
	nodes := [dynamic]MatNodeInfo{}
	linearize_mat_tree(tree, &nodes, &visited, &addr, var_shapes)

	total_mem_size = addr // just the var + activation region; grads are in out_grads

	node_to_idx := make(map[^MatNode]int)
	defer delete(node_to_idx)
	for info, idx in nodes {
		node_to_idx[info.node] = idx
	}

	instrs = [dynamic]DiffInstr{}
	emit_mat_forward_dinstr(nodes[:], node_to_idx, var_shapes, &instrs)

	return instrs, total_mem_size
}

emit_mat_forward_vinstr :: proc(
	nodes: []MatNodeInfo,
	node_to_idx: map[^MatNode]int,
	var_shapes: []MatShape,
	out: ^[dynamic]VInstr,
) {
	for info in nodes {
		shape := mat_node_shape(info.node, var_shapes)
		size := mat_size(shape)

		switch n in info.node {
		case int:
			base := mat_var_offset(n, var_shapes)
			for j in 0 ..< size {
				append(out, FLoad{addr = base + j, dst = .Ret}, FStore{addr = info.act_addr + j, src = .Ret})
			}

		case ^MatConst:
			for j in 0 ..< size {
				append(out, FMovI{imm = n.data[j], dst = .Ret}, FStore{addr = info.act_addr + j, src = .Ret})
			}

		case MatOp:
			l_act := mat_act_addr(n.l, nodes, node_to_idx)
			l_shape := mat_node_shape(n.l, var_shapes)

			switch n.type {
			case .MatMul:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				r_shape := mat_node_shape(n.r, var_shapes)
				for i in 0 ..< l_shape.r {
					for j in 0 ..< r_shape.c {
						append(out, FMovI{imm = 0, dst = .Ret})
						for k in 0 ..< l_shape.c {
							append(out,
								FLoad{addr = l_act + i * l_shape.c + k, dst = .R1},
								FLoad{addr = r_act + k * r_shape.c + j, dst = .R2},
								FMul{src1 = .R1, src2 = .R2, dst = .R1},
								FAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
							)
						}
						append(out, FStore{addr = info.act_addr + i * r_shape.c + j, src = .Ret})
					}
				}

			case .Add:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				for j in 0 ..< size {
					append(out,
						FLoad{addr = l_act + j, dst = .Ret},
						FLoad{addr = r_act + j, dst = .R1},
						FAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
						FStore{addr = info.act_addr + j, src = .Ret},
					)
				}

			case .Sub:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				for j in 0 ..< size {
					append(out,
						FLoad{addr = l_act + j, dst = .Ret},
						FLoad{addr = r_act + j, dst = .R1},
						FSub{src1 = .Ret, src2 = .R1, dst = .Ret},
						FStore{addr = info.act_addr + j, src = .Ret},
					)
				}

			case .Mul:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				for j in 0 ..< size {
					append(out,
						FLoad{addr = l_act + j, dst = .Ret},
						FLoad{addr = r_act + j, dst = .R1},
						FMul{src1 = .Ret, src2 = .R1, dst = .Ret},
						FStore{addr = info.act_addr + j, src = .Ret},
					)
				}

			case .Div:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				for j in 0 ..< size {
					append(out,
						FLoad{addr = l_act + j, dst = .Ret},
						FLoad{addr = r_act + j, dst = .R1},
						FDiv{src1 = .Ret, src2 = .R1, dst = .Ret},
						FStore{addr = info.act_addr + j, src = .Ret},
					)
				}

			case .ReLU:
				// write mask (1.0 if pre > 0, else 0.0) then the relu output
				for j in 0 ..< size {
					append(out,
						FLoad{addr = l_act + j, dst = .Ret},
						FReLU{src = .Ret, dst = .R1},        // R1 = max(0, pre)
						FMul{src1 = .R1, src2 = .R1, dst = .R2}, // R2 = relu^2, nonzero iff pre>0
						// mask: R2 > 0 ? 1 : 0  — use FReLU again then divide
						// Avoid division: store relu output as mask proxy.
						// Backward uses: grad * mask where mask = (relu_out > 0 ? 1 : 0).
						// Store relu_out in mask_addr; backward emits FReLU on it to get the mask.
						FStore{addr = info.mask_addr + j, src = .Ret}, // store pre-activation
						FStore{addr = info.act_addr + j, src = .R1},   // store relu output
					)
				}

			case .ReduceSum:
				append(out, FMovI{imm = 0, dst = .Ret})
				for j in 0 ..< mat_size(l_shape) {
					append(out,
						FLoad{addr = l_act + j, dst = .R1},
						FAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
					)
				}
				append(out, FStore{addr = info.act_addr, src = .Ret})

			case .Exp:
			}
		}
	}
}

// node_grad_addr computes the address of the gradient for a node in the grad region.
// Node grads are at grad_offset + (act_addr - vars_end).
node_grad_addr :: #force_inline proc(act_addr, grad_offset, vars_end: int) -> int {
	return grad_offset + (act_addr - vars_end)
}

emit_mat_backward_vinstr :: proc(
	nodes: []MatNodeInfo,
	node_to_idx: map[^MatNode]int,
	var_shapes: []MatShape,
	grad_offset: int,
	vars_end: int,
	var_grad_base: int,
	out: ^[dynamic]VInstr,
) {
	ng :: node_grad_addr

	#reverse for info in nodes {
		shape := mat_node_shape(info.node, var_shapes)
		size := mat_size(shape)
		g := ng(info.act_addr, grad_offset, vars_end)

		switch n in info.node {
		case int:
			param_base := mat_var_offset(n, var_shapes)
			for j in 0 ..< size {
				append(out,
					FLoad{addr = g + j, dst = .Ret},
					FLoad{addr = var_grad_base + param_base + j, dst = .R1},
					FAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
					FStore{addr = var_grad_base + param_base + j, src = .Ret},
				)
			}

		case ^MatConst:

		case MatOp:
			l_act := mat_act_addr(n.l, nodes, node_to_idx)
			l_shape := mat_node_shape(n.l, var_shapes)
			gl := ng(l_act, grad_offset, vars_end)

			switch n.type {
			case .MatMul:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				r_shape := mat_node_shape(n.r, var_shapes)
				gr := ng(r_act, grad_offset, vars_end)
				// dA[i,k] += sum_j dC[i,j] * B[k,j]
				for i in 0 ..< l_shape.r {
					for k in 0 ..< l_shape.c {
						append(out, FMovI{imm = 0, dst = .Ret})
						for j in 0 ..< r_shape.c {
							append(out,
								FLoad{addr = g + i * r_shape.c + j, dst = .R1},
								FLoad{addr = r_act + k * r_shape.c + j, dst = .R2},
								FMul{src1 = .R1, src2 = .R2, dst = .R1},
								FAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
							)
						}
						append(out,
							FLoad{addr = gl + i * l_shape.c + k, dst = .R1},
							FAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
							FStore{addr = gl + i * l_shape.c + k, src = .Ret},
						)
					}
				}
				// dB[k,j] += sum_i A[i,k] * dC[i,j]
				for k in 0 ..< r_shape.r {
					for j in 0 ..< r_shape.c {
						append(out, FMovI{imm = 0, dst = .Ret})
						for i in 0 ..< l_shape.r {
							append(out,
								FLoad{addr = l_act + i * l_shape.c + k, dst = .R1},
								FLoad{addr = g + i * r_shape.c + j, dst = .R2},
								FMul{src1 = .R1, src2 = .R2, dst = .R1},
								FAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
							)
						}
						append(out,
							FLoad{addr = gr + k * r_shape.c + j, dst = .R1},
							FAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
							FStore{addr = gr + k * r_shape.c + j, src = .Ret},
						)
					}
				}

			case .Add:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				gr := ng(r_act, grad_offset, vars_end)
				for j in 0 ..< size {
					append(out,
						FLoad{addr = g + j, dst = .Ret},
						FLoad{addr = gl + j, dst = .R1},
						FAdd{src1 = .R1, src2 = .Ret, dst = .R1},
						FStore{addr = gl + j, src = .R1},
						FLoad{addr = gr + j, dst = .R1},
						FAdd{src1 = .R1, src2 = .Ret, dst = .R1},
						FStore{addr = gr + j, src = .R1},
					)
				}

			case .Sub:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				gr := ng(r_act, grad_offset, vars_end)
				for j in 0 ..< size {
					append(out,
						FLoad{addr = g + j, dst = .Ret},
						FLoad{addr = gl + j, dst = .R1},
						FAdd{src1 = .R1, src2 = .Ret, dst = .R1},
						FStore{addr = gl + j, src = .R1},
						FLoad{addr = gr + j, dst = .R1},
						FSub{src1 = .R1, src2 = .Ret, dst = .R1},
						FStore{addr = gr + j, src = .R1},
					)
				}

			case .Mul:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				gr := ng(r_act, grad_offset, vars_end)
				for j in 0 ..< size {
					append(out,
						FLoad{addr = g + j, dst = .Ret},
						FLoad{addr = r_act + j, dst = .R1},
						FMul{src1 = .Ret, src2 = .R1, dst = .R2},
						FLoad{addr = gl + j, dst = .R1},
						FAdd{src1 = .R1, src2 = .R2, dst = .R1},
						FStore{addr = gl + j, src = .R1},
						FLoad{addr = l_act + j, dst = .R1},
						FMul{src1 = .Ret, src2 = .R1, dst = .R2},
						FLoad{addr = gr + j, dst = .R1},
						FAdd{src1 = .R1, src2 = .R2, dst = .R1},
						FStore{addr = gr + j, src = .R1},
					)
				}

			case .Div:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				gr := ng(r_act, grad_offset, vars_end)
				for j in 0 ..< size {
					append(out,
						FLoad{addr = g + j, dst = .Ret},
						FLoad{addr = r_act + j, dst = .R1},
						FDiv{src1 = .Ret, src2 = .R1, dst = .R2},
						FLoad{addr = gl + j, dst = .R1},
						FAdd{src1 = .R1, src2 = .R2, dst = .R1},
						FStore{addr = gl + j, src = .R1},
					)
					append(out,
						FLoad{addr = g + j, dst = .Ret},
						FLoad{addr = l_act + j, dst = .R1},
						FMul{src1 = .Ret, src2 = .R1, dst = .Ret},
						FLoad{addr = r_act + j, dst = .R1},
						FMul{src1 = .R1, src2 = .R1, dst = .R1},
						FDiv{src1 = .Ret, src2 = .R1, dst = .Ret},
						FLoad{addr = gr + j, dst = .R1},
						FSub{src1 = .R1, src2 = .Ret, dst = .R1},
						FStore{addr = gr + j, src = .R1},
					)
				}

			case .ReLU:
				// mask_addr holds the pre-activation stored during forward.
				// Branchless mask: relu(pre) / (relu(pre) + 1e-37)
				// For pre > 0: result ≈ 1.0 (since relu(pre) >> 1e-37)
				// For pre <= 0: 0 / 1e-37 = 0.0
				// 1e-37 is smaller than the smallest normal f32 (~1.18e-38 is subnormal),
				// so for any positive relu output this ratio rounds to 1.0 in f32.
				for j in 0 ..< size {
					append(out,
						FLoad{addr = info.mask_addr + j, dst = .Ret},
						FReLU{src = .Ret, dst = .R1},
						FMovI{imm = 1e-37, dst = .R2},
						FAdd{src1 = .R1, src2 = .R2, dst = .R2},
						FDiv{src1 = .R1, src2 = .R2, dst = .R1},
						FLoad{addr = g + j, dst = .Ret},
						FMul{src1 = .Ret, src2 = .R1, dst = .Ret},
						FLoad{addr = gl + j, dst = .R1},
						FAdd{src1 = .R1, src2 = .Ret, dst = .R1},
						FStore{addr = gl + j, src = .R1},
					)
				}

			case .ReduceSum:
				for j in 0 ..< mat_size(l_shape) {
					append(out,
						FLoad{addr = gl + j, dst = .Ret},
						FLoad{addr = g, dst = .R1},
						FAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
						FStore{addr = gl + j, src = .Ret},
					)
				}

			case .Exp:
			}
		}
	}
}

emit_mat_forward_dinstr :: proc(
	nodes: []MatNodeInfo,
	node_to_idx: map[^MatNode]int,
	var_shapes: []MatShape,
	out: ^[dynamic]DiffInstr,
) {
	for info in nodes {
		shape := mat_node_shape(info.node, var_shapes)
		size := mat_size(shape)

		switch n in info.node {
		case int:
			base := mat_var_offset(n, var_shapes)
			for j in 0 ..< size {
				append(out, DLoad{addr = base + j, dst = .Ret}, DStore{addr = info.act_addr + j, src = .Ret})
			}

		case ^MatConst:
			for j in 0 ..< size {
				append(out, DMovI{imm = n.data[j], dst = .Ret}, DStore{addr = info.act_addr + j, src = .Ret})
			}

		case MatOp:
			l_act := mat_act_addr(n.l, nodes, node_to_idx)
			l_shape := mat_node_shape(n.l, var_shapes)

			switch n.type {
			case .MatMul:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				r_shape := mat_node_shape(n.r, var_shapes)
				for i in 0 ..< l_shape.r {
					for j in 0 ..< r_shape.c {
						append(out, DMovI{imm = 0, dst = .Ret})
						for k in 0 ..< l_shape.c {
							append(out,
								DLoad{addr = l_act + i * l_shape.c + k, dst = .R1},
								DLoad{addr = r_act + k * r_shape.c + j, dst = .R2},
								DMul{src1 = .R1, src2 = .R2, dst = .R1},
								DAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
							)
						}
						append(out, DStore{addr = info.act_addr + i * r_shape.c + j, src = .Ret})
					}
				}

			case .Add:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				for j in 0 ..< size {
					append(out,
						DLoad{addr = l_act + j, dst = .Ret},
						DLoad{addr = r_act + j, dst = .R1},
						DAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
						DStore{addr = info.act_addr + j, src = .Ret},
					)
				}

			case .Sub:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				for j in 0 ..< size {
					append(out,
						DLoad{addr = l_act + j, dst = .Ret},
						DLoad{addr = r_act + j, dst = .R1},
						DSub{src1 = .Ret, src2 = .R1, dst = .Ret},
						DStore{addr = info.act_addr + j, src = .Ret},
					)
				}

			case .Mul:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				for j in 0 ..< size {
					append(out,
						DLoad{addr = l_act + j, dst = .Ret},
						DLoad{addr = r_act + j, dst = .R1},
						DMul{src1 = .Ret, src2 = .R1, dst = .Ret},
						DStore{addr = info.act_addr + j, src = .Ret},
					)
				}

			case .Div:
				r_act := mat_act_addr(n.r, nodes, node_to_idx)
				for j in 0 ..< size {
					append(out,
						DLoad{addr = l_act + j, dst = .Ret},
						DLoad{addr = r_act + j, dst = .R1},
						DDiv{src1 = .Ret, src2 = .R1, dst = .Ret},
						DStore{addr = info.act_addr + j, src = .Ret},
					)
				}

			case .ReLU:
				// DReLU records operand_vals[0] = pre-activation in the tape.
				// diff_sim_backward gates the gradient via that recorded value.
				for j in 0 ..< size {
					append(out,
						DLoad{addr = l_act + j, dst = .Ret},
						DReLU{src = .Ret, dst = .Ret},
						DStore{addr = info.act_addr + j, src = .Ret},
					)
				}

			case .ReduceSum:
				append(out, DMovI{imm = 0, dst = .Ret})
				for j in 0 ..< mat_size(l_shape) {
					append(out,
						DLoad{addr = l_act + j, dst = .R1},
						DAdd{src1 = .Ret, src2 = .R1, dst = .Ret},
					)
				}
				append(out, DStore{addr = info.act_addr, src = .Ret})

			case .Exp:
			}
		}
	}
}


