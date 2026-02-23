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
