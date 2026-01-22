package main

import "base:intrinsics"

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
