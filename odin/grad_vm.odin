package main

// what is the bare minimum required to make this possible


// this currently looks like a subset of what's in VInstr..
// consider refactor
GMov :: struct {
	dst: Reg,
	src: Reg,
}

GMovI :: struct {
	dst: Reg,
	imm: f32,
}

GAdd :: struct {
	src1: Reg,
	src2: Reg,
	dst:  Reg,
}

GMul :: struct {
	src1: Reg,
	src2: Reg,
	dst:  Reg,
}

GLoad :: struct {
	addr: int,
	dst:  Reg,
}

GStore :: struct {
	addr: int,
	src:  Reg,
}

// reg of floats ofc
VGradInstr :: union {
	GMovI,
	GMov,
	GAdd,
	GMul,
	GLoad,
	GStore,
	// branching needed eventually
}


sim_forward :: proc(instrs: []VGradInstr, mem: []f32) -> f32 {
	state: [Reg]f32

	// explicit PC is much slower this
	for inst in instrs {
		switch i in inst {
		case GAdd:
			state[i.dst] = state[i.src1] + state[i.src2]
		case GMul:
			state[i.dst] = state[i.src1] * state[i.src2]
		case GMov:
			state[i.dst] = state[i.src]
		case GMovI:
			state[i.dst] = i.imm
		case GLoad:
			state[i.dst] = mem[i.addr]
		case GStore:
			mem[i.addr] = state[i.src]
		}
	}

	return state[.Ret]
}

sim_backward :: proc(instrs: []VGradInstr, mem: []f32) -> f32 {
	return 0.0
}
