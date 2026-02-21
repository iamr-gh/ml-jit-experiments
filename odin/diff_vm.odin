package main

DMovI :: struct {
	imm: f32,
	dst: Reg,
}

DMov :: struct {
	src: Reg,
	dst: Reg,
}

DLoad :: struct {
	addr: int,
	dst:  Reg,
}

DStore :: struct {
	addr: int,
	src:  Reg,
}

DAdd :: struct {
	src1: Reg,
	src2: Reg,
	dst:  Reg,
}

DMul :: struct {
	src1: Reg,
	src2: Reg,
	dst:  Reg,
}

DiffInstr :: union {
	DMovI,
	DMov,
	DLoad,
	DStore,
	DAdd,
	DMul,
}

TapeEntry :: struct {
	instr:        DiffInstr,
	operand_vals: [2]f32,
}

diff_sim_forward :: proc(instrs: []DiffInstr, mem: []f32) -> (f32, []TapeEntry) {
	state: [Reg]f32
	tape := make([]TapeEntry, len(instrs))

	for inst, i in instrs {
		switch instr in inst {
		case DMovI:
			tape[i] = TapeEntry{instr = inst}
			state[instr.dst] = instr.imm
		case DMov:
			tape[i] = TapeEntry{instr = inst}
			state[instr.dst] = state[instr.src]
		case DLoad:
			tape[i] = TapeEntry{instr = inst}
			state[instr.dst] = mem[instr.addr]
		case DStore:
			tape[i] = TapeEntry{instr = inst, operand_vals = {state[instr.src], 0}}
			mem[instr.addr] = state[instr.src]
		case DAdd:
			tape[i] = TapeEntry{instr = inst}
			state[instr.dst] = state[instr.src1] + state[instr.src2]
		case DMul:
			tape[i] = TapeEntry{instr = inst, operand_vals = {state[instr.src1], state[instr.src2]}}
			state[instr.dst] = state[instr.src1] * state[instr.src2]
		}
	}

	return state[.Ret], tape
}

diff_sim_backward :: proc(tape: []TapeEntry, out_grads: []f32, seed: f32 = 1.0) {
	reg_grads: [Reg]f32
	reg_grads[.Ret] = seed

	#reverse for entry in tape {
		switch instr in entry.instr {
		case DMovI:
			reg_grads[instr.dst] = 0
		case DMov:
			reg_grads[instr.src] += reg_grads[instr.dst]
			reg_grads[instr.dst] = 0
		case DLoad:
			out_grads[instr.addr] += reg_grads[instr.dst]
			reg_grads[instr.dst] = 0
		case DStore:
			reg_grads[instr.src] += out_grads[instr.addr]
		case DAdd:
			reg_grads[instr.src1] += reg_grads[instr.dst]
			reg_grads[instr.src2] += reg_grads[instr.dst]
			reg_grads[instr.dst] = 0
		case DMul:
			reg_grads[instr.src1] += reg_grads[instr.dst] * entry.operand_vals[1]
			reg_grads[instr.src2] += reg_grads[instr.dst] * entry.operand_vals[0]
			reg_grads[instr.dst] = 0
		}
	}
}
