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

DSub :: struct {
	src1: Reg,
	src2: Reg,
	dst:  Reg,
}

DMul :: struct {
	src1: Reg,
	src2: Reg,
	dst:  Reg,
}

DDiv :: struct {
	src1: Reg,
	src2: Reg,
	dst:  Reg,
}

DReLU :: struct {
	src: Reg,
	dst: Reg,
}

DiffInstr :: union {
	DMovI,
	DMov,
	DLoad,
	DStore,
	DAdd,
	DSub,
	DMul,
	DDiv,
	DReLU,
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
		case DSub:
			tape[i] = TapeEntry{instr = inst}
			state[instr.dst] = state[instr.src1] - state[instr.src2]
		case DMul:
			tape[i] = TapeEntry{instr = inst, operand_vals = {state[instr.src1], state[instr.src2]}}
			state[instr.dst] = state[instr.src1] * state[instr.src2]
		case DDiv:
			tape[i] = TapeEntry{instr = inst, operand_vals = {state[instr.src1], state[instr.src2]}}
			state[instr.dst] = state[instr.src1] / state[instr.src2]
		case DReLU:
			tape[i] = TapeEntry{instr = inst, operand_vals = {state[instr.src], 0}}
			state[instr.dst] = max(0, state[instr.src])
		}
	}

	return state[.Ret], tape
}

// note there is some very particular behavior of out grads with respect to addresses
diff_sim_backward :: proc(tape: []TapeEntry, out_grads: []f32, seed: f32 = 1.0) {
	reg_grads: [Reg]f32
	reg_grads[.Ret] = seed

	#reverse for entry in tape {
		switch instr in entry.instr {
		case DMovI:
			reg_grads[instr.dst] = 0
		case DMov:
			g := reg_grads[instr.dst]
			reg_grads[instr.dst] = 0
			reg_grads[instr.src] += g
		case DLoad:
			out_grads[instr.addr] += reg_grads[instr.dst]
			reg_grads[instr.dst] = 0
		case DStore:
			reg_grads[instr.src] += out_grads[instr.addr]
		case DAdd:
			g := reg_grads[instr.dst]
			reg_grads[instr.dst] = 0
			reg_grads[instr.src1] += g
			reg_grads[instr.src2] += g
		case DSub:
			g := reg_grads[instr.dst]
			reg_grads[instr.dst] = 0
			reg_grads[instr.src1] += g
			reg_grads[instr.src2] -= g
		case DMul:
			g := reg_grads[instr.dst]
			reg_grads[instr.dst] = 0
			reg_grads[instr.src1] += g * entry.operand_vals[1]
			reg_grads[instr.src2] += g * entry.operand_vals[0]
		case DDiv:
			g := reg_grads[instr.dst]
			reg_grads[instr.dst] = 0
			reg_grads[instr.src1] += g / entry.operand_vals[1]
			reg_grads[instr.src2] -= g * entry.operand_vals[0] / (entry.operand_vals[1] * entry.operand_vals[1])
		case DReLU:
			g := reg_grads[instr.dst]
			reg_grads[instr.dst] = 0
			if entry.operand_vals[0] > 0 {
				reg_grads[instr.src] += g
			}
		}
	}
}
