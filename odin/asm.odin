package main

import "core:fmt"

// want to support just a small number of instructions
Reg :: enum {
	R1 = 0,
	R2,
	R3,
	R4,
	R5,
	Ret,
}

// apparently fadd.S is the instruction I want
// and there are special fp registers

// I do need them all to be f32 for now
FAddI :: struct {
	src: Reg,
	imm: f32,
	dst: Reg,
}

FMov :: struct {
	src: Reg,
	dst: Reg,
}

FMovI :: struct {
	imm: f32,
	dst: Reg,
}

FAdd :: struct {
	src1: Reg,
	src2: Reg,
	dst:  Reg,
}

FSub :: struct {
	src1: Reg,
	src2: Reg,
	dst:  Reg,
}

FMulI :: struct {
	src: Reg,
	imm: f32,
	dst: Reg,
}

FMul :: struct {
	src1: Reg,
	src2: Reg,
	dst:  Reg,
}

FDiv :: struct {
	src1: Reg,
	src2: Reg,
	dst:  Reg,
}

// point into some statically allocated array
// addr can be indices, prefer this over pointers
FLoad :: struct {
	addr: int,
	dst:  Reg,
}

FStore :: struct {
	addr: int,
	src:  Reg,
}


// the small amount of virtual instructions I support, which will get generated into asm
VInstr :: union {
	FAdd,
	FSub,
	FAddI,
	FMul,
	FMulI,
	FDiv,
	FMov,
	FMovI,
	FLoad,
	FStore,
	// pick an allocation and force all pointers into that allocation
	// FLoad, need to figure out a memory system for compilation
	// FStore,
}


// small virtual machine before I get real jit working
simulate :: proc(instrs: []VInstr, mem: []f32) -> f32 {
	state: [Reg]f32

	// explicit PC is much slower this
	for inst in instrs {
		switch i in inst {
		case FAddI:
			state[i.dst] = state[i.src] + i.imm
		case FAdd:
			state[i.dst] = state[i.src1] + state[i.src2]
		case FSub:
			state[i.dst] = state[i.src1] - state[i.src2]
		case FMulI:
			state[i.dst] = state[i.src] * i.imm
		case FMul:
			state[i.dst] = state[i.src1] * state[i.src2]
		case FDiv:
			state[i.dst] = state[i.src1] / state[i.src2]
		case FMov:
			state[i.dst] = state[i.src]
		case FMovI:
			state[i.dst] = i.imm
		case FLoad:
			state[i.dst] = mem[i.addr]
		case FStore:
			mem[i.addr] = state[i.src]
		}
	}

	return state[.Ret]
}

Push :: struct {
	imm: f32,
}

// very simple stack machine, may eventually expand
VInstrStack :: union {
	Push,
	OpType,
}


simulate_stack :: proc(instrs: []VInstrStack) -> f32 {
	stack: [dynamic]f32

	// no pc is faster
	for inst in instrs {
		switch i in inst {
		case Push:
			append(&stack, i.imm)
		case OpType:
			r := pop(&stack)
			l := pop(&stack)
			switch i {
			case .Add:
				append(&stack, l + r)
			case .Sub:
				append(&stack, l - r)
			case .Mul:
				append(&stack, l * r)
			case .Div:
				append(&stack, l / r)
			}
		}
		// pc += 1
	}
	return stack[len(stack) - 1]
}

// could write a translator from earlier IRs to this
VInstrStackTiny :: i64
simulate_stack_tiny :: proc(instrs: []VInstrStackTiny) -> f32 {
	stack: [dynamic]f32
	for inst in instrs {
		if inst & 0b111 == 0 {
			// push
			bits := u32(inst >> 32)
			append(&stack, transmute(f32)bits)
		} else {
			// op
			r := pop(&stack)
			l := pop(&stack)

			// invariant, rest of values are 0s
			switch inst {
			case 0b1:
				// add
				append(&stack, l + r)
			case 0b10:
				// sub
				append(&stack, l - r)
			case 0b11:
				// mul
				append(&stack, l * r)
			case 0b100:
				// div
				append(&stack, l / r)
			}
		}
	}
	return stack[len(stack) - 1]
}


// eventually will have real instructions which convert to binary

// write to a file initially, eventually write the bytes directly
