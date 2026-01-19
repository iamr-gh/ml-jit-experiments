package main

import "core:fmt"

// want to support just a small number of instructions
Reg :: enum {}

// apparently fadd.S is the instruction I want
// and there are special fp registers

// I do need them all to be f32 for now
FAddI :: struct {
	src: Reg,
	imm: f32,
	dst: Reg,
}

FAdd :: struct {
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


// the small amount of virtual instructions I support, which will get generated into asm
VInstr :: union {
	FAdd,
	FAddI,
	FMul,
	FMulI,
}

// eventually will have real instructions which convert to binary

// write to a file initially, eventually write the bytes directly


// will also need  a validation machine
