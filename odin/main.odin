#+feature dynamic-literals
package main

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:unicode"

// might need to abstract a method around eval and values
OpType :: enum {
	Add,
	Sub,
	Mul,
	Div,
}

Op :: struct {
	type: OpType,
	l:    ^Node,
	r:    ^Node,
}

Node :: union {
	Op,
	f32,
	string,
}

ValGrad :: struct {
	val:  f32,
	grad: f32,
}

// eventually abstract out grad one?
eval_op :: proc(op: OpType, l, r: f32) -> f32 {
	switch op {
	case .Add:
		return l + r
	case .Sub:
		return l - r
	case .Mul:
		return l * r
	case .Div:
		return l / r
	}
	return 0
}

// want a series of optimizations to collapse the tree
// all modify in place without freeing
constant_prop :: proc(node: ^Node) -> ^Node {
	if op, ok := node.(Op); ok {
		// recurse first to allow fixing of children
		constant_prop(op.l)
		constant_prop(op.r)

		lv, l_ok := op.l.(f32)
		rv, r_ok := op.r.(f32)

		if l_ok && r_ok {
			node^ = eval_op(op.type, lv, rv)
		}
	}
	return node
}


eval_grad :: proc(node: ^Node, binding: map[string]f32, respect: string) -> ValGrad {
	switch n in node {
	case f32:
		return ValGrad{n, 0}
	case string:
		val, ok := binding[n]
		if !ok {
			panic("variable not found")
		}
		return ValGrad{val, n == respect ? 1 : 0}
	case Op:
		lvg := eval_grad(n.l, binding, respect)
		rvg := eval_grad(n.r, binding, respect)
		switch n.type {
		case .Add:
			return ValGrad{lvg.val + rvg.val, lvg.grad + rvg.grad}
		case .Sub:
			return ValGrad{lvg.val - rvg.val, lvg.grad - rvg.grad}
		case .Mul:
			return ValGrad{lvg.val * rvg.val, lvg.grad * rvg.val + rvg.grad * lvg.val}
		case .Div:
			return ValGrad {
				lvg.val / rvg.val,
				(lvg.grad * rvg.val - rvg.grad * lvg.val) / (rvg.val * rvg.val),
			}
		}
	}
	panic("unreachable")
}


eval :: proc(node: ^Node, binding: map[string]f32) -> f32 {
	switch n in node {
	case f32:
		return n
	case string:
		val, ok := binding[n]
		if !ok {
			panic("variable not found")
		}
		return val
	case Op:
		lv := eval(n.l, binding)
		rv := eval(n.r, binding)
		return eval_op(n.type, lv, rv)
	}
	panic("unreachable")
}

parse :: proc(s: string) -> ^Node {
	ast, end := parse_helper(s, 0)
	return ast
}

parse_helper :: proc(s: string, start: int) -> (^Node, int) {
	// simple recursive descent
	// reads left tok then recurses
	end := start
	is_digit := true

	acc := [dynamic]u8{}
	l_node := new(Node, context.temp_allocator)

	if (unicode.is_alpha(rune(s[end]))) {
		// variable
		for end < len(s) && unicode.is_alpha(rune(s[end])) {
			append(&acc, s[end])
			end += 1
		}

		l_node^ = string(acc[:])
	} else if (unicode.is_digit(rune(s[end]))) {
		// constant
		for end < len(s) && unicode.is_digit(rune(s[end])) {
			append(&acc, s[end])
			end += 1
		}

		val, ok := strconv.parse_f32(string(acc[:]))
		if !ok {
			panic("failed parsing float")
		}

		l_node^ = val
	} else {
		panic("not variable or constant")
	}

	if len(s) <= end {
		return l_node, end
	}

	op := rune(s[end])
	end += 1

	r_node, r_end := parse_helper(s, end)

	o_node := new(Node, context.temp_allocator)
	op_type: OpType
	switch op {
	case '+':
		op_type = .Add
	case '-':
		op_type = .Sub
	case '*':
		op_type = .Mul
	case '/':
		op_type = .Div
	case:
		panic("not a recognized op")
	}

	o_node^ = Op{op_type, l_node, r_node}

	return o_node, r_end
}

// normal %#v won't recurse all pointers
print_ast :: proc(n: ^Node) {
	print_ast_helper(n)
	fmt.println()
}

print_ast_helper :: proc(node: ^Node) {
	fmt.print("(")
	switch n in node {
	case f32:
		fmt.printf("%f", n)
	case string:
		fmt.printf("%s", n)
	case Op:
		print_ast_helper(n.l)
		switch n.type {
		case .Add:
			fmt.printf("+")
		case .Sub:
			fmt.printf("-")
		case .Mul:
			fmt.printf("*")
		case .Div:
			fmt.printf("/")
		}
		print_ast_helper(n.r)
	}
	fmt.print(")")
}

main :: proc() {
	ast := parse("x*3/2+1")
	print_ast(ast)

	constant_prop(ast)
	print_ast(ast)

	val := eval(ast, map[string]f32{"x" = 2, "y" = 2})
	fmt.printf("%v\n", val)

	free_all(context.temp_allocator)

	ast2 := parse("x+y*y")
	print_ast(ast2)
	bound := map[string]f32 {
		"x" = 2,
		"y" = 6,
	}
	dfdx := eval_grad(ast2, bound, "x")
	dfdy := eval_grad(ast2, bound, "y")

	assert(dfdx.val == 38)
	assert(dfdy.val == 38)
	assert(dfdx.grad == 1)
	assert(dfdy.grad == 12)


	free_all(context.temp_allocator)
}
