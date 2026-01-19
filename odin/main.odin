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
	f32, // constant
	int, // variable, index into string arr
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


eval_grad :: proc(node: ^Node, binding: []f32, respect: int) -> ValGrad {
	switch n in node {
	case f32:
		return ValGrad{n, 0}
	case int:
		val := binding[n]
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


eval :: proc(node: ^Node, binding: []f32) -> f32 {
	switch n in node {
	case f32:
		return n
	case int:
		val := binding[n]
		return val
	case Op:
		lv := eval(n.l, binding)
		rv := eval(n.r, binding)
		return eval_op(n.type, lv, rv)
	}
	panic("unreachable")
}

binding_to_arr :: proc(binding: map[string]f32, var_idx: map[string]int) -> []f32 {
	arr: [dynamic]f32 // gc handled by arena alloc
	resize(&arr, len(var_idx))
	for s, idx in var_idx {
		arr[idx] = binding[s]
	}
	return arr[:]
}

parse :: proc(s: string) -> (^Node, map[string]int) {
	found_map := map[string]int{}
	ast, end := parse_helper(s, 0, &found_map)
	return ast, found_map
}

// this pass by pointer shouldn't be needed??
parse_helper :: proc(s: string, start: int, found: ^map[string]int) -> (^Node, int) {
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

		var_s := string(acc[:])
		var_idx, ok := found[var_s]
		if !ok {
			var_idx = len(found) + 1
			found[var_s] = var_idx
		}
		l_node^ = var_idx
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

	r_node, r_end := parse_helper(s, end, found)

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
print_ast :: proc(n: ^Node, var_map: map[string]int) {
	inverse_map := map[int]string{}
	for k, v in var_map {
		inverse_map[v] = k
	}

	print_ast_helper(n, inverse_map)
	fmt.println()
}

print_ast_helper :: proc(node: ^Node, var_map: map[int]string) {
	fmt.print("(")
	switch n in node {
	case f32:
		fmt.printf("%f", n)
	case int:
		val, ok := var_map[n]
		assert(ok)
		fmt.printf("%s", val)
	case Op:
		print_ast_helper(n.l, var_map)
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
		print_ast_helper(n.r, var_map)
	}
	fmt.print(")")
}

main :: proc() {
	// I really just want everything in arena allocator for now
	old_allocator := context.allocator
	context.allocator = context.temp_allocator
	defer context.allocator = old_allocator

	ast, vars := parse("x*3/2+1")
	fmt.printf("vars: %v\n", vars)
	print_ast(ast, vars)

	constant_prop(ast)
	print_ast(ast, vars)

	val := eval(ast, binding_to_arr(map[string]f32{"x" = 2, "y" = 2}, vars))
	fmt.printf("%v\n", val)

	free_all(context.temp_allocator)

	ast2, vars2 := parse("x+y*y")
	print_ast(ast2, vars2)
	vals := map[string]f32 {
		"x" = 2,
		"y" = 6,
	}

	bound := binding_to_arr(vals, vars2)

	dfdx := eval_grad(ast2, bound, vars["x"])
	dfdy := eval_grad(ast2, bound, vars["y"])

	assert(dfdx.val == 38)
	assert(dfdy.val == 38)
	assert(dfdx.grad == 1)
	assert(dfdy.grad == 12)


	free_all(context.temp_allocator)
}
