#+feature dynamic-literals
package main

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:unicode"

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


ValGrad :: struct {
	val:  f32,
	grad: f32,
}

// should create a respect to all situation
eval_grad_forward :: proc(node: ^Node, binding: []f32, respect: int) -> ValGrad {
	switch n in node {
	case f32:
		return ValGrad{n, 0}
	case int:
		val := binding[n]
		return ValGrad{val, n == respect ? 1 : 0}
	case Op:
		lvg := eval_grad_forward(n.l, binding, respect)
		rvg := eval_grad_forward(n.r, binding, respect)
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

// propogate grads of all variables
// yes, this will scale badly at high param counts
// might be worth testing at what point that is
eval_grad_forward_all :: proc(node: ^Node, binding: []f32) -> (f32, []f32) {
	// need to figure out a more efficient alloc structure
	grads: [dynamic]f32
	resize(&grads, len(binding)) // zero initialized

	switch n in node {
	case f32:
		return n, grads[:]
	case int:
		val := binding[n]
		grads[n] = 1
		return val, grads[:]
	case Op:
		lv, lgrad := eval_grad_forward_all(n.l, binding)
		rv, rgrad := eval_grad_forward_all(n.r, binding)
		switch n.type {
		case .Add:
			for i in 0 ..< len(binding) {
				grads[i] = lgrad[i] + rgrad[i]
			}

			return lv + rv, grads[:]
		case .Sub:
			for i in 0 ..< len(binding) {
				grads[i] = lgrad[i] - rgrad[i]
			}

			return lv - rv, grads[:]
		case .Mul:
			for i in 0 ..< len(binding) {
				grads[i] = lgrad[i] * rv + rgrad[i] * lv
			}

			return lv * rv, grads[:]
		case .Div:
			for i in 0 ..< len(binding) {
				grads[i] = (lgrad[i] * rv - rgrad[i] * lv) / (rv * rv)
			}
			return lv / rv, grads[:]
		}
	}
	panic("unreachable")
}


// computes with respect to all
// out_grads must be pre-allocated with length >= len(binding)
// activation_cache and grad_cache can be provided to avoid repeated allocations in tight loops
eval_grad_reverse :: proc(
	node: ^Node,
	binding: []f32,
	out_grads: []f32,
	activation_cache: ^map[^Node]f32 = nil,
	grad_cache: ^map[^Node]f32 = nil,
) -> f32 {
	for i in 0 ..< len(out_grads) {
		out_grads[i] = 0
	}

	local_activation_cache: map[^Node]f32
	local_grad_cache: map[^Node]f32

	act_cache := activation_cache if activation_cache != nil else &local_activation_cache
	grd_cache := grad_cache if grad_cache != nil else &local_grad_cache

	clear(act_cache)
	clear(grd_cache)

	val := eval_grad_reverse_helper_forward(node, binding, act_cache)

	grd_cache[node] = 1
	eval_grad_reverse_helper_backward(node, out_grads, act_cache, grd_cache)

	if activation_cache == nil {
		delete(local_activation_cache)
	}
	if grad_cache == nil {
		delete(local_grad_cache)
	}

	return val
}

eval_grad_reverse_helper_forward :: proc(
	node: ^Node,
	binding: []f32,
	activation_cache: ^map[^Node]f32,
) -> f32 {
	val: f32
	switch n in node {
	case f32:
		val = n
	case int:
		val = binding[n]
	case Op:
		lv := eval_grad_reverse_helper_forward(n.l, binding, activation_cache)
		rv := eval_grad_reverse_helper_forward(n.r, binding, activation_cache)
		val = eval_op(n.type, lv, rv)
	}
	activation_cache[node] = val
	return val
}

eval_grad_reverse_helper_backward :: proc(
	node: ^Node,
	binding_grads: []f32,
	activation_cache: ^map[^Node]f32,
	grad_cache: ^map[^Node]f32,
) {
	switch n in node {
	case f32:
		return
	case int:
		// gradients accumulate in vars
		// because they can show up in different places
		binding_grads[n] += grad_cache[node]
	case Op:
		switch n.type {
		case .Add, .Sub:
			grad_cache[n.l] = grad_cache[node]
			grad_cache[n.r] = grad_cache[node]
		case .Mul:
			grad_cache[n.l] = grad_cache[node] * activation_cache[n.r]
			grad_cache[n.r] = grad_cache[node] * activation_cache[n.l]
		case .Div:
		// you just need to split and rearrange for each side
		// totally possible, just need to write it out

		// annoying?
		// grad_cache[n.l] =
		// return ValGrad {
		// 	lvg.val / rvg.val,
		// 	(lvg.grad * rvg.val - rvg.grad * lvg.val) / (rvg.val * rvg.val),
		// }
		}
		eval_grad_reverse_helper_backward(n.l, binding_grads, activation_cache, grad_cache)
		eval_grad_reverse_helper_backward(n.r, binding_grads, activation_cache, grad_cache)
	}
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
	resize(&arr, len(var_idx) + 1)
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
