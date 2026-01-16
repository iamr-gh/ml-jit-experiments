#+feature dynamic-literals
package main

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:unicode"

NodeType :: enum {
	Constant = 0,
	Variable,
	Add,
	Sub,
	Mul,
	Div,
}

NodeChildren :: struct {
	l: ^Node,
	r: ^Node,
}

// could be made smaller with top level union
Node :: struct {
	type: NodeType,
	data: union {
		NodeChildren,
		f32,
		string,
	},
}

eval :: proc(n: ^Node, binding: map[string]f32) -> f32 {
	switch n.type {
	case .Constant:
		return n.data.(f32)
	case .Variable:
		val, ok := binding[n.data.(string)]
		if !ok {
			panic("variable not found")
		}
		return val
	case .Add, .Sub, .Mul, .Div:
		children := n.data.(NodeChildren)
		#partial switch n.type {
		case .Add:
			return eval(children.l, binding) + eval(children.r, binding)
		case .Sub:
			return eval(children.l, binding) - eval(children.r, binding)
		case .Mul:
			return eval(children.l, binding) * eval(children.r, binding)
		case .Div:
			return eval(children.l, binding) / eval(children.r, binding)
		}
	}
	panic("unreachable")
}

parse :: proc(s: string) -> ^Node {
	ast, end := parse_helper(s, 0)
	return ast
}

parse_helper :: proc(s: string, start: int) -> (^Node, int) {
	fmt.println("Calling parse helper on %s, %v", s, start)
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

		l_node.type = .Variable
		l_node.data = string(acc[:])
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

		l_node.type = .Constant
		l_node.data = val
	} else {
		panic("not variable or constant")
	}

	if len(s) <= end {
		return l_node, end
	}

	op := rune(s[end])
	end += 1

	r_node, r_end := parse_helper(s, end)

	o_node := new(Node, context.allocator)
	switch op {
	case '+':
		o_node.type = .Add
	case '-':
		o_node.type = .Sub
	case '*':
		o_node.type = .Mul
	case '/':
		o_node.type = .Div
	case:
		panic("not a recognized op")
	}

	o_node.data = NodeChildren{l_node, r_node}

	return o_node, r_end
}

// normal %#v won't recurse all pointers
print_ast :: proc(n: ^Node) {
	print_ast_helper(n)
	fmt.println()
}

print_ast_helper :: proc(n: ^Node) {
	fmt.print("(")
	switch n.type {
	case .Constant:
		fmt.printf("%f", n.data.(f32))
	case .Variable:
		fmt.printf("%s", n.data.(string))
	case .Add, .Sub, .Mul, .Div:
		children := n.data.(NodeChildren)
		print_ast_helper(children.l)
		#partial switch n.type {
		case .Add:
			fmt.printf("+")
		case .Sub:
			fmt.printf("-")
		case .Mul:
			fmt.printf("*")
		case .Div:
			fmt.printf("/")
		}
		print_ast_helper(children.r)
	}
	fmt.print(")")
}

main :: proc() {
	ast := parse("1+2+x")
	print_ast(ast)
	val := eval(ast, map[string]f32{"x" = 2, "y" = 2})

	fmt.printf("%v", val)

	// arena allocator used for most things
	free_all(context.temp_allocator)
}
