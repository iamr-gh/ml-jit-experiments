package main

import "core:fmt"
import "core:os"

main :: proc() {
	fmt.println("Hello world from odin")

	#unroll for i in 0 ..= 10 {
		fmt.println(i)
	}
}
