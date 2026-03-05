package main

// what are the interfaces I want
// ability to declare each layer separately
// which is just an input vector, and output vector
// layer is then an y = relu(Ax + b)
// let's keep bias vector separate for now, in future it may be fused

// oh shape propogation sucks and doesn't exist
// I have no idea what's on main...
// nn_layer :: proc(in: MatNode, out_dim: int, start_idx: int) -> (MatNode, int) {
//     // assert(in.)
//
// 	weights, weight_end := makeVarMat(in_dim, out_dim, start_idx)
// 	bias, bias_end := makeVarMat(1, out_dim, weight_end)
//
// }

// indexing system
// each changeable variable is kept track of using an id system
// it's important that ids 1) don't overlap, and 2)
