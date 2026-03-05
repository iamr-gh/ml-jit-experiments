pub fn mat_t(comptime r: i32, comptime c: i32) type {
    return struct {
        data: [r * c]f32,
        comptime rows: i32 = r,
        comptime cols: i32 = c,
    };
}

// pub fn mat_add(mat1: i32, mat2: i32) i32 {
//     // needs to at compile time compare two types, and then generate interactions between them
//     // and yes, this can and probably should just be a full language at this point
//     return 0;
// }
