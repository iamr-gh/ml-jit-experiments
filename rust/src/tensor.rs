// a tensor is a wrapper around a 1d array
// shape defines the indexing logic
// reshape changes the indexling logic
// operations are type safe or not based upon the shape properties
//

pub struct matrix<const R: usize, const C: usize> {
    data: [f32; R * C],
}

// core idea, mat mul needs to check dimensions of both types
// reshapes need to change the type
// this really should not be so hard to pull the compile time types.
