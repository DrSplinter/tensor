#![allow(unused)]

mod adapters;
mod index;
mod shape;
mod sources;
mod traits;

pub use adapters::{Map, Zip};
pub use index::{index, Index};
pub use shape::Shape;
pub use sources::{empty, hash_map, vector, HashMap, Vector};
pub use traits::{Tensor, TensorMut};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mut_vector_test() {
        let a = vector(0..12, [4, 3]);
        let mut b = empty([4, 3]);

        assert_ne!(a, b);
        b.assign(&a);
        assert_eq!(a, b);
    }

    #[test]
    fn collect_test() {
        let a = vector(0..12, [4, 3]);

        assert_eq!(a.clone(), a.collect())
    }

    #[test]
    fn hash_map_test() {
        let a = hash_map([([1usize, 0], 4), ([0, 1], 3)], [2, 2]);
        let b = vector([0, 4, 3, 0], [2, 2]);

        assert_eq!(a.collect(), b);
    }

    #[test]
    fn map_test() {
        let a = vector(0..12, [4, 3]);
        let b = vector(1..13, [4, 3]);
        assert_eq!(a.map(|a| a + 1).collect(), b);
    }

    #[test]
    fn zip_test() {
        let a = vector(0..4, [2, 2]);
        let b = vector(0..4, [2, 2]);
        let c = vector((0..4).map(|x| (x, x)), [2, 2]);

        assert_eq!(a.zip(b).collect(), c);
    }
}
