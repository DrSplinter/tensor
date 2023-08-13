use tensor::{empty, hash_map, shape, vector, Tensor, TensorMut};

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

#[test]
fn reorder_test() {
    let a = vector(0..4, [2, 2]);
    let b = a.reorder([1, 0]);
    let c = vector([0, 2, 1, 3], [2, 2]);

    assert_eq!(b.collect(), c)
}

#[test]
fn tile_test() {
    let a = vector(0..6, [3, 2]);
    let b = a.tile(shape([3, 4]));
    let c = vector((0..6).chain(0..6).collect::<Vec<_>>(), [3, 4]);

    assert_eq!(b.collect(), c)
}

#[test]
fn filter_test() {
    let a = vector(0..12, [2, 6]);
    let f = vector([false, true, false, true, true, false], [6]);
    let b = a.filter(f, 1);
    let c = vector([2, 3, 6, 7, 8, 9], [2, 3]);

    assert_eq!(b.collect(), c);
}
