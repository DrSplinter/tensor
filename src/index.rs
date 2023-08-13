use std::ops::Deref;

#[cfg(test)]
use proptest_derive::Arbitrary;

use crate::Shape;

pub fn index<const D: usize>(index: impl Into<Index<D>>) -> Index<D> {
    index.into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, Hash)]
#[cfg_attr(test, derive(Arbitrary))]
pub struct Index<const D: usize>([usize; D]);

impl<const D: usize> From<[usize; D]> for Index<D> {
    fn from(value: [usize; D]) -> Self {
        Self(value)
    }
}

impl From<usize> for Index<1> {
    fn from(value: usize) -> Self {
        Self([value])
    }
}

impl<const D: usize> From<[u8; D]> for Index<D> {
    fn from(value: [u8; D]) -> Self {
        let mut arr = [0usize; D];

        arr.iter_mut().zip(value).for_each(|(a, b)| {
            *a = b.into();
        });

        arr.into()
    }
}

impl<const D: usize> PartialOrd for Index<D> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;
        for cmp in self
            .0
            .into_iter()
            .zip(other.0)
            .rev()
            .map(|(a, b)| a.cmp(&b))
        {
            match cmp {
                Less => return Some(Less),
                Equal => (),
                Greater => return Some(Greater),
            }
        }
        Some(Equal)
    }
}

// TODO: Here is a lot of pub(crate) methods. We need to make this cleaner.

impl<const D: usize> Index<D> {
    pub(crate) fn rank(&self, maximum: &Self) -> usize {
        let Self(a) = self;
        let Self(b) = maximum;

        let b = b.iter().scan(1usize, |a, b| {
            let x = *a;
            *a *= b;
            Some(x)
        });
        a.iter().zip(b).map(|(a, b)| *a * b).sum()
    }

    pub(crate) fn lower_bound_size(&self) -> usize {
        let Self(a) = self;
        let x = *a;
        let mut s = [0usize; D];
        s[D - 1] = x[D - 1];
        index(s).rank(self)
    }

    pub(crate) fn from_rank(rank: usize, maximum: &Self) -> Self {
        let Self(m) = maximum;
        let mut out = [0usize; D];

        let sizes = m
            .iter()
            .scan(1usize, |a, b| {
                let x = *a;
                *a *= b;
                Some(x)
            })
            .collect::<Vec<_>>();
        sizes
            .into_iter()
            .enumerate()
            .rev()
            .fold((out, rank), |(mut idx, rest), (pos, size)| {
                idx[pos] = rest / size;
                (idx, rest - idx[pos] * size)
            })
            .0
            .into()
    }

    pub(crate) fn bounded_by(&self, bound: &Self) -> bool {
        self.0.into_iter().zip(bound.0).all(|(a, b)| a < b)
    }

    pub(crate) fn indices(&self) -> [usize; D] {
        self.0
    }

    pub(crate) fn reorder(&self, order: &[usize; D]) -> Index<D> {
        let old_sizes = self.indices();
        let mut new_sizes = [0; D];
        // TODO: do we need to check this every time?
        // assert!(order.into_iter().max().unwrap() < D);

        order
            .iter()
            .enumerate()
            .for_each(|(old, new)| new_sizes[*new] = old_sizes[old]);
        new_sizes.into()
    }

    pub(crate) fn modulo(&self, shape: Shape<D>) -> Index<D> {
        let value = self.indices();
        let modulus = shape.sizes();
        let mut modulo = [0; D];

        modulo
            .iter_mut()
            .zip(value)
            .zip(modulus)
            .for_each(|(((r, v), m))| *r = v % m);
        modulo.into()
    }
}

impl Index<1> {
    pub(crate) fn lower_bound<'a>(&'a self) -> impl Iterator<Item = Self> + 'a {
        let Self([a]) = self;
        (0..*a).map(Self::from)
    }
}

impl Index<2> {
    pub(crate) fn lower_bound<'a>(&'a self) -> impl Iterator<Item = Self> + 'a {
        let Self([a, b]) = self;
        (0..*b).flat_map(move |b| (0..*a).map(move |a| Index([a, b])))
    }
}

impl Index<3> {
    pub(crate) fn lower_bound<'a>(&'a self) -> impl Iterator<Item = Self> + 'a {
        let Self([a, b, c]) = self;
        (0..*c).flat_map(move |c| (0..*b).flat_map(move |b| (0..*a).map(move |a| Index([a, b, c]))))
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use crate::shape::{self, shape};

    use super::*;
    use proptest::{prelude::*, proptest};

    fn gen_small_index<const D: usize>() -> impl Strategy<Value = Index<D>> {
        any::<[u8; D]>().prop_map(Index::<D>::from)
    }

    fn gen_bounded_indexes<const D: usize>(
    ) -> impl Strategy<Value = ((Index<D>, Index<D>), Index<D>)> {
        any::<(Index<D>, Index<D>, Index<D>)>().prop_map(|(a, b, bound)| ((a, b), bound))
    }

    proptest! {
        #[test]
        fn rank_exists(a: Index<1>, max: Index<1>) {
            prop_assume!(a < max);
            prop_assert!(a.rank(&max) < max.rank(&max));
        }

        #[test]
        fn rank_follows_ordering_1(a: Index<1>, b: Index<1>, max: Index<1>) {
          prop_assert_eq!(a.cmp(&b), a.rank(&max).cmp(&b.rank(&max)))
        }

        #[ignore = "solve multiply overflow in multiplication"]
        #[test]
        fn rank_follows_ordering_2(a: Index<2>, b: Index<2>, max: Index<2>) {
            prop_assume!(a < max && b < max);
            prop_assert_eq!(a.cmp(&b), a.rank(&max).cmp(&b.rank(&max)))
        }

        #[test]
        fn lower_bound_contains_lesser_1(a in gen_small_index::<1>()) {
            prop_assert!(a.lower_bound().all(|b| b < a))
        }

        #[test]
        fn prop_lower_bound_contains_lesser_2(a in gen_small_index::<2>()) {
            prop_assert!(a.lower_bound().all(|b| b < a))
        }

        #[test]
        fn prop_lower_bound_size_is_correct_2(a in gen_small_index::<2>()) {
            // TODO: make method index_values?
            prop_assume!(a.0.into_iter().all(|i| i != 0));
            prop_assert_eq!(a.lower_bound().count(), a.lower_bound_size())
        }

        #[test]
        fn prop_lower_bound_is_sorted_1(a in gen_small_index::<1>()) {
            prop_assume!(a.0.into_iter().all(|i| i != 0));
            let bound:Vec<_> = a.lower_bound().collect();
            prop_assert!(bound.windows(2).all(|w| w[0] <= w[1]));
        }

        #[test]
        fn prop_lower_bound_is_sorted_2(a in gen_small_index::<2>()) {
            prop_assume!(a.0.into_iter().all(|i| i != 0));
            let bound:Vec<_> = a.lower_bound().collect();
            prop_assert!(bound.windows(2).all(|w| w[0] <= w[1]));
        }

        #[test]
        fn prop_from_rank_is_inverse_to_rank(a in gen_small_index::<2>(), max in gen_small_index::<2>()) {
            prop_assume!(a.bounded_by(&max) && max.0.into_iter().all(|i| i != 0));
            prop_assert_eq!(Index::from_rank(a.rank(&max), &max), a);
        }
    }

    #[test]
    fn from_rank_is_inverse_to_rank() {
        let a = index([0usize, 1]);
        let max = index([1usize, 2]);

        assert_eq!(Index::from_rank(a.rank(&max), &max), a)
    }

    #[test]
    fn lower_bound_is_sorted_2_err_2() {
        let a = index([1usize, 3]);
        let bound: Vec<_> = a.lower_bound().collect();

        assert!(index([0usize, 1]) < index([0usize, 2]));
        assert!(bound.windows(2).all(|w| w[0] <= w[1]))
    }

    #[test]
    fn lower_bound_is_sorted_2_err_1() {
        let a = index([2usize, 2]);
        let bound: Vec<_> = a.lower_bound().collect();
        assert!(bound.windows(2).all(|w| w[0] <= w[1]))
    }

    #[test]
    fn lower_bound_size_is_correct_2() {
        let a = index([1usize, 2]);

        assert_eq!(a.lower_bound().count(), a.lower_bound_size())
    }

    #[test]
    fn rank_of_itself_is_max_limit_for_lower_bound_ranks_3() {
        let a = index([2usize, 1, 1]);
        let b = index([3usize, 4, 2]);
        let e = index([0, 0, 2usize]);

        assert!(a.rank(&b) < b.rank(&b));
    }

    #[test]
    fn rank_of_itself_is_max_limit_for_lower_bound_ranks_2() {
        let a = index([2usize, 1]);
        let b = index([3usize, 4]);
        let e = index([0, 4usize]);

        assert!(a.rank(&b) < b.rank(&b));
    }

    #[test]
    fn rank_is_ok() {
        assert_eq!(index([2usize, 2]).rank(&index([3usize, 4])), 8);
    }

    #[test]
    fn ordering_is_ok() {
        assert!(index([0usize, 9]) < index([1usize, 9]));
        assert!(index([2usize, 1]) == index([2usize, 1]));
        assert!(index([3usize, 8]) < index([3usize, 9]));
    }

    #[test]
    fn lower_bound_works() {
        assert_eq!(
            index([1usize, 1]).lower_bound().collect::<Vec<_>>(),
            vec![index([0usize, 0])]
        );

        assert_eq!(
            index([2usize, 3]).lower_bound().collect::<Vec<_>>(),
            vec![
                index([0usize, 0]),
                index([1usize, 0]),
                index([0usize, 1]),
                index([1usize, 1]),
                index([0usize, 2]),
                index([1usize, 2])
            ]
        );
    }

    #[test]
    fn modulo_works() {
        let whole = shape([3usize, 4]);
        let modulus = shape([3usize, 2]);
        let idx = index([2usize, 2]);

        assert_eq!(idx.modulo(modulus), index([2usize, 0]));
    }
}
