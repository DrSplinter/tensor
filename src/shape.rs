use std::ops::Deref;

use crate::index::Index;

pub fn shape<const D: usize>(shape: impl Into<Shape<D>>) -> Shape<D> {
    shape.into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape<const D: usize>(pub(crate) Index<D>);

impl<const D: usize> From<[usize; D]> for Shape<D> {
    fn from(value: [usize; D]) -> Self {
        Shape(value.into())
    }
}

impl<const D: usize> From<Index<D>> for Shape<D> {
    fn from(value: Index<D>) -> Self {
        Shape(value)
    }
}

impl<const D: usize> Shape<D> {
    pub fn sizes(&self) -> [usize; D] {
        self.0.indices()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = Index<D>> + 'a {
        (0..self.0.lower_bound_size()).map(|rank| Index::from_rank(rank, &self.0))
    }

    pub fn reorder(&self, order: &[usize; D]) -> Shape<D> {
        Shape(self.0.reorder(order))
    }

    pub fn divisible(&self, other: &Shape<D>) -> bool {
        self.sizes()
            .into_iter()
            .zip(other.sizes())
            .all(|(a, b)| a % b == 0)
    }
}
