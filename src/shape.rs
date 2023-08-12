use crate::index::Index;

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
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = Index<D>> + 'a {
        (0..self.0.lower_bound_size()).map(|rank| Index::from_rank(rank, &self.0))
    }
}
