use crate::{Index, Shape, Tensor};

pub struct Zip<A, B, const D: usize> {
    a: A,
    b: B,
}

impl<A, B, const D: usize> Zip<A, B, D>
where
    A: Tensor<D>,
    B: Tensor<D>,
{
    pub(crate) fn new(a: A, b: B) -> Self {
        assert_eq!(a.shape(), b.shape());
        Self { a, b }
    }
}

impl<A, B, const D: usize> Tensor<D> for Zip<A, B, D>
where
    A: Tensor<D>,
    B: Tensor<D>,
{
    type Item = (A::Item, B::Item);

    fn get(&self, index: impl Into<Index<D>>) -> Self::Item {
        let index = index.into();
        (self.a.get(index), self.b.get(index))
    }

    fn shape(&self) -> Shape<D> {
        // We asserted in ctor that its the same as b's shape
        self.a.shape()
    }
}
