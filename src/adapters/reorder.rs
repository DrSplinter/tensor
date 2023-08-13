use crate::{Index, Shape, Tensor};

pub struct Reorder<A, const D: usize> {
    order: [usize; D],
    a: A,
}

impl<A, const D: usize> Reorder<A, D>
where
    A: Tensor<D>,
{
    pub(crate) fn new(a: A, order: [usize; D]) -> Self {
        assert!(order.into_iter().max().unwrap() < D);
        Self { order, a }
    }
}

impl<A, const D: usize> Tensor<D> for Reorder<A, D>
where
    A: Tensor<D>,
{
    type Item = A::Item;

    fn get(&self, index: impl Into<Index<D>>) -> Self::Item {
        let Self { order, a } = self;
        a.get(index.into().reorder(order))
    }

    fn shape(&self) -> Shape<D> {
        let Self { order, a } = self;
        a.shape().reorder(order)
    }
}
