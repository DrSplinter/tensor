use crate::{Index, Shape, Tensor};

pub struct Tile<A, const D: usize> {
    shape: Shape<D>,
    a: A,
}

impl<A, const D: usize> Tile<A, D>
where
    A: Tensor<D>,
{
    pub(crate) fn new(a: A, shape: Shape<D>) -> Self {
        assert!(shape.divisible(&a.shape()));
        Self { shape, a }
    }
}

impl<A, const D: usize> Tensor<D> for Tile<A, D>
where
    A: Tensor<D>,
{
    type Item = A::Item;

    fn get(&self, index: Index<D>) -> Self::Item {
        let Self { a, .. } = self;
        a.get(index.modulo(a.shape()))
    }

    fn shape(&self) -> Shape<D> {
        self.shape
    }
}
