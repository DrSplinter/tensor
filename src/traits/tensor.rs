use crate::{empty, Index, Map, Shape, TensorMut, Vector, Zip};

pub trait Tensor<const D: usize> {
    type Item;

    fn get(&self, index: Index<D>) -> Self::Item;
    fn shape(&self) -> Shape<D>;

    fn map<F, MI>(self, f: F) -> Map<F, Self, D>
    where
        F: Fn(Self::Item) -> MI,
        Self: Sized,
    {
        Map::new(f, self)
    }

    fn zip<B>(self, other: B) -> Zip<Self, B, D>
    where
        B: Tensor<D>,
        Self: Sized,
    {
        Zip::new(self, other)
    }

    fn collect(self) -> Vector<Self::Item, D>
    where
        Self::Item: Copy + Default,
        Self: Sized,
    {
        let mut e = empty(self.shape());
        e.assign(&self);
        e
    }
}
