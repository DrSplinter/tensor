use crate::{empty, Index, Map, Reorder, Shape, TensorMut, Tile, Vector, Zip};

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

    fn reorder(self, order: [usize; D]) -> Reorder<Self, D>
    where
        Self: Sized,
    {
        Reorder::new(self, order)
    }

    fn tile(self, shape: impl Into<Shape<D>>) -> Tile<Self, D>
    where
        Self: Sized,
    {
        Tile::new(self, shape.into())
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
