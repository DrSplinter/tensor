use crate::{index::Index, Tensor};

pub trait TensorMut<const D: usize>: Tensor<D> {
    fn get_mut(&mut self, index: Index<D>) -> &mut Self::Item;

    fn set(&mut self, index: Index<D>, value: Self::Item) {
        *self.get_mut(index) = value;
    }

    fn update<T, F>(&mut self, f: F, other: &T) -> &mut Self
    where
        T: Tensor<D>,
        F: Fn(&mut Self::Item, T::Item),
    {
        self.shape().iter().for_each(|i| {
            f(self.get_mut(i), other.get(i));
        });
        self
    }

    fn assign<T: Tensor<D, Item = Self::Item>>(&mut self, other: &T) -> &mut Self {
        self.update(|old, new| *old = new, other);
        self
    }
}
