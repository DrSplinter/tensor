use crate::{index::Index, Tensor};

pub trait TensorMut<const D: usize>: Tensor<D> {
    fn set(&mut self, index: Index<D>, value: Self::Item) -> &mut Self;

    fn assign<T: Tensor<D, Item = Self::Item>>(&mut self, other: &T) -> &mut Self {
        self.shape().iter().for_each(|i| {
            self.set(i, other.get(i));
        });
        self
    }
}
