use crate::{Index, Shape, Tensor, TensorMut};
use rayon::prelude::*;

pub fn vector<T, const D: usize>(
    data: impl IntoIterator<Item = T>,
    shape: impl Into<Shape<D>>,
) -> Vector<T, D> {
    Vector::new(data.into_iter().collect(), shape.into())
}

pub fn empty<T: Default, const D: usize>(shape: impl Into<Shape<D>>) -> Vector<T, D> {
    let shape = shape.into();
    let data: Vec<T> = std::iter::repeat_with(|| T::default())
        .take(shape.0.lower_bound_size())
        .collect();
    Vector::new(data, shape)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T, const D: usize> {
    data: Vec<T>,
    supremum: Index<D>,
}

impl<T, const D: usize> Vector<T, D> {
    pub fn new(data: Vec<T>, shape: Shape<D>) -> Self {
        let supremum = shape.0;
        Self { data, supremum }
    }

    pub fn par_update<A, F>(&mut self, f: F, other: &A) -> &mut Self
    where
        A: Tensor<D> + Sync,
        F: Fn(&mut T, A::Item) + Sync,
        T: Copy + Sync + Send,
    {
        let shape = self.shape();
        if shape != other.shape() {
            //TODO: return result?
            panic!("incompatible shapes {:?} != {:?}", shape, other.shape());
        }
        self.data
            .as_mut_slice()
            .into_par_iter()
            .enumerate()
            .for_each(|(rank, old)| f(old, other.get(Index::from_rank(rank, &shape.0))));
        self
    }

    pub fn clear(&mut self) {
        self.data.clear()
    }

    pub fn into_inner(self) -> Vec<T> {
        self.data
    }
}

impl<T: Copy, const D: usize> Tensor<D> for Vector<T, D> {
    type Item = T;

    fn get(&self, index: impl Into<Index<D>>) -> Self::Item {
        let Self { data, supremum } = self;
        data[index.into().rank(supremum)]
    }

    fn shape(&self) -> Shape<D> {
        Shape(self.supremum)
    }
}

impl<T: Copy, const D: usize> TensorMut<D> for Vector<T, D> {
    fn get_mut(&mut self, index: impl Into<Index<D>>) -> &mut Self::Item {
        let Self { data, supremum } = self;
        &mut data[index.into().rank(supremum)]
    }
}
