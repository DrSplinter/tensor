use crate::{Index, Shape, Tensor};
use rayon::prelude::*;

#[derive(Debug, PartialEq)]
pub struct VectorMutRef<'d, T, const D: usize> {
    data: &'d mut [T],
    supremum: Index<D>,
}

impl<'d, T, const D: usize> VectorMutRef<'d, T, D> {
    pub fn new(data: &'d mut [T], shape: Shape<D>) -> Self {
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
            panic!("incompatible shapes")
        }
        self.data
            .into_par_iter()
            .enumerate()
            .for_each(|(rank, old)| f(old, other.get(Index::from_rank(rank, &shape.0))));
        self
    }
}

impl<'d, T: Copy, const D: usize> Tensor<D> for VectorMutRef<'d, T, D> {
    type Item = T;

    fn get(&self, index: impl Into<Index<D>>) -> Self::Item {
        let Self { data, supremum } = self;
        data[index.into().rank(supremum)]
    }

    fn shape(&self) -> Shape<D> {
        Shape(self.supremum)
    }
}
