use crate::{index::Index, shape::Shape, traits::TensorMut, Tensor};

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
}

impl<T: Copy, const D: usize> Tensor<D> for Vector<T, D> {
    type Item = T;

    fn get(&self, index: Index<D>) -> Self::Item {
        let Self { data, supremum } = self;
        data[index.rank(&supremum)]
    }

    fn shape(&self) -> Shape<D> {
        Shape(self.supremum)
    }
}

impl<T: Copy, const D: usize> TensorMut<D> for Vector<T, D> {
    fn set(&mut self, index: Index<D>, value: Self::Item) -> &mut Self {
        let Self { data, supremum } = self;
        data[index.rank(&supremum)] = value;
        self
    }
}
