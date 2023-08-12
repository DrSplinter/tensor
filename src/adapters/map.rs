use crate::Tensor;

pub struct Map<F, A, const D: usize> {
    f: F,
    a: A,
}

impl<F, A, BI, const D: usize> Map<F, A, D>
where
    A: Tensor<D>,
    F: Fn(A::Item) -> BI,
{
    pub(crate) fn new(f: F, a: A) -> Self {
        Self { f, a }
    }
}

impl<F, A, BI, const D: usize> Tensor<D> for Map<F, A, D>
where
    A: Tensor<D>,
    F: Fn(A::Item) -> BI,
{
    type Item = BI;

    fn get(&self, index: crate::Index<D>) -> Self::Item {
        let Self { f, a } = self;
        f(a.get(index))
    }

    fn shape(&self) -> crate::Shape<D> {
        self.a.shape()
    }
}
