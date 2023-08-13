use crate::{Index, Shape, Tensor};

pub struct Filter<A, const D: usize> {
    a: A,
    d: usize,
    is: Vec<usize>,
}

impl<A, const D: usize> Filter<A, D> {
    pub(crate) fn new<F>(a: A, f: F, d: usize) -> Self
    where
        A: Tensor<D>,
        F: Tensor<1, Item = bool>,
    {
        // TODO: comment on why use asserts or refactor
        assert!(d < D);
        assert_eq!(a.shape().sizes()[d], f.shape().sizes()[0]);

        let is = bools_into_indexes(f);

        Filter { a, d, is }
    }
}

fn bools_into_indexes<B>(bs: B) -> Vec<usize>
where
    B: Tensor<1, Item = bool>,
{
    bs.shape()
        .iter()
        .map(|i| bs.get(i))
        .map(usize::from)
        .scan(0usize, |len, count| {
            *len += count;
            Some(*len)
        })
        .enumerate()
        .fold(vec![], |mut indexes, (idx, len)| {
            if len != indexes.len() {
                indexes.push(idx);
            }
            indexes
        })
}

impl<A, const D: usize> Tensor<D> for Filter<A, D>
where
    A: Tensor<D>,
{
    type Item = A::Item;

    fn get(&self, index: impl Into<Index<D>>) -> Self::Item {
        let Filter { a, d, is } = self;

        let mut indices = index.into().indices();
        indices[*d] = is[indices[*d]];

        a.get(indices)
    }

    fn shape(&self) -> Shape<D> {
        let Filter { a, d, is } = self;

        let mut sizes = a.shape().sizes();

        sizes[*d] = is.len();
        sizes.into()
    }
}
