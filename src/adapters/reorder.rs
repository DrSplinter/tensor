use crate::{Index, Shape, Tensor};

pub struct Reorder<A, const D: usize> {
    order: [usize; D],
    inverse_order: [usize; D],
    a: A,
}

impl<A, const D: usize> Reorder<A, D>
where
    A: Tensor<D>,
{
    pub(crate) fn new(a: A, order: [usize; D]) -> Self {
        assert!(order.into_iter().max().unwrap() < D);
        let inverse_order = inverse_order(&order);
        Self {
            order,
            a,
            inverse_order,
        }
    }
}

fn inverse_order<const D: usize>(order: &[usize; D]) -> [usize; D] {
    let mut new_order = [0; D];
    // TODO: do we need to check this every time?
    // assert!(order.into_iter().max().unwrap() < D);
    let mut pairs: Vec<_> = order.iter().cloned().enumerate().collect();
    pairs.sort_by_key(|(_, i)| *i);
    new_order
        .iter_mut()
        .zip(pairs)
        .for_each(|(old, (new, _))| *old = new);
    new_order
}

impl<A, const D: usize> Tensor<D> for Reorder<A, D>
where
    A: Tensor<D>,
{
    type Item = A::Item;

    fn get(&self, index: impl Into<Index<D>>) -> Self::Item {
        let Self {
            inverse_order, a, ..
        } = self;
        a.get(index.into().reorder(inverse_order))
    }

    fn shape(&self) -> Shape<D> {
        let Self { order, a, .. } = self;
        a.shape().reorder(order)
    }
}
