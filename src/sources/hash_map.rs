use crate::{Index, Shape, Tensor};

pub fn hash_map<T: Default, const D: usize>(
    map: impl IntoIterator<Item = (impl Into<Index<D>>, T)>,
    shape: impl Into<Shape<D>>,
) -> HashMap<T, D> {
    HashMap::new(
        map.into_iter().map(|(i, v)| (i.into(), v)).collect(),
        shape.into(),
    )
}

pub struct HashMap<T, const D: usize> {
    map: std::collections::HashMap<Index<D>, T>,
    shape: Shape<D>,
}

impl<T: Default, const D: usize> HashMap<T, D> {
    pub fn new(map: std::collections::HashMap<Index<D>, T>, shape: Shape<D>) -> Self {
        HashMap { map, shape }
    }
}

impl<T, const D: usize> Tensor<D> for HashMap<T, D>
where
    T: Default + Copy,
{
    type Item = T;

    fn get(&self, index: impl Into<Index<D>>) -> Self::Item {
        let Self { map, shape } = self;
        map.get(&index.into()).map(|a| *a).unwrap_or_default()
    }

    fn shape(&self) -> Shape<D> {
        self.shape
    }
}
