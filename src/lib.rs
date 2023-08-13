#![allow(unused)]

mod adapters;
mod index;
mod shape;
mod sources;
mod traits;

pub use adapters::{Filter, Map, Reorder, Tile, Zip};
pub use index::{index, Index};
pub use shape::{shape, Shape};
pub use sources::{empty, hash_map, vector, HashMap, Vector};
pub use traits::{Tensor, TensorMut};
