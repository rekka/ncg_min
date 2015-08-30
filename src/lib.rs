
extern crate num;

mod lin;
mod ncg;
pub mod secant2;

pub use lin::{Lin, Rn};
pub use ncg::{NonlinearCG, NonlinearCGError, NonlinearCGIteration, NonlinearCGMethod};
