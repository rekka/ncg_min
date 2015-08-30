
extern crate num;

mod lin;
mod ncg;

pub use lin::{Lin, Rn};
pub use ncg::{Secant2, Secant2Error};
pub use ncg::{NonlinearCG, NonlinearCGError, NonlinearCGIteration, NonlinearCGMethod};
