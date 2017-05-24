//! Implementation of a nonlinear conjugate gradient (NCG) method
//!
//! Nonlinear conjugate gradient methods are used for numerical minimization
//! of  general functions of multiple variables by using only
//! the function values and gradient.
//!
//! This crate provides a simple implementation in Rust.
//!
//! Currently implemented methods:
//!
//!   - steepest descent (for testing)
//!   - `CG_DESCENT`: based on [Hager & Zhang '06][HZ]
//!
//! # Simple example
//!
//! ```
//! let ncg = ncg_min::NonlinearCG::<f64>::new();
//!
//! let r = ncg.minimize(&1f64, |x, grad| { *grad = 2. * x; x * x }).unwrap();
//!
//! assert!(r.abs() <= 1e-3);
//! ```
//!
//! [HZ]: http://dx.doi.org/10.1145/1132973.1132979

extern crate num;

mod lin;
mod ncg;
pub mod secant2;

pub use lin::{Lin, Rn};
pub use ncg::{NonlinearCG, NonlinearCGError, NonlinearCGIteration, NonlinearCGMethod};
