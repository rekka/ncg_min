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
//! let r = ncg.minimize(&[1f64], |x, grad| { grad[0] = 2. * x[0]; x[0] * x[0] }).unwrap();
//!
//! assert!(r[0].abs() <= 1e-3);
//! ```
//!
//! [HZ]: http://dx.doi.org/10.1145/1132973.1132979

extern crate num_traits;
#[macro_use]
extern crate ndarray;

mod ncg;
pub mod secant2;

pub use ncg::{NonlinearCG, NonlinearCGError, NonlinearCGIteration, NonlinearCGMethod};
