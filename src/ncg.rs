//! Implementation of a nonlinear conjugate gradient method.

use num_traits::Float;
use secant2::{Secant2, Secant2Error};
use ndarray::prelude::*;
use std::fmt;
use std::error::Error;

/// Implementation of a nonlinear conjugate gradient method.
#[derive(Debug,Clone)]
pub struct NonlinearCG<S: Float> {
    /// Nonlinear CG method
    pub method: NonlinearCGMethod<S>,
    /// Parameters for line minimization `secant2` method
    pub line_method: Secant2<S>,
    /// Initial line minimization bracketing interval: `[0, alpha0]`
    pub alpha0: S,
    /// Multiplier for initial line minimization bracketing interval: `[0, psi2 * alpha]`,
    /// where `alpha` was obtained in previous iteration.
    pub psi2: S,
    /// Desired norm of the gradient
    pub grad_norm_tol: S,
    /// Maximum number of iterations to take
    pub max_iter: i32,
}

#[derive(Debug,Clone)]
pub enum NonlinearCGMethod<S> {
    /// Naive method of steepest descent
    SteepestDescent,
    /// `CG_DESCENT` method from [HZ'06] with `eta` parameter
    HagerZhang(S),
}

#[derive(Debug,Clone)]
pub enum NonlinearCGError<S> {
    /// `secant2` method failed to converge; returns current point and search direction.
    LineMethodError(Vec<S>, Vec<S>, Secant2Error),
    MaxIterReached(i32),
}

/// Information about a performed iteration of the nonlinear CG method
#[derive(Debug, Clone)]
pub struct NonlinearCGIteration<S> {
    /// Iteration number (indexed from 0)
    pub k: i32,
    /// Gradient norm at the beginning of the iteration
    pub grad_norm: S,
    /// Function value at the beginning of the iteration
    pub value: S,
    /// `beta` coefficient for the nonlinear CG search direction update
    pub beta: S,
    /// Line minimization result
    pub alpha: S,
    /// Number of function evaluations by the line minimization method
    pub line_eval_count: i32,
}

impl<S: From<f32> + Float> NonlinearCG<S> {
    /// Defaults: values mostly based on [HZ'06]
    pub fn new() -> Self {
        NonlinearCG {
            method: NonlinearCGMethod::HagerZhang(From::from(0.01f32)),
            line_method: Default::default(),
            alpha0: From::from(1f32),
            psi2: From::from(2f32),
            grad_norm_tol: From::from(1e-3f32),
            max_iter: 100,
        }
    }
}

trait Norm<S> {
    fn norm(&self) -> S;
    fn norm_squared(&self) -> S;
}

impl<S: Float + 'static> Norm<S> for Array1<S> {
    fn norm(&self) -> S {
        self.dot(self).sqrt()
    }

    fn norm_squared(&self) -> S {
        self.dot(self)
    }
}

impl<S: Float + 'static> NonlinearCG<S> {
    /// Mininimize the given nonlinear function over a linear space.
    ///
    /// The function `f` must provide its value as well as its gradient,
    /// returned in the provided `&mut V` (to avoid allocation).
    /// `x0` is used as the initial guess.
    pub fn minimize<Func>(&self, x0: &[S], f: Func) -> Result<Vec<S>, NonlinearCGError<S>>
        where Func: FnMut(&[S], &mut [S]) -> S
    {
        self.minimize_with_trace(x0, f, |_, _| {})
    }

    /// The same as `minimize`, but allows to pass in a callback function that
    /// is called after every iteration.
    /// It is provided with the new point after the iteration is finished,
    /// and with additional information about the performed iteration.
    pub fn minimize_with_trace<Func, Callback>(&self,
                                               x0: &[S],
                                               mut f: Func,
                                               mut callback: Callback)
                                               -> Result<Vec<S>, NonlinearCGError<S>>
        where Func: FnMut(&[S], &mut [S]) -> S,
              Callback: FnMut(&[S], NonlinearCGIteration<S>)
    {
        let x0 = ArrayView::from_shape(x0.len(), x0).unwrap();
        // allocate storage
        let mut x = x0.to_owned();
        let mut g_k_1 = x0.to_owned();
        let mut g_k = x0.to_owned();
        let mut d_k;
        let mut d_k_1 = Array1::from_elem(x0.dim(), S::zero());
        let mut x_temp = x0.to_owned();
        let mut grad_temp = x0.to_owned();
        let mut y = x0.to_owned();

        let mut alpha = self.alpha0;

        for k in 0..self.max_iter {
            // move from previous iteration (swap by moving)
            g_k = {
                let t = g_k_1;
                g_k_1 = g_k;
                t
            };
            d_k = d_k_1;
            // compute gradient
            // TODO: use the last evaluation in the line minimization to save this call
            let fx = f(x.as_slice().unwrap(), g_k_1.as_slice_mut().unwrap());

            // test gradient
            let grad_norm = g_k_1.norm();
            if grad_norm < self.grad_norm_tol {
                return Ok(x.into_raw_vec());
            }


            // compute new direction
            let beta = if k == 0 {
                S::zero()
            } else {
                match self.method {
                    NonlinearCGMethod::SteepestDescent => S::zero(),
                    NonlinearCGMethod::HagerZhang(eta) => {
                        y.clone_from(&g_k_1);
                        y = y - &g_k;
                        let dk_yk = d_k.dot(&y);
                        let two = S::one() + S::one();
                        let betan_k = (y.dot(&g_k_1) -
                                       two * d_k.dot(&g_k_1) * y.norm_squared() / dk_yk) /
                                      dk_yk;
                        let eta_k = -S::one() / (d_k.norm() * eta.min(g_k.norm()));
                        betan_k.max(eta_k)
                    }
                }
            };

            // compute new direction
            d_k_1 = {
                azip!(mut d_k, g_k_1 in { *d_k = *d_k * beta - g_k_1});
                d_k
            };
            assert!(d_k_1.dot(&g_k_1) < S::zero());

            // minimize along the ray
            let mut line_eval_count = 0;
            let r = {
                let mut f_line = |t| {
                    line_eval_count += 1;
                    x_temp.clone_from(&x);
                    azip!(mut x_temp, d_k_1 in { *x_temp = *x_temp + t * d_k_1});
                    let v = f(x_temp.as_slice().unwrap(),
                              grad_temp.as_slice_mut().unwrap());
                    (v, grad_temp.dot(&d_k_1))
                };
                self.line_method
                    .find_wolfe(self.psi2 * alpha,
                                &mut f_line,
                                Some((fx, g_k_1.dot(&d_k_1))))
            };
            match r {
                Ok(t) => alpha = t,
                Err(e) => {
                    return Err(NonlinearCGError::LineMethodError(x.into_raw_vec(),
                                                                 d_k_1.into_raw_vec(),
                                                                 e))
                }
            }

            // update position
            azip!(mut x, d_k_1 in { *x = *x + alpha * d_k_1});

            callback(x.as_slice().unwrap(),
                     NonlinearCGIteration {
                         k: k,
                         beta: beta,
                         grad_norm: grad_norm,
                         value: fx,
                         alpha: alpha,
                         line_eval_count: line_eval_count,
                     });
        }

        return Err(NonlinearCGError::MaxIterReached(self.max_iter));
    }
}

impl<S: fmt::Display> fmt::Display for NonlinearCGError<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &NonlinearCGError::LineMethodError(_, _, ref e) => {
                write!(f, "Line minimization failed due to: {}", e)
            }
            &NonlinearCGError::MaxIterReached(n) => {
                write!(f, "Maximum number of iterations reached: {}", n)
            }
        }
    }
}

impl<S: fmt::Display + fmt::Debug> Error for NonlinearCGError<S> {
    fn description(&self) -> &str {
        match self {
            &NonlinearCGError::LineMethodError(_, _, ref e) => e.description(),
            &NonlinearCGError::MaxIterReached(_) => "Maximum number of iterations reached",
        }
    }

    fn cause(&self) -> Option<&Error> {
        match self {
            &NonlinearCGError::LineMethodError(_, _, ref e) => Some(e),
            &NonlinearCGError::MaxIterReached(_) => None,
        }
    }
}
