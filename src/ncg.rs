//! Implementation of a nonlinear conjugate gradient method.

use num::{Float, Zero, One};
use lin::{Lin};
use secant2::{Secant2, Secant2Error};

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
pub enum NonlinearCGError<V> {
    /// `secant2` method failed to converge; returns current point and search direction.
    LineMethodError(V, V, Secant2Error),
    MaxIterReached(i32),
}

/// Information about a performed iteration of the nonlinear CG method
#[derive(Debug,Clone)]
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

impl NonlinearCG<f32> {
    /// Defaults for `f32` type: values mostly based on [HZ'06]
    pub fn new() -> Self {
        NonlinearCG {
            method: NonlinearCGMethod::HagerZhang(0.01),
            line_method: Default::default(),
            alpha0: 1.,
            psi2: 2.,
            grad_norm_tol: 1e-3,
            max_iter: 100,
        }
    }
}

impl NonlinearCG<f64> {
    /// Defaults for `f64` type: values mostly based on [HZ'06]
    pub fn new() -> Self {
        NonlinearCG {
            method: NonlinearCGMethod::HagerZhang(0.01),
            line_method: Default::default(),
            alpha0: 1.,
            psi2: 2.,
            grad_norm_tol: 1e-3,
            max_iter: 100,
        }
    }
}

impl<S: Float> NonlinearCG<S> {
    /// Mininimize the given nonlinear function over a linear space.
    ///
    /// The function `f` must provide its value as well as its gradient,
    /// returned in the provided `&mut V` (to avoid allocation).
    /// `x0` is used as the initial guess.
    pub fn minimize<Func, V>(&self,
                                       x0: &V,
                                       f: Func) -> Result<V, NonlinearCGError<V>>
        where Func: FnMut(&V, &mut V) -> S,
              V : Lin<S=S> + Clone {
        self.minimize_with_trace(x0, f, |_, _| {})
    }

    /// The same as `minimize`, but allows to pass in a callback function that
    /// is called after every iteration.
    /// It is provided with the new point after the iteration is finished,
    /// and with additional information about the performed iteration.
    pub fn minimize_with_trace<Func, V, Callback>(&self,
                                       x0: &V,
                                       mut f: Func,
                                       mut callback: Callback) -> Result<V, NonlinearCGError<V>>
        where Func: FnMut(&V, &mut V) -> S,
              V : Lin<S=S> + Clone,
              Callback: FnMut(&V, NonlinearCGIteration<V::S>) {
        // allocate storage
        let mut x = x0.clone();
        let mut g_k_1 = x0.clone();
        let mut g_k = x0.clone();
        let mut d_k;
        let mut d_k_1 = x0.origin();
        let mut x_temp = x0.clone();
        let mut grad_temp = x0.clone();
        let mut y = x0.clone();

        let mut alpha = self.alpha0;

        for k in 0..self.max_iter {
            // move from previous iteration (swap by moving)
            g_k = { let t = g_k_1; g_k_1 = g_k; t };
            d_k = d_k_1;
            // compute gradient
            // TODO: use the last evaluation in the line minimization to save this call
            let fx = f(&x, &mut g_k_1);

            // test gradient
            let grad_norm = g_k_1.norm();
            if grad_norm < self.grad_norm_tol {
                return Ok(x);
            }


            // compute new direction
            let beta = if k == 0 {
                V::S::zero()
            } else {
                match self.method {
                    NonlinearCGMethod::SteepestDescent => V::S::zero(),
                    NonlinearCGMethod::HagerZhang(eta) => {
                        // g_{k+1} - g_k
                        y.clone_from(&g_k_1);
                        y.ray_to(&g_k, -V::S::one());
                        let dk_yk = d_k.dot(&y);
                        let two = V::S::one() + V::S::one();
                        let betan_k = (y.dot(&g_k_1)
                                       - two * d_k.dot(&g_k_1) * y.norm_squared() / dk_yk) / dk_yk;
                        let eta_k = -V::S::one() / (d_k.norm() * eta.min(g_k.norm()));
                        betan_k.max(eta_k)
                    },
                }
            };

            // compute new direction
            d_k_1 = { d_k.combine(beta, &g_k_1, -V::S::one()); d_k };
            assert!(d_k_1.dot(&g_k_1) < V::S::zero());

            // minimize along the ray
            let mut line_eval_count = 0;
            let r = {
                let mut f_line = |t| {
                    line_eval_count += 1;
                    x_temp.clone_from(&x);
                    x_temp.ray_to(&d_k_1, t);
                    let v = f(&x_temp, &mut grad_temp);
                    (v, grad_temp.dot(&d_k_1))
                };
                self.line_method.find_wolfe(self.psi2 * alpha, &mut f_line,
                                               Some((fx, g_k_1.dot(&d_k_1))))
            };
            match r {
                Ok(t) => alpha = t,
                Err(e) => return Err(NonlinearCGError::LineMethodError(x, d_k_1, e)),
            }

            // update position
            x.ray_to(&d_k_1, alpha);

            callback(&x, NonlinearCGIteration {
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
