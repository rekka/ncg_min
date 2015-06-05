//! Implementation of a nonlinear conjugate gradient method.

use num::{Float, Zero, One};
use lin::{Lin};

/// Implementation of `secant2` method by _Hager & Zhang'06_.
#[derive(Debug,Clone)]
pub struct Secant2<F: Float> {
    /// `delta` for Wolfe condition
    pub delta: F,
    /// `sigma` for Wolfe condition
    pub sigma: F,
    /// `epsilon` for approximate Wolfe condition (to allow for value function to increase
    /// because of rounding of errors when close to the minimum)
    pub epsilon: F,
    /// Bisection coefficient when secant fails; allowed values in `(0, 1)`
    /// (`0.5` is the midpoint of the interval)
    pub theta: F,
    /// Extension factor for finding the initial bracketing interval; `> 1`
    pub rho: F,
    /// Maximum number of iterations
    pub max_iter: i32,
    /// Maximum number of U3a--b bracketing iterations
    pub ubracket_max_iter: i32,
    /// Maximum number of initial bracketing iterations
    pub init_bracket_max_iter: i32,
}

#[derive(Debug,Clone)]
pub enum Secant2Error {
    MaxIterReached(i32),
    InitBracketMaxIterReached(i32),
    UBracketMaxIterReached(i32),
}

trait Secant2Default<F: Float> {
    fn secant2_default() -> Secant2<F>;
}

impl Secant2Default<f32> for f32 {
    // Defaults for `secant2` method given in [HZ'06]
    fn secant2_default() -> Secant2<f32> {
        Secant2 {
            delta: 0.1,
            sigma: 0.9,
            epsilon: 1e-6,
            theta: 0.5,
            rho: 5.,
            max_iter: 32,
            ubracket_max_iter: 32,
            init_bracket_max_iter: 16,
        }
    }
}

impl Secant2Default<f64> for f64 {
    // Defaults for `secant2` method given in [HZ'06]
    fn secant2_default() -> Secant2<f64> {
        Secant2 {
            delta: 0.1,
            sigma: 0.9,
            epsilon: 1e-6,
            theta: 0.5,
            rho: 5.,
            max_iter: 32,
            ubracket_max_iter: 32,
            init_bracket_max_iter: 16,
        }
    }
}

impl<F: Float + Secant2Default<F>> Secant2<F> {
    pub fn new() -> Self {
        F::secant2_default()
    }
}

enum BracketResult<F> {
    Ok((F, F, F), (F, F, F)),
    MaxIterReached(i32),
    // Wolfe(F), // TODO: implement early escape if solution is found during bracketing
}

impl<F: Float> Secant2<F> {
    /// Find an approximate minimum satisfying the Wolfe condition of a given function.
    ///
    /// The search interval is `(0, c)`.
    /// `hint` may contain the value `(f(0), f'(0))` to avoid unnecessary evalution if
    /// this value is already known (as is the case of the `NonlinearCG` minimization method).
    pub fn find_wolfe<Func>(&self, c: F,
                            f: &mut Func,
                            hint: Option<(F, F)>) -> Result<F, Secant2Error>
        where Func: FnMut(F) -> (F, F) {
        assert!(c > F::zero());

        // save origin
        let o = match hint {
            Some((v, d)) => (F::zero(), v, d),
            None => triple(f, F::zero()),
        };

        // ϕ(0) + ε_k
        let fi = o.1 + self.epsilon;
        // bracket starting interval
        let mut ab = match self.bracket(o, triple(f, c), f) {
            BracketResult::Ok(a, b) => (a, b),
            // BracketResult::Wolfe(x) => return Ok(x),
            BracketResult::MaxIterReached(n) =>
                return Err(Secant2Error::InitBracketMaxIterReached(n)),
        };

        // implementation of secant2 method from [HZ'06]
        for _ in 0..self.max_iter {
            let (a, b) = ab;
            let mut cx = secant(a, b);
            // here we handle the case when b gets stuck at a local minimum of ϕ with
            // ϕ(b) > ϕ(0) + ε
            let ctheta = a.0 + self.theta * (b.0 - a.0);
            if cx > ctheta && b.1 + (cx - b.0) * b.2 > fi {
                // Taylor series estimate tells us that phi(b) does not decrease enough
                // => let's help it
                cx = ctheta;
            }

            let c = triple(f, cx);
            if self.wolfe(c, fi, o.2) { return Ok(c.0); }

            let cx;
            if c.2 >= F::zero() {
                cx = secant(b, c);
                ab.1 = c;
            } else if c.1 <= fi {
                cx = secant(a, c);
                ab.0 = c;
            } else {
                ab = match self.ubracket(a, c, f, fi) {
                    BracketResult::Ok(a, b) => (a, b),
                    // BracketResult::Wolfe(x) => return Ok(x),
                    BracketResult::MaxIterReached(n) =>
                        return Err(Secant2Error::UBracketMaxIterReached(n)),
                };

                continue;
            }
            let (a, b) = ab;
            if a.0 < cx && cx < b.0 {
                let c = triple(f, cx);
                if self.wolfe(c, fi, o.2) { return Ok(c.0); }

                if c.2 >= F::zero() {
                    ab.1 = c;
                } else if c.1 <= fi {
                    ab.0 = c;
                } else {
                    ab = match self.ubracket(a, c, f, fi) {
                        BracketResult::Ok(a, b) => (a, b),
                        // BracketResult::Wolfe(x) => return Ok(x),
                        BracketResult::MaxIterReached(n) =>
                            return Err(Secant2Error::UBracketMaxIterReached(n)),
                    };
                }
            }

        }
        Err(Secant2Error::MaxIterReached(self.max_iter))
    }

    // Implementation of U3a--b bracketing loop in [HZ'06]
    //
    // The format of the triples is `(x, f(x), f'(x))`.
    //
    // - `fi` is `\phi(0) + \epsilon_k`
    fn ubracket<Func>(&self,
                      mut a: (F, F, F),
                      mut b: (F, F, F),
                      f: &mut Func, fi: F) -> BracketResult<F>
        where Func: FnMut(F) -> (F, F) {
        // preconditions
        assert!(a.0 < b.0);
        assert!(a.2 < F::zero());
        assert!(b.2 < F::zero());
        assert!(a.1 <= fi && fi < b.1);

        for _ in 0..self.ubracket_max_iter {
            let cx = a.0 + self.theta * (b.0 - a.0);
            let c = triple(f, cx);

            if c.2 >= F::zero() {
                return BracketResult::Ok(a, c);
            } else {
                if c.1 <= fi {
                    a = c;
                } else {
                    b = c;
                }
            }
        }

        BracketResult::MaxIterReached(self.ubracket_max_iter)
    }

    //Initial bracketing: `bracket(c)` method in [HZ'06]
    fn bracket<Func>(&self,
                     mut a: (F, F, F),
                     mut b: (F, F, F),
                     f: &mut Func) -> BracketResult<F>
        where Func: FnMut(F) -> (F, F) {
        // preconditions
        assert!(a.0 < b.0);
        assert!(a.2 < F::zero());

        let o = a;
        let fi = o.1 + self.epsilon;

        for _ in 0..self.init_bracket_max_iter {
            if b.2 >= F::zero() {
                return BracketResult::Ok(a, b);
            } else if b.1 > fi {
                return self.ubracket(o, b, f, fi);
            } else {
                a = b;
                let bx = self.rho * b.0;
                b = triple(f, bx);
            }
        }

        BracketResult::MaxIterReached(self.init_bracket_max_iter)
    }

    fn wolfe(&self, c: (F, F, F), fi: F, fd0: F) -> bool {
        // approximate Wolfe condition
        c.1 <= fi && self.sigma * fd0 <= c.2 && c.2 <= (self.delta + self.delta - F::one()) * fd0
    }

}

fn triple<Func, F: Float>(f: &mut Func, x: F) -> (F, F, F)
    where Func: FnMut(F) -> (F, F) {
    let (fx, fdx) = f(x);
    (x, fx, fdx)
}

fn secant<F: Float>(a: (F, F, F), b: (F, F, F)) -> F {
    (a.0 * b.2 - b.0 * a.2) / (b.2 - a.2)
}

/// Implementation of a nonlinear conjugate gradient method.
#[derive(Debug,Clone)]
pub struct NonlinearCG<F: Float> {
    /// Nonlinear CG method
    pub method: NonlinearCGMethod<F>,
    /// Parameters for line minimization `secant2` method
    pub line_method: Secant2<F>,
    /// Initial line minimization bracketing interval: `[0, alpha0]`
    pub alpha0: F,
    /// Multiplier for initial line minimization bracketing interval: `[0, psi2 * alpha]`,
    /// where `alpha` was obtained in previous iteration.
    pub psi2: F,
    /// Desired norm of the gradient
    pub grad_norm_tol: F,
    /// Maximum number of iterations to take
    pub max_iter: i32,
}

#[derive(Debug,Clone)]
pub enum NonlinearCGMethod<F> {
    /// Naive method of steepest descent
    SteepestDescent,
    /// `CG_DESCENT` method from [HZ'06] with `eta` parameter
    HagerZhang(F),
}

#[derive(Debug,Clone)]
pub enum NonlinearCGError {
    LineMethodError(Secant2Error),
    MaxIterReached(i32),
}

/// Information concerning each iteration of the nonlinear CG method
#[derive(Debug,Clone)]
pub struct NonlinearCGIteration<F> {
    /// Iteration number (indexed from 0)
    pub k: i32,
    /// Gradient norm at the beginning of the iteration
    pub grad_norm: F,
    /// Function value at the beginning of the iteration
    pub value: F,
    /// `beta` coefficient for the nonlinear CG search direction update
    pub beta: F,
    /// Line minimization result
    pub alpha: F,
    /// Number of function evaluations by the line minimization method
    pub line_eval_count: i32,
}

impl NonlinearCG<f32> {
    /// Defaults for `f32` type: values mostly based on [HZ'06]
    pub fn new() -> Self {
        NonlinearCG {
            method: NonlinearCGMethod::HagerZhang(0.01),
            line_method: f32::secant2_default(),
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
            line_method: f64::secant2_default(),
            alpha0: 1.,
            psi2: 2.,
            grad_norm_tol: 1e-3,
            max_iter: 100,
        }
    }
}

impl<F: Float> NonlinearCG<F> {
    /// Mininimize the given nonlinear function over a linear space.
    ///
    /// The function `f` must provide its value as well as its gradient,
    /// returned in the provided `&mut V` (to avoid allocation).
    /// `x0` is used as the initial guess.
    pub fn minimize<Func, V>(&self,
                                       x0: &V,
                                       f: &mut Func) -> Result<V, NonlinearCGError>
        where Func: FnMut(&V, &mut V) -> F,
              V : Lin<F=F> + Clone {
        self.minimize_with_trace(x0, f, &mut |_, _| {})
    }

    /// The same as `minimize`, but allows to pass in a callback function that
    /// is called after every iteration.
    /// It is provided with the new point after the iteration is finished,
    /// and with additional information about the performed iteration.
    pub fn minimize_with_trace<Func, V, Callback>(&self,
                                       x0: &V,
                                       f: &mut Func,
                                       callback: &mut Callback) -> Result<V, NonlinearCGError>
        where Func: FnMut(&V, &mut V) -> F,
              V : Lin<F=F> + Clone,
              Callback: FnMut(&V, NonlinearCGIteration<V::F>) {
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
                V::F::zero()
            } else {
                match self.method {
                    NonlinearCGMethod::SteepestDescent => V::F::zero(),
                    NonlinearCGMethod::HagerZhang(eta) => {
                        // g_{k+1} - g_k
                        y.clone_from(&g_k_1);
                        y.ray_to(&g_k, -V::F::one());
                        let dk_yk = d_k.dot(&y);
                        let two = V::F::one() + V::F::one();
                        let betan_k = (y.dot(&g_k_1)
                                       - two * d_k.dot(&g_k_1) * y.norm_squared() / dk_yk) / dk_yk;
                        let eta_k = -V::F::one() / (d_k.norm() * eta.min(g_k.norm()));
                        betan_k.max(eta_k)
                    },
                }
            };

            // compute new direction
            d_k_1 = { d_k.combine(beta, &g_k_1, -V::F::one()); d_k };
            assert!(d_k_1.dot(&g_k_1) < V::F::zero());

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
                Err(e) => return Err(NonlinearCGError::LineMethodError(e)),
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

#[cfg(test)]

mod test {
    use super::*;

    #[test]
    fn quadratic() {
        let s = Secant2::new();

        fn f(x: f64) -> (f64, f64) {
            (x * (x - 1.), 2. * x - 1.)
        }

        let r = s.find_wolfe(5., &mut f, None);

        assert!(r.is_ok());
    }

    #[test]
    fn quartic() {
        let s = Secant2::new();

        fn f(x: f64) -> (f64, f64) {
            (0.25 * x.powi(4) - 0.7066666* x.powi(3) + 0.611 * x * x - 0.102 * x,
                (x - 0.1) * (x - 1.) * (x - 1.02))
        }

        let r = s.find_wolfe(1.025, &mut f, None);

        assert!(r.is_ok());
    }

    #[test]
    fn quadratic_wrong_dir() {
        let s = Secant2::new();

        // gradient has wrong sign
        fn f(t: f64) -> (f64, f64) {
            ((1. + 2. * t).powi(2), -4. * (1. + 2. * t))
        }

        let r = s.find_wolfe(1., &mut f, None);

        match r {
            Err(Secant2Error::InitBracketMaxIterReached(_)) => (),
            _ => panic!("unexpected result: {:?}", r),
        }
    }
}
