//! Implementation of the `secant2` line minimization method by _Hager & Zhang'06_.
//!
//! The method is slightly modified to avoid some corner cases where the
//! original method failed.

use num_traits::Float;

/// Implementation of the `secant2` line minimization method by _Hager & Zhang'06_.
#[derive(Debug,Clone)]
pub struct Secant2<S: Float> {
    /// `delta` for Wolfe condition
    pub delta: S,
    /// `sigma` for Wolfe condition
    pub sigma: S,
    /// `epsilon` for approximate Wolfe condition (to allow for value function to increase
    /// because of rounding of errors when close to the minimum)
    pub epsilon: S,
    /// Bisection coefficient when secant fails; allowed values in `(0, 1)`
    /// (`0.5` is the midpoint of the interval)
    pub theta: S,
    /// Extension factor for finding the initial bracketing interval; `> 1`
    pub rho: S,
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

impl Default for Secant2<f32> {
    // Defaults for `secant2` method given in [HZ'06]
    fn default() -> Self {
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

impl Default for Secant2<f64> {
    // Defaults for `secant2` method given in [HZ'06]
    fn default() -> Secant2<f64> {
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

impl Secant2<f32> {
    pub fn new() -> Self {
        Default::default()
    }
}

impl Secant2<f64> {
    pub fn new() -> Self {
        Default::default()
    }
}

enum BracketResult<S> {
    Ok((S, S, S), (S, S, S)),
    MaxIterReached(i32),
    // Wolfe(S), // TODO: implement early escape if solution is found during bracketing
}

impl<S: Float> Secant2<S> {
    /// Find an approximate minimum of a function _ϕ_ satisfying the Wolfe condition.
    ///
    ///   - `f` should be the function `f = |x| (ϕ(x), ϕ'(x))`.
    ///   - `c` specifies the initial search interval `(0, c)`.
    ///   - `hint` may contain the value `(ϕ(0), ϕ'(0))` to avoid unnecessary evalution if
    ///     this value is already known (as is the case of the `NonlinearCG` minimization method).
    pub fn find_wolfe<Func>(&self,
                            c: S,
                            mut f: Func,
                            hint: Option<(S, S)>) -> Result<S, Secant2Error>
        where Func: FnMut(S) -> (S, S) {
        assert!(c > S::zero());

        let mut f = |x| { let (fx, fdx) = f(x); (x, fx, fdx) };

        // save origin
        let o = match hint {
            Some((v, d)) => (S::zero(), v, d),
            None => f(S::zero()),
        };

        // ϕ(0) + ε
        let f0_eps = o.1 + self.epsilon;
        // bracket starting interval
        let mut ab = match self.bracket(o, f(c), &mut f) {
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
            if cx > ctheta && b.1 + (cx - b.0) * b.2 > f0_eps {
                // Taylor series estimate tells us that phi(b) does not decrease enough
                // => let's help it
                cx = ctheta;
            }

            let c = f(cx);
            if self.wolfe(c, f0_eps, o.2) { return Ok(c.0); }

            let mut cx;
            if c.2 >= S::zero() {
                cx = secant(b, c);
                ab.1 = c;
            } else if c.1 <= f0_eps {
                cx = secant(a, c);
                ab.0 = c;
            } else {
                ab = match self.ubracket(a, c, &mut f, f0_eps) {
                    BracketResult::Ok(a, b) => (a, b),
                    // BracketResult::Wolfe(x) => return Ok(x),
                    BracketResult::MaxIterReached(n) =>
                        return Err(Secant2Error::UBracketMaxIterReached(n)),
                };

                continue;
            }
            let (a, b) = ab;

            if cx <= a.0 || b.0 <= cx {
                // here we diverge from `secant2` method: if the second secant
                // produces a point outside of the bracket interval, let's bisect
                // TODO: this should be maybe handled as in Brent's method
                cx = a.0 + self.theta * (b.0 - a.0);
            }

            let c = f(cx);
            if self.wolfe(c, f0_eps, o.2) { return Ok(c.0); }

            if c.2 >= S::zero() {
                ab.1 = c;
            } else if c.1 <= f0_eps {
                ab.0 = c;
            } else {
                ab = match self.ubracket(a, c, &mut f, f0_eps) {
                    BracketResult::Ok(a, b) => (a, b),
                    // BracketResult::Wolfe(x) => return Ok(x),
                    BracketResult::MaxIterReached(n) =>
                        return Err(Secant2Error::UBracketMaxIterReached(n)),
                };
            }

        }
        Err(Secant2Error::MaxIterReached(self.max_iter))
    }

    // Implementation of U3a--b bracketing loop in [HZ'06]
    //
    // The format of the triples is `(x, f(x), f'(x))`.
    //
    // - `f0_eps` is `\phi(0) + \epsilon_k`
    fn ubracket<Func>(&self,
                      mut a: (S, S, S),
                      mut b: (S, S, S),
                      mut f: Func, f0_eps: S) -> BracketResult<S>
        where Func: FnMut(S) -> (S, S, S) {
        // preconditions
        assert!(a.0 < b.0);
        assert!(a.2 < S::zero());
        assert!(b.2 < S::zero());
        assert!(a.1 <= f0_eps && f0_eps < b.1);

        for _ in 0..self.ubracket_max_iter {
            let cx = a.0 + self.theta * (b.0 - a.0);
            let c = f(cx);

            if c.2 >= S::zero() {
                return BracketResult::Ok(a, c);
            } else {
                if c.1 <= f0_eps {
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
                     mut a: (S, S, S),
                     mut b: (S, S, S),
                     mut f: Func) -> BracketResult<S>
        where Func: FnMut(S) -> (S, S, S) {
        // preconditions
        assert!(a.0 < b.0);
        assert!(a.2 < S::zero());

        let o = a;
        let f0_eps = o.1 + self.epsilon;

        for _ in 0..self.init_bracket_max_iter {
            if b.2 >= S::zero() {
                return BracketResult::Ok(a, b);
            } else if b.1 > f0_eps {
                return self.ubracket(o, b, f, f0_eps);
            } else {
                a = b;
                let bx = self.rho * b.0;
                b = f(bx);
            }
        }

        BracketResult::MaxIterReached(self.init_bracket_max_iter)
    }

    fn wolfe(&self, c: (S, S, S), f0_eps: S, fd0: S) -> bool {
        // approximate Wolfe condition
        // ϕ(x)≤ϕ(0)+ε && σϕ'(0)≤ϕ'(x)≤(2δ-1)ϕ'(0)
        c.1 <= f0_eps && self.sigma * fd0 <= c.2 && c.2 <= (self.delta + self.delta - S::one()) * fd0
    }

}

fn secant<S: Float>(a: (S, S, S), b: (S, S, S)) -> S {
    (a.0 * b.2 - b.0 * a.2) / (b.2 - a.2)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn quadratic() {
        let s: Secant2<_> = Default::default();

        fn f(x: f64) -> (f64, f64) {
            (x * (x - 1.), 2. * x - 1.)
        }

        let r = s.find_wolfe(5., f, None);

        assert!(r.is_ok());
    }

    #[test]
    fn quartic() {
        let s: Secant2<_> = Default::default();

        fn f(x: f64) -> (f64, f64) {
            (0.25 * x.powi(4) - 0.7066666* x.powi(3) + 0.611 * x * x - 0.102 * x,
                (x - 0.1) * (x - 1.) * (x - 1.02))
        }

        let r = s.find_wolfe(1.025, f, None);

        assert!(r.is_ok());
    }

    #[test]
    fn quadratic_wrong_dir() {
        let s: Secant2<_> = Default::default();

        // gradient has wrong sign
        fn f(t: f64) -> (f64, f64) {
            ((1. + 2. * t).powi(2), -4. * (1. + 2. * t))
        }

        let r = s.find_wolfe(1., f, None);

        match r {
            Err(Secant2Error::InitBracketMaxIterReached(_)) => (),
            _ => panic!("unexpected result: {:?}", r),
        }
    }

    // This example breaks the original `secant2` method.
    // A solution is to perform a bisection.
    #[test]
    fn not_good_for_secant() {
        let s: Secant2<_> = Default::default();

        fn f(t: f64) -> (f64, f64) {
            let a = 0.001;
            let x  = t - 1.;
            let s = (a * a + x * x).sqrt();
            (0.5 * (x * (s + x) - a * a * (s + x).ln()),
                x * x / s + x)
        }

        let r = s.find_wolfe(2., f, None);

        assert!(r.is_ok());
    }
}
