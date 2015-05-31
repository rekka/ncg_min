//! Implementation of a Nonlinear Conjugate Gradient method.

#![allow(dead_code)]
use num::{Float};

/// Implementation of `secant2` method by _Hager & Zhang'06_.
#[derive(Debug,Clone,Copy)]
pub struct Secant2<F: Float> {
    delta: F,
    sigma: F,
    epsilon: F,
    theta: F,
    rho: F,
    max_iter: i32,
}

#[derive(Debug,Clone,Copy)]
pub enum Secant2Error {
    MaxIterReached(i32),
}

impl Secant2<f64> {
    // Defaults for `secant2` method given in [HZ'06]
    pub fn new() -> Self {
        Secant2 {
            delta: 0.1,
            sigma: 0.9,
            epsilon: 1e-6,
            theta: 0.5,
            rho: 5.,
            max_iter: 32,
        }
    }
}

impl<F: Float> Secant2<F> {
    /// Find an approximate minimum satisfying the Wolfe condition of a given function.
    ///
    /// The search interval is `(0, c)`.
    pub fn find_wolfe<Func>(&self, c: F, f: &mut Func) -> Result<F, Secant2Error>
        where Func: FnMut(F) -> (F, F) {
        assert!(c > F::zero());

        // save origin
        let o = triple(f, F::zero());
        // ϕ(0) + ε_k
        let fi = o.1 + self.epsilon;
        // bracket starting interval
        let mut ab = self.bracket(o, triple(f, c), f);

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
                ab = self.ubracket(a, c, f, fi);
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
                    ab = self.ubracket(a, c, f, fi);
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
    fn ubracket<Func>(&self, mut a: (F, F, F), mut b: (F, F, F), f: &mut Func, fi: F) -> ((F, F, F), (F, F, F))
        where Func: FnMut(F) -> (F, F) {
        // preconditions
        assert!(a.0 < b.0);
        assert!(a.2 < F::zero());
        assert!(b.2 < F::zero());
        assert!(a.1 <= fi && fi < b.1);

        loop {
            let cx = a.0 + self.theta * (b.0 - a.0);
            let c = triple(f, cx);

            if c.2 >= F::zero() {
                return (a, c);
            } else {
                if c.1 <= fi {
                    a = c;
                } else {
                    b = c;
                }
            }
        }
    }

    //Initial bracketing: `bracket(c)` method in [HZ'06]
    fn bracket<Func>(&self, mut a: (F, F, F), mut b: (F, F, F), f: &mut Func) -> ((F, F, F), (F, F, F))
        where Func: FnMut(F) -> (F, F) {
        // preconditions
        assert!(a.0 < b.0);
        assert!(a.2 < F::zero());

        let o = a;
        let fi = o.1 + self.epsilon;

        loop {
            if b.2 >= F::zero() {
                return (a, b);
            } else if b.1 > fi {
                return self.ubracket(o, b, f, fi);
            } else {
                a = b;
                let bx = self.rho * b.0;
                b = triple(f, bx);
            }
        }

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


#[cfg(test)]

mod test {
    use super::Secant2;

    #[test]
    fn quadratic() {
        let s = Secant2::new();

        fn f(x: f64) -> (f64, f64) {
            (x * (x - 1.), 2. * x - 1.)
        }

        let r = s.find_wolfe(5., &mut f);

        assert!(r.is_ok());
    }

    #[test]
    fn quartic() {
        let s = Secant2::new();

        fn f(x: f64) -> (f64, f64) {
            (0.25 * x.powi(4) - 0.7066666* x.powi(3) + 0.611 * x * x - 0.102 * x,
                (x - 0.1) * (x - 1.) * (x - 1.02))
        }

        let r = s.find_wolfe(1.025, &mut f);

        assert!(r.is_ok());
    }
}
