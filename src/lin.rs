//! Implementation of a basic linear space trait.
//!
//! The main interface is `Lin` with an implementation for a vector of real
//! numbers: `Rn<S>`.
//!
//! ```rust
//! use ncg_min::{Lin, Rn};
//!
//! let x = Rn::new(vec![1.,2.]);
//! let y = Rn::new(vec![2.,-3.]);
//!
//! assert_eq!(x.dot(&y), -4.);
//! assert_eq!(Rn::new(vec![3., -1.]), x.clone() + &y);
//! assert_eq!(Rn::new(vec![-1., 5.]), x.clone() - y);
//! assert_eq!(Rn::new(vec![2., 4.]), x * 2.);
//! ```
use num::{Zero, One, Float};
use std::borrow::Borrow;
use std::ops::{Add, Mul, Sub, Div, Deref, DerefMut};
use std::iter::repeat;


/// Trait defining basic operations for an element of a linear space.
///
/// The focus is on operations _in place_: methods that return a `Lin` object
/// modify the object in place.
pub trait Lin {
    /// Scalars for this linear space.
    type S: Float;

    /// Dot product (inner product).
    fn dot(&self, other: &Self) -> Self::S;

    /// Multiplication by a constant.
    fn scale(&mut self, a: Self::S) -> &mut Self;

    /// Adds a vector multiplied by a constant to this vector.
    fn ray_to(&mut self, other: &Self, t: Self::S) -> &mut Self;

    /// Return the origin of the vector space to which self belongs too.
    fn origin(&self) -> Self;

    /// Norm of the vector.
    fn norm(&self) -> Self::S {
        self.norm_squared().sqrt()
    }

    /// Square of the norm.
    fn norm_squared(&self) -> Self::S {
        self.dot(self)
    }

    /// Scale the vector so that it has a norm one.
    ///
    /// If norm is zero, causes division by zero.
    fn normalize(&mut self) -> &mut Self {
        let norm = self.norm();
        self.scale(Self::S::one() / norm)
    }

    /// Distance between two vectors.
    ///
    /// Default implementation uses
    /// `|x - v| = sqrt(x.x - 2 x.y + y.y)`
    /// to avoid copying.
    /// Therefore it is recommended to reimplement this method.
    fn dist(&self, other: &Self) -> Self::S {
        (self.norm_squared() + other.norm_squared()
            - (Self::S::one() + Self::S::one()) * self.dot(other)).sqrt()
    }

    /// Adds a vector to this vector.
    fn add_mut(&mut self, other: &Self) -> &mut Self {
        self.ray_to(other, Self::S::one())
    }

    /// Creates a linear combination.
    fn combine(&mut self,  a: Self::S, other: &Self, b: Self::S) -> &mut Self {
        self.scale(a).ray_to(other,b)
    }

    /// Project on a line given by a given direction.
    ///
    /// `dir` does not have to be normalized, but must be nonzero.
    fn project_on(&mut self, dir: &Self) -> &mut Self {
        let a = self.dot(dir) / dir.norm_squared();
        self.combine(Self::S::zero(), dir, a)
    }

    /// Project on a plane orthogonal to the given direction.
    ///
    /// `dir` does not have to be normalized, but must be nonzero.
    fn project_ortho(&mut self, dir: &Self) -> &mut Self {
        let a = self.dot(dir) / dir.norm_squared();
        self.combine(Self::S::one(), dir, -a)
    }
}

pub trait Dot {
    type S: Float;

    /// Dot product (inner product).
    fn dot(&self, other: &Self) -> Self::S;

    /// Norm of the vector.
    fn norm(&self) -> Self::S {
        self.norm_squared().sqrt()
    }

    /// Square of the norm.
    fn norm_squared(&self) -> Self::S {
        self.dot(self)
    }
}

/// An implementation of the Lin trait: an n-dimensional real vector.
///
/// Backed by a `Vec<S>`, where `S` is `Float`.
#[derive(Clone,Debug,PartialEq)]
pub struct Rn<S: Float> {
    vec: Vec<S>,
}

impl<S: Float> Rn<S> {
    pub fn new(v: Vec<S>) -> Self {
        Rn {vec: v}
    }
}

impl<S: Float> Deref for Rn<S> {
    type Target = Vec<S>;

    fn deref<'a>(&'a self) -> &'a Vec<S> {
        &self.vec
    }
}

impl<S: Float> DerefMut for Rn<S> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut Vec<S> {
        &mut self.vec
    }
}

impl<S: Float> Mul<S> for Rn<S> {
    type Output = Rn<S>;

    fn mul(mut self, other: S) -> Self {
        for x in self.iter_mut() {
            *x = *x * other;
        }
        self
    }
}

impl<S: Float> Div<S> for Rn<S> {
    type Output = Rn<S>;

    fn div(mut self, other: S) -> Self {
        for x in self.iter_mut() {
            *x = *x / other;
        }
        self
    }
}

impl<S: Float, T> Add<T> for Rn<S> where T: Borrow<Rn<S>> {
    type Output = Rn<S>;

    fn add(mut self, other: T) -> Self {
        let other = other.borrow();
        assert_eq!(self.len(), other.len());
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x = *x + *y;
        }
        self
    }
}

impl<S: Float, T> Sub<T> for Rn<S> where T: Borrow<Rn<S>> {
    type Output = Rn<S>;

    fn sub(mut self, other: T) -> Self {
        let other = other.borrow();
        assert_eq!(self.len(), other.len());
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x = *x - *y;
        }
        self
    }
}

impl<S: Float> Dot for Rn<S> {
    type S = S;

    fn dot(&self, other: &Self) -> Self::S {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter())
            .fold(Self::S::zero(), |sum, (&x, &y)| sum + x * y)
    }
}

impl<S: Float> Lin for Rn<S> {
    type S = S;

    fn dist(&self, other: &Self) -> Self::S {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter())
            .fold(Self::S::zero(),
                |sum, (&x, &y)| sum + (x - y).powi(2))
            .sqrt()
    }

    fn dot(&self, other: &Self) -> Self::S {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter())
            .fold(Self::S::zero(), |sum, (&x, &y)| sum + x * y)
    }

    fn scale(&mut self, a: Self::S) -> &mut Self {
        for x in self.iter_mut() {
            *x = *x * a;
        }
        self
    }

    fn combine(&mut self,  a: Self::S, other: &Self, b: Self::S) -> &mut Self {
        assert_eq!(self.len(), other.len());
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x = *x * a + *y * b;
        }
        self
    }

    fn ray_to(&mut self,  other: &Self, b: Self::S) -> &mut Self {
        assert_eq!(self.len(), other.len());
        for (x, y) in self.iter_mut().zip(other.iter()) {
            *x = *x + *y * b;
        }
        self
    }

    fn origin(&self) -> Self {
        Rn::new(repeat(Self::S::zero()).take(self.len()).collect())
    }
}

// Trivial implementation of `Lin` for `f32` and `f64 as an element of a
// linear space over scalars of the same type.
// This cannot be done directly using
//
// ```
// impl<S: Float> Lin for S {
// ...
// ```
//
// since I am hitting
//
// ```
// error: conflicting implementations for trait `Lin` [E0119]
// ```
macro_rules! lin_float_impl {
    ($T:ident) => (
        impl Lin for $T {
            type S = $T;

            fn dot(&self, other: &Self) -> Self::S {
                *self * *other
            }

            fn scale(&mut self, a: Self::S) -> &mut Self {
                *self = *self * a;
                self
            }

            fn combine(&mut self,  a: Self::S, other: &Self, b: Self::S) -> &mut Self {
                *self = *self * a + *other * b;
                self
            }

            fn ray_to(&mut self,  other: &Self, b: Self::S) -> &mut Self {
                *self = *self + *other * b;
                self
            }

            fn origin(&self) -> Self {
                Self::S::zero()
            }

            fn dist(&self, other: &Self) -> Self::S {
                (*self - *other).abs()
            }

            fn norm(&self) -> Self::S {
                self.abs()
            }
        }
    )
}

lin_float_impl!(f32);
lin_float_impl!(f64);

#[cfg(test)]
mod test {
    extern crate quickcheck;

    use super::Lin;
    use super::Rn;
    use std::cmp::min;
    use num::Float;
    use self::quickcheck::quickcheck;

    // relative error: this shouldn't be to small
    const EPS: f64 = 1e-10;

    fn eps_eq(a: f64, b: f64) -> bool {
        let m = a.abs() + b.abs();
        (a - b).abs() <= EPS * m
    }

    fn trunc<T: Float>(v: Vec<T>, w: Vec<T>) -> (Rn<T>, Rn<T>) {
            let mut v = Rn::new(v);
            let mut w = Rn::new(w);
            let l = min(v.len(), w.len());
            v.truncate(l);
            w.truncate(l);
            (v, w)
    }

    #[test]
    fn test_lin_f64() {
        fn prop(a: f64, b: f64, c: f64) -> bool {
            let mut d = a;
            d.ray_to(&b,c);
            eps_eq(a.norm(), a.dist(&a.origin())) &&
                eps_eq(d.dot(&a), a.norm_squared() + c * a * b)

        }

        quickcheck(prop as fn(f64, f64, f64) -> bool);
        let mut a = 1.;
        a.combine(2.,&3.,4.);
        assert_eq!(a.norm(), 1. * 2. + 3. * 4.);
    }

    #[test]
    fn scale_norm_zero() {
        fn prop(v: Vec<f64>) -> bool {
            let mut v = Rn::new(v);
            eps_eq(0., v.scale(0.).norm())
        }
        quickcheck(prop as fn(Vec<f64>) -> bool);
    }

    #[test]
    fn dot_equal_norm_squared() {
        fn prop(v: Vec<f64>) -> bool {
            let v = Rn::new(v);
            eps_eq(v.dot(&v), v.norm().powi(2))
        }
        quickcheck(prop as fn(Vec<f64>) -> bool);
    }

    #[test]
    fn origin() {
        fn prop(v: Vec<f64>) -> bool {
            let v = Rn::new(v);
            let o = v.origin();
            eps_eq(o.norm(), 0.) && o.len() == v.len()
        }
        quickcheck(prop as fn(Vec<f64>) -> bool);
    }

    #[test]
    fn dist_norm() {
        fn prop(v: Vec<f64>, w: Vec<f64>) -> bool {
            let (mut v, w) = trunc(v, w);

            let d = v.dist(&w);
            v.ray_to(&w, -1.);

            eps_eq(d, v.norm())
        }
        quickcheck(prop as fn(Vec<f64>, Vec<f64>) -> bool);
    }

    #[test]
    fn combine_ray_to() {
        fn prop(v: Vec<f64>, w: Vec<f64>, a: f64) -> bool {
            let (mut v, w) = trunc(v, w);

            let mut v1 = v.clone();
            v.combine(1., &w, a);
            v1.ray_to(&w, a);

            eps_eq(v.dist(&v1), 0.)
        }
        quickcheck(prop as fn(Vec<f64>, Vec<f64>, f64) -> bool);
    }

    #[test]
    fn combine_dot() {
        fn prop(v: Vec<f64>, w: Vec<f64>, z: Vec<f64>, a: f64) -> bool {
            let mut v = Rn::new(v);
            let mut w = Rn::new(w);
            let mut z = Rn::new(z);
            let l = min(v.len(), min(w.len(), z.len()));
            v.truncate(l);
            w.truncate(l);
            z.truncate(l);

            let b = 3.5;

            let dv = v.dot(&z);
            let dw = w.dot(&z);

            v.combine(a, &w, b);

            let dvw = v.dot(&z);
            eps_eq(a * dv + b * dw, dvw)
        }
        quickcheck(prop as fn(Vec<f64>, Vec<f64>, Vec<f64>, f64) -> bool);
    }

    #[test]
    fn project_pythagoras() {
        fn prop(v: Vec<f64>, w: Vec<f64>) -> bool {
            let (v, w) = trunc(v, w);

            let mut p = v.clone();
            let mut perp = v.clone();

            p.project_on(&w);
            perp.project_ortho(&w);
            eps_eq(p.norm_squared() + perp.norm_squared(), v.norm_squared())
        }
        quickcheck(prop as fn(Vec<f64>, Vec<f64>) -> bool);
    }

}
