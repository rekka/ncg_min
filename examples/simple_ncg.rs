extern crate linear_space;

use linear_space::lin::{Rn,Lin};
use linear_space::ncg::{NonlinearCG};

fn quad(x: &f64, grad: &mut f64) -> f64 {
    println!("{}", x);
    *grad = 2. * x;
    x * x
}

fn main() {
    let m = NonlinearCG::new();

    let mut ev: Vec<f64> = vec![];

    let r = {
        let mut f = |x: &f64, grad: &mut f64| {ev.push(*x); *grad = 2. * x; x * x};
        m.minimize(&1f64, &mut f)
    };

    println!("f(x) = x^2");
    println!("\tNCG result: {:?}", r);
    println!("\tEvaluations: x = {:?}", ev);
}
