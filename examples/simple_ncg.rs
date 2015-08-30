extern crate ncg_min;

use ncg_min::{Rn, NonlinearCG};

fn quad2d(x: &Rn<f64>, grad: &mut Rn<f64>) -> f64 {
    assert_eq!(x.len(), 2);
    assert_eq!(grad.len(), 2);

    grad[0] = 2. * x[0];
    grad[1] = 20. * x[1];

    x[0].powi(2) + 10. * x[1].powi(2)
}

fn main() {
    let m = NonlinearCG::new();

    let mut ev: Vec<f64> = vec![];

    let r = {
        let f = |x: &f64, grad: &mut f64| {ev.push(x.clone()); *grad = 2. * x; x * x};
        m.minimize(&1f64, f)
    };

    println!("f(x) = x^2");
    println!("\tNCG result: {:?}", r);
    println!("\tEvaluations: x = {:?}", ev);


    let mut ev: Vec<Rn<f64>> = vec![];

    println!("f(x) = x1^2 + 10 x2^2");

    let r = {
        let f = |x: &Rn<f64>, grad: &mut Rn<f64>| {ev.push(x.clone()); quad2d(x, grad)};
        let x0 = Rn::new(vec![1.,1.]);
        m.minimize_with_trace(&x0, f, |x, info| {
            println!("{:?}, {:?}", x, info);
        })
    };

    println!("\tNCG result: {:?}", r);
    println!("\tEvaluations: x = {:?}", ev);
}
