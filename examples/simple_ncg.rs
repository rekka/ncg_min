extern crate ncg_min;

use ncg_min::NonlinearCG;

fn quad2d(x: &[f64], grad: &mut [f64]) -> f64 {
    assert_eq!(x.len(), 2);
    assert_eq!(grad.len(), 2);

    grad[0] = 2. * x[0];
    grad[1] = 20. * x[1];

    x[0].powi(2) + 10. * x[1].powi(2)
}

fn main() {
    let m = NonlinearCG::<f64>::new();

    let mut ev: Vec<f64> = vec![];

    let r = {
        let f = |x: &[f64], grad: &mut [f64]| {
            ev.push(x[0].clone());
            grad[0] = 2. * x[0];
            x[0] * x[0]
        };
        m.minimize(&[1f64], f)
    };

    println!("f(x) = x^2");
    println!("\tNCG result: {:?}", r);
    println!("\tEvaluations: x = {:?}", ev);


    let mut ev: Vec<Vec<f64>> = vec![];

    println!("f(x) = x1^2 + 10 x2^2");

    let r = {
        let f = |x: &[f64], grad: &mut [f64]| {
            ev.push(x.to_owned());
            quad2d(x, grad)
        };
        let x0 = vec![1., 1.];
        m.minimize_with_trace(&x0, f, |x, info| {
            println!("{:?}, {:?}", x, info);
        })
    };

    println!("\tNCG result: {:?}", r);
    println!("\tEvaluations: x = {:?}", ev);
}
