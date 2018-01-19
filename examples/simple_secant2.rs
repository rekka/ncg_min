extern crate ncg_min;

use ncg_min::secant2::Secant2;

fn secant2_with_tracking<Func>(m: &Secant2<f64>, c: f64, f: &Func)
where
    Func: Fn(f64) -> (f64, f64),
{
    let mut ev = vec![];

    let r;
    {
        let mut g = |x| {
            ev.push(x);
            f(x)
        };
        r = m.find_wolfe(c, &mut g, None);
    }

    println!("Found point {:?}", r);
    println!("Function evaluations: {:?}", ev);
}

fn main() {
    let m = Secant2::<f64>::new();
    // bad initial range guess
    let f = |x: f64| (x * (x - 1.), 2. * x - 1.);
    println!("f(x) = x (x - 1)");
    secant2_with_tracking(&m, 0.001, &f);

    // example of a functions where original secant2 method fails (gets stuck at the local
    // minimum at 1.02)
    let f = |x: f64| {
        (
            0.25 * x.powi(4) - 0.7066666 * x.powi(3) + 0.611 * x * x - 0.102 * x,
            (x - 0.1) * (x - 1.) * (x - 1.02),
        )
    };
    println!("f'(x) = (x - 0.1) (x - 1) (x - 1.02)");
    secant2_with_tracking(&m, 1.025, &f);
}
