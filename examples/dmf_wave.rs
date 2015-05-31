//! Discrete Morse Flow solver of the 1D wave equation.
extern crate gnuplot;

extern crate linear_space;

use gnuplot::{Figure, Caption, Color, Fix, AxesCommon, PlotOption, DashType};
use linear_space::lin::{Lin,Rn};
use linear_space::ncg::{NonlinearCG};
use std::f64::consts::PI;

// Discrete Morse Flow energy for the wave equation.
fn wave_energy(u: &Rn<f64>, u_n: &Rn<f64>, u_n_1: &Rn<f64>,
               grad_u: &mut Rn<f64>, h: f64, kappa: f64) -> f64 {
    let mut v = u.clone();
    v.ray_to(&u_n, -2.);
    v.add_mut(&u_n_1);

    grad_u[0] = 0.;
    grad_u[u.len() - 1] = 0.;
    for i in 1..u.len() - 1 {
       grad_u[i] = (4. * v[i] + v[i-1] + v[i+1]) / (6. * h * h)
           + kappa * (2. * u[i] - u[i-1] - u[i+1]);
    }

    let mut e = 0.;
    for i in 1..u.len() {
        e += (v[i-1].powi(2) + v[i-1] * v[i] + v[i].powi(2)) / (6. * h * h)
            + (kappa / 2.) * (u[i - 1] - u[i]).powi(2);
    }
    e
}

fn main() {
    let domain = (0., 1.);
    let gamma = 1;
    let k = 32 * gamma;
    let h_dx_ratio = 0.1;
    let h = 1. / k as f64 * h_dx_ratio; // time-step
    let kappa = 1. * (k as f64 / (domain.1 - domain.0)).powi(2);
    let t_final = 1.5;
    let n_max = (t_final / h) as i32;
    let xi: Vec<f64> = (0..k+1).map(|i| domain.0 + (domain.1 - domain.0) * i as f64 / k as f64)
                               .collect();
    let u_n: Vec<f64> = xi.iter().map(|x| (PI * x).sin() + (4. * PI * x).sin()).collect();
    let mut u_n = Rn::new(u_n);
    let mut u_n_1 = u_n.clone();

    let mut ncg = NonlinearCG::new();
    let mut fg = Figure::new();

    for n in 1..n_max + 1 {
        let r = {
            let mut f = |x: &Rn<f64>, g: &mut Rn<f64>| wave_energy(x, &u_n, &u_n_1, g, h, kappa);
            ncg.minimize(&u_n, &mut f)
        };

        match r {
            Ok(v) => {
                println!("{:?}", v);
                let t = n as f64 * h;
                let exact_u = xi.iter().map(|x| (PI * x).sin() * (PI * t).cos()
                                            + (4. * PI * x).sin() * (4. * PI * t).cos());
                fg.clear_axes();
                fg.axes2d()
                    .set_y_range(Fix(-1.0), Fix(1.0))
                    .lines(xi.iter(), v.iter(), &[])
                    .lines(xi.iter(), exact_u, &[PlotOption::LineStyle(DashType::Dash)]);
                fg.show();
                u_n_1 = u_n;
                u_n = v;
            },
            Err(e) => {
                println!("Error: {:?}", e);
                return;
            },
        }
    }
}
