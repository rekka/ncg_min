# ncg_min

[![Build Status](https://travis-ci.org/rekka/ncg_min.svg?branch=master)](https://travis-ci.org/rekka/ncg_min)

This [Rust] crate implements a [nonlinear conjugate gradient
method][NCGwiki] for numerical minimization of general differentiable
functions based only on their value and gradient.

## Documentation

[Full documentation](https://rekka.github.io/ncg_min)

Also see examples in `examples/` subdirectory.

## Installation

Add this crate to your `Cargo.toml`:

```toml
[dependencies.ncg_min]
git = "https://github.com/rekka/ncg_min"
```

## TODO list

- More NCG methods...
- Constrained minimization: with respect to one or more constraints of
  the type `g(x) = 0`.
- Better API (`Lin` trait is kinda wonky).
- Accept [nalgebra] and [cgmath] data.

[NCGwiki]: https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
[Rust]: https://www.rust-lang.org/
[nalgebra]: http://nalgebra.org/
[cgmath]: https://github.com/bjz/cgmath-rs
