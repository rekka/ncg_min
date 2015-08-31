# ncg_min

[![Build Status](https://travis-ci.org/rekka/ncg_min.svg?branch=master)](https://travis-ci.org/rekka/ncg_min)

This [Rust] crate implements a [nonlinear conjugate gradient method][NCG
wiki] for numerical minimization of general differentiable functions
based only on their value and gradient.

## Documentation

[Full documentation](https://rekka.github.io/ncg_min)

Also see examples in `examples/` subdirectory.

## Installation

Add this crate to your `Cargo.toml`:

```toml
[dependencies.ncg_min]
git = "https://github.com/rekka/ncg_min"
```

[NCG wiki]: https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method
[Rust]: https://www.rust-lang.org/
