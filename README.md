# LocalErgotropy

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cerisola.github.io/LocalErgotropy.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cerisola.github.io/LocalErgotropy.jl/dev/)
[![Build Status](https://github.com/cerisola/LocalErgotropy.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cerisola/LocalErgotropy.jl/actions/workflows/CI.yml?query=branch%3Amain)

`LocalErgotropy.jl` is a Julia package to compute the local ergotropy of a
quantum system, that is the maximum amount of work that can be extracted from a
specific subsystem using only local unitary operations, without acting on the
rest of the surrounding environment or coupled systems.

Given the lack of known analytical solution to this problem, the package finds
the local ergotropy via optimisation, using gradient descent over the unitary
manifold.

The package also provides methods to compute the total ergotropy using the known
analytical solutions for this case.

NOTE: while `MKL.jl` is not a dependency of this package, it is recommended to
install it and try using it (import it before importing `LocalErgotropy`),
since in most scenarios it will accelerate the linear algebra calculations.

