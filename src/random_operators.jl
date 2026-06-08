using LinearAlgebra
using Random
using Distributions
using RandomMatrices

"""
    random_density_matrix_eigen(d)

Generate a random density matrix with eigen-decomposition `ρ = U * Diagonal(p) *
U'` where `p` is drawn uniformly from the `d`-simplex and `U` is Haar-random.

# Arguments
- `d`: dimension of the Hilbert space.

# Returns
- `p`: vector of random eigenvalues of the density matrix.
- `U`: Random unitary eigenbasis of the density matrix.
"""
function random_density_matrix_eigen(d)
    p = rand(Dirichlet(d, 1.0))
    U = rand(Haar(2), d)
    return p, U
end

"""
    random_density_matrix(d)

Return a random density matrix of dimension `d×d` where the eigenvalues are
drawn uniformly from the `d`-simplex and the eigenvectors are Haar-random.

# Arguments
- `d`: dimension of the Hilbert space.
"""
function random_density_matrix(d)
    p, U = random_density_matrix_eigen(d)
    ρ = U * Diagonal(p) * U'
    return ρ
end

"""
    random_hamiltonian_spectrum(d; Emin=0.0, Emax=1.0)

Sample `d` energy levels uniformly in the interval `[Emin, Emax]` and return
them sorted in ascending order.

# Arguments
- `d`: number of energy levels to generate.
- `Emin`: minimum energy (default 0.0).
- `Emax`: maximum energy (default 1.0).
"""
function random_hamiltonian_spectrum(d, Emin=0.0, Emax=1.0)
    E = sort(rand(Uniform(Emin, Emax), d))
    return E
end

"""
    random_hamiltonian(d; Emin=0.0, Emax=1.0)

Construct a random Hamiltonian with eigenvalues sampled uniformly
from `[Emin, Emax]` and eigenvectors drawn from the Haar measure.

# Arguments
- `d`: matrix dimension.
- `Emin`, `Emax`: energy range (defaults 0.0 and 1.0).
"""
function random_hamiltonian(d, Emin=0.0, Emax=1.0)
    E = random_hamiltonian_spectrum(d, Emin, Emax)
    U = rand(Haar(2), d)
    H = U * Diagonal(E) * U'
    return H
end
