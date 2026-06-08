using LinearAlgebra
using SparseArrays
using KrylovKit
using RecursiveArrayTools
using Random
using Distributions
using RandomMatrices
using Logging

"""
    extractable_work(ρ, H, U)

Compute the extractable work from state `ρ` with Hamiltonian `H` using unitary
`U`, that is
```math
    W = \\mathrm{tr}(ρ H) - \\mathrm{tr}(U ρ U' H)
```

# Arguments
- `ρ`: density matrix (possibly unnormalised).
- `H`: Hamiltonian.
- `U`: unitary used to extract the work.
"""
function extractable_work(ρ, H, U)
    Einitial = real(tr(ρ * H))
    ρfinal = U * ρ * U'
    Efinal = real(tr(ρfinal * H))
    W = Einitial - Efinal
    return W
end

"""
    ergotropy(p, pE, E)

Compute the ergotropy for a state with eigenvalues `p` and energy populations
`pE` corresponding to energy levels `E`, that is
```math
    W = \\sum_i p_{E_i} E_i - \\sum_i p^{\\downarrow}_i E_i
```
where `p^{\\downarrow}` is the vector `p` sorted in descending order.

# Arguments
- `p`: vector of eigenvalues of the state.
- `pE`: vector of energy populations of the state.
- `E`: vector of energy levels.
"""
function ergotropy(p, pE, E)
    d = length(p)
    Eavg = sum(pE[i] * E[i] for i in 1:d)
    sorted_p = sort(p; rev=true)
    W = Eavg - sum(sorted_p[i] * E[i] for i in 1:d)
    return W
end

"""
    ergotropy(ρ, H)

Compute the ergotropy of a state `ρ` with respect to Hamiltonian `H`.

# Arguments
- `ρ`: density matrix.
- `H`: Hamiltonian.
"""
function ergotropy(ρ, H)
    p = real(eigvals(ρ))
    F = eigen(H)
    E = real(F.values)
    U = F.vectors
    pE = real(diag(U' * ρ * U))
    return ergotropy(p, pE, E)
end

function ergotropy(ρ, H::SparseMatrixCSC)
    d = size(H, 1)
    p = real(eigvals(ρ))
    F = eigsolve(H, rand(ComplexF64, d), d, :SR; krylovdim=2d, ishermitian=true)
    E = real(F[1])
    U = Matrix{ComplexF64}(undef, d, d)
    U .= VectorOfArray(F[2])
    pE = real(diag(U' * ρ * U))
    return ergotropy(p, pE, E)
end

"""
    ergotropy_objective(V, ρ, H, X, T)

Evaluate the objective function used in ergotropy optimisation, which is given
by
``math
    \\mathrm{tr}(V ρ V' H)
```

# Arguments
- `V`: unitary matrix.
- `ρ`: density matrix.
- `H`: Hamiltonian.
- `X`, `T`: preallocated workspaces used for intermediate computations.
"""
function ergotropy_objective(V, ρ, H, X, T)
    # X := V * ρ * V'
    conjugate!(X, ρ, V, T)
    # T := X * H = V * ρ * V' * H
    mul!(T, X, H)
    return real(tr(T))
end

"""
    ergotropy_gradient!(G, V, ρ, H, X, T)

Compute the gradient of the ergotropy objective function with respect to the
unitary `V` and write it in-place into the preallocated matrix `G`.

# Arguments
- `G`: preallocated matrix to store the gradient in-place.
- `V`: unitary matrix.
- `ρ`: density matrix.
- `H`: Hamiltonian.
- `X`, `T`: preallocated workspaces used for intermediate computations.
"""
function ergotropy_gradient!(G, V, ρ, H, X, T)
    # X := V * ρ * V'
    conjugate!(X, ρ, V, T)
    # G := [H, X]
    commutator!(G, H, X)
    return G
end

"""
    ergotropy_linesearch!(Vnext, Vprev, ρ, H, G, X, T; c=1e-4, α=1.0, β=0.7, maxiter=10_000)

Backtracking line search for the ergotropy optimisation unitary update using
the Armijo stopping condition.

# Arguments
- `Vnext`: preallocated matrix to store the updated unitary in-place.
- `Vprev`: previous unitary.
- `ρ`: density matrix.
- `H`: Hamiltonian.
- `G`: gradient matrix (must be skew-Hermitian).
- `X`, `T`: preallocated workspaces used for intermediate computations.
- `c`: Armijo condition parameter (default 1e-4).
- `α`: initial step size parameter (default 1.0).
- `β`: step size reduction factor (default 0.7).
- `maxiter`: maximum number of line search iterations (default 10,000).
"""
function ergotropy_linesearch!(
    Vnext,
    Vprev,
    ρ,
    H,
    G,
    X,
    T;
    c=1e-4,
    α=1.0,
    β=0.7,
    maxiter=10_000
)
    normG = norm(G)
    # η = α/normG
    η = α

    obj = ergotropy_objective(Vprev, ρ, H, X, T)
    normGsq = normG^2

    cayley_step!(Vnext, Vprev, G, η, X, T)
    obj_new = ergotropy_objective(Vnext, ρ, H, X, T)

    iter = 1
    while (obj_new ≤ obj - c * η * normGsq) && (iter ≤ maxiter)
        η *= 1/β
        cayley_step!(Vnext, Vprev, G, η, X, T)
        obj_new = ergotropy_objective(Vnext, ρ, H, X, T)
        iter += 1
    end

    while (obj_new > obj - c * η * normGsq) && (iter ≤ maxiter)
        η *= β
        cayley_step!(Vnext, Vprev, G, η, X, T)
        obj_new = ergotropy_objective(Vnext, ρ, H, X, T)
        iter += 1
    end

    if iter > maxiter
        @warn "Line search failed to converge (maxiter reached)"
    end

    return obj_new, η
end

"""
    ergotropy_optimisation(ρ, H; rtol=1e-6, atol=0.0, maxiter=100_000, maxiter_linesearch=10_000)
    ergotropy_optimisation(ρ, H, Vinit; rtol=1e-6, atol=0.0, maxiter=100_000, maxiter_linesearch=10_000)

Run gradient-based optimisation over unitaries to maximise the extractable work
for state `ρ` and Hamiltonian `H`. Returns a tuple with the optimal ergotropy
value `W` and the optimal unitary `Vopt`. If provided, `Vinit` is used as the
initial unitary for the gradient descent; otherwise, a random Haar unitary is
generated.

# Arguments
- `ρ`: density matrix.
- `H`: Hamiltonian.
- `Vinit`: (optional) initial unitary for the optimisation. If not provided, a Haar-random unitary is used.
- `rtol`: relative tolerance for convergence (default 1e-6).
- `atol`: absolute tolerance for convergence (default 0.0).
- `grtol`: gradient norm relative tolerance for convergence (default 1e-8).
- `gatol`: gradient norm absolute tolerance for convergence (default 0.0).
- `maxiter`: maximum number of optimisation iterations (default 100,000).
- `maxiter_linesearch`: maximum number of line search iterations (default 10,000).

# Returns
- `Wopt`: optimal extractable work.
- `Vopt`: unitary achieving the optimum.
"""
function ergotropy_optimisation(
    ρ,
    H;
    rtol=1e-16,
    atol=0.0,
    grtol=1e-8,
    gatol=0.0,
    maxiter=600_000,
    maxiter_linesearch=10_000
)
    # Initialise unitary as Haar random
    d = size(ρ, 1)
    Vinit = similar(ρ)
    copy!(Vinit, rand(Haar(2), d))
    return ergotropy_optimisation(ρ, H, Vinit;
                                  rtol, atol, grtol, gatol, maxiter, maxiter_linesearch)
end

function ergotropy_optimisation(
    ρ,
    H,
    Vinit;
    rtol=1e-16,
    atol=0.0,
    grtol=1e-8,
    gatol=0.0,
    maxiter=600_000,
    maxiter_linesearch=10_000
)
    Eavg = real(tr(ρ * H))

    # Cache for the transformed state
    T = similar(Vinit)
    X = similar(Vinit)

    # Cache for the gradient
    G = similar(Vinit)

    # Cache for the previous and next unitaries
    Vprev = copy(Vinit)
    Vnext = similar(Vinit)

    # Initial objective and gradient
    obj = ergotropy_objective(Vprev, ρ, H, X, T)
    ergotropy_gradient!(G, Vprev, ρ, H, X, T)
    normG = norm(G)

    # Gradient descent loop
    normG0 = normG
    obj0 = 0.0
    η0 = 1.0
    iter = 1
    while (abs(obj - obj0) ≥ rtol*abs(Eavg - obj) + atol) && (normG ≥ grtol*normG0 + gatol) && (iter ≤ maxiter)
        # Line search to find next unitary
        obj0, η0 = ergotropy_linesearch!(Vnext, Vprev, ρ, H, G, X, T;
                                         α=η0, maxiter=maxiter_linesearch)

        # Update gradient
        ergotropy_gradient!(G, Vnext, ρ, H, X, T)
        normG = norm(G)

        # Swap prev and next
        Vprev, Vnext = Vnext, Vprev
        obj0, obj = obj, obj0

        iter += 1
    end

    if iter > maxiter
        @warn "Minimisation failed to converge to desired accuracy (maxiter reached)"
    end

    Vopt = Vprev
    opt = ergotropy_objective(Vopt, ρ, H, X, T)
    Wopt = Eavg - opt
    return Wopt, Vopt
end
