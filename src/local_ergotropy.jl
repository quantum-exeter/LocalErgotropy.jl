using LinearAlgebra
using Random
using Distributions
using RandomMatrices
using Logging

"""
    local_extractable_work(ρ0, ρ1, H0, H1, U)

Compute the weighted extractable work for two (possibly unnormalised)
states `ρ0`, `ρ1` with corresponding Hamiltonians `H0`, `H1` under the
same unitary `U`, that is
```math
    W = p_0 W_0 + p_1 W_1
```
where `p_n = \\mathrm{tr}(ρ_n)` and `W_n` is the extractable work from state
`σ_n = ρ_n / p_n` with Hamiltonian `H_n` using unitary `U`.

# Arguments
- `ρ0`, `ρ1`: local density matrices (possibly unnormalised).
- `H0`, `H1`: Hamiltonians for the respective states.
- `U`: unitary used to compute the extractable work.
"""
function local_extractable_work(ρ0, ρ1, H0, H1, U)
    p0 = tr(ρ0)
    p1 = tr(ρ1)
    σ0 = ρ0 / p0
    σ1 = ρ1 / p1
    W0 = extractable_work(σ0, H0, U)
    W1 = extractable_work(σ1, H1, U)
    W = p0*W0 + p1*W1
    return W
end

"""
    local_ergotropy_objective(V, ρ0, ρ1, H0, H1, X, T)

Evaluate the objective function used in the local ergotropy optimisation, which
is given by
``math
    \\mathrm{tr}(V ρ_0 V' H_0) + \\mathrm{tr}(V ρ_1 V' H_1)
```
# Arguments
- `V`: unitary matrix.
- `ρ0`, `ρ1`: local density matrices (possibly unnormalised).
- `H0`, `H1`: Hamiltonians for the respective states.
- `X`, `T`: preallocated workspaces used for intermediate computations.
"""
function local_ergotropy_objective(V, ρ0, ρ1, H0, H1, X, T)
    obj0 = ergotropy_objective(V, ρ0, H0, X, T)
    obj1 = ergotropy_objective(V, ρ1, H1, X, T)
    return obj0 + obj1
end

"""
    local_ergotropy_gradient!(G, V, ρ0, ρ1, H0, H1, X, T)

Compute the gradient of the local ergotropy objective with respect to the
unitary `V` and write it in-place into the preallocated matrix `G`.

# Arguments
- `G`: preallocated matrix to store the gradient.
- `V`: unitary matrix.
- `ρ0`, `ρ1`: local density matrices (possibly unnormalised).
- `H0`, `H1`: Hamiltonians for the respective states.
- `X`, `T`: preallocated workspaces used for intermediate computations.
"""
function local_ergotropy_gradient!(G, V, ρ0, ρ1, H0, H1, X, T)
    ergotropy_gradient!(G, V, ρ0, H0, X, T)
    # this reuses T which is safe due to the implementation of ergotropy_gradient!
    ergotropy_gradient!(T, V, ρ1, H1, X, T)
    G .+= T
    return G
end

"""
    local_ergotropy_linesearch!(Vnext, Vprev, ρ0, ρ1, H0, H1, G, X, T; c=1e-4, α=1.0, β=0.7, maxiter=10_000)

Backtracking line search for the local ergotropy optimisation unitary update
using the Armijo stopping condition.

# Arguments
- `Vnext`: preallocated matrix to store the updated unitary in-place.
- `Vprev`: previous unitary.
- `ρ0`, `ρ1`: local density matrices (possibly unnormalised).
- `H0`, `H1`: Hamiltonians for the respective states.
- `G`: gradient matrix (must be skew-Hermitian).
- `X`, `T`: preallocated workspaces used for intermediate computations.
- `c`: Armijo condition parameter (default 1e-4).
- `α`: initial step size (default 1.0).
- `β`: step size reduction factor (default 0.7).
- `maxiter`: maximum number of line search iterations (default 10,000).
"""
function local_ergotropy_linesearch!(
    Vnext,
    Vprev,
    ρ0,
    ρ1,
    H0,
    H1,
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

    obj = local_ergotropy_objective(Vprev, ρ0, ρ1, H0, H1, X, T)
    normGsq = normG^2

    cayley_step!(Vnext, Vprev, G, η, X, T)
    obj_new = local_ergotropy_objective(Vnext, ρ0, ρ1, H0, H1, X, T)

    iter = 1
    while (obj_new ≤ obj - c * η * normGsq) && (iter ≤ maxiter)
        η *= 1/β
        cayley_step!(Vnext, Vprev, G, η, X, T)
        obj_new = local_ergotropy_objective(Vnext, ρ0, ρ1, H0, H1, X, T)
        iter += 1
    end

    while (obj_new > obj - c * η * normGsq) && (iter ≤ maxiter)
        η *= β
        cayley_step!(Vnext, Vprev, G, η, X, T)
        obj_new = local_ergotropy_objective(Vnext, ρ0, ρ1, H0, H1, X, T)
        iter += 1
    end

    if iter > maxiter
        @warn "Line search failed to converge (maxiter reached)"
    end

    return obj_new, η
end

"""
    local_ergotropy_optimisation(ρ0, ρ1, H0, H1; rtol=1e-6, atol=0.0, maxiter=100_000, maxiter_linesearch=10_000) 
    local_ergotropy_optimisation(ρ0, ρ1, H0, H1, Vinit; rtol=1e-6, atol=0.0, maxiter=100_000, maxiter_linesearch=10_000) 

Run gradient-based optimisation over unitaries to maximise the local extractable work
for states `ρ0`, `ρ1` and Hamiltonians `H0`, `H1`. Returns a tuple with the
optimal ergotropy value `W` and the optimal unitary `Vopt`. If provided, `Vinit`
is used as the initial unitary for the gradient descent; otherwise, a random
Haar unitary is generated.

# Arguments
- `ρ0`, `ρ1`: local density matrices (possibly unnormalised).
- `H0`, `H1`: Hamiltonians for the respective states.
- `Vinit`: (optional) initial unitary for the optimisation. If not provided, a Haar-random unitary is used. 
- `rtol`: relative tolerance for convergence (default 1e-6).
- `atol`: absolute tolerance for convergence (default 0.0).
- `grtol`: gradient norm relative tolerance for convergence (default 1e-8).
- `gatol`: gradient norm absolute tolerance for convergence (default 0.0).
- `maxiter`: maximum number of optimisation iterations (default 100,000).
- `maxiter_linesearch`: maximum number of line search iterations (default 10,000

Returns
- `Wopt`: optimal local extractable work.
- `Vopt`: unitary achieving the optimum (approximate).
"""
function local_ergotropy_optimisation(
    ρ0,
    ρ1,
    H0,
    H1;
    rtol=1e-16,
    atol=0.0,
    grtol=1e-8,
    gatol=0.0,
    maxiter=600_000,
    maxiter_linesearch=10_000
)
    # Initialise unitary as Haar random
    d = size(ρ0, 1)
    elT = eltype(ρ0)
    Vinit = zeros(elT, d, d)
    copy!(Vinit, rand(Haar(2), d))
    return local_ergotropy_optimisation(ρ0, ρ1, H0, H1, Vinit;
                                        rtol, atol, grtol, gatol, maxiter, maxiter_linesearch)
end

function local_ergotropy_optimisation(
    ρ0,
    ρ1,
    H0,
    H1,
    Vinit;
    rtol=1e-16,
    atol=0.0,
    grtol=1e-8,
    gatol=0.0,
    maxiter=600_000,
    maxiter_linesearch=10_000
)
    @assert size(ρ1, 1) == size(ρ0, 1)
    Eavg = real(tr(ρ0 * H0)) + real(tr(ρ1 * H1))

    # Cache for the transformed state
    T = similar(Vinit)
    X = similar(Vinit)

    # Cache for the gradient
    G = similar(Vinit)

    # Cache for the previous and next unitaries
    Vprev = copy(Vinit)
    Vnext = similar(Vinit)

    # Initial objective and gradient
    obj = local_ergotropy_objective(Vprev, ρ0, ρ1, H0, H1, X, T)
    local_ergotropy_gradient!(G, Vprev, ρ0, ρ1, H0, H1, X, T)
    normG = norm(G)

    # Gradient descent loop
    normG0 = normG
    obj0 = 0.0
    η0 = 1.0
    iter = 1
    while (abs(obj - obj0) ≥ rtol*abs(Eavg - obj) + atol) && (normG ≥ grtol*normG0 + gatol) && (iter ≤ maxiter)
        # Line search to find next unitary
        obj0, η0 = local_ergotropy_linesearch!(Vnext, Vprev, ρ0, ρ1, H0, H1, G, X, T;
                                               α=η0, maxiter=maxiter_linesearch)

        # Update gradient
        local_ergotropy_gradient!(G, Vnext, ρ0, ρ1, H0, H1, X, T)
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
    opt = local_ergotropy_objective(Vopt, ρ0, ρ1, H0, H1, X, T)
    Wopt = Eavg - opt
    return Wopt, Vopt
end
