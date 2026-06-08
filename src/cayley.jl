using LinearAlgebra
using LinearAlgebra: BLAS, BlasFloat

"""
    cayley_step!(Vnew, V, G, η, M, GV)

Compute the gradient descent unitary update via the Cayley transform.
The Cayley update is given by:
```math
    V_{new} = \\left(I + \\frac{η}{2} G\\right)^{-1} \\left(V - \\frac{η}{2} G V\\right)
```

# Arguments
- `Vnew`: preallocated matrix to store the updated unitary in-place.
- `V`: previous unitary.
- `G`: skew-Hermitian gradient.
- `η`: step size (real scalar).
- `M`: preallocated workspace for intermediate calculations.
- `GV`: preallocated workspace for intermediate calculations.
"""
function cayley_step!(
    Vnew::StridedMatrix{T},
    V::StridedMatrix{T},
    G::StridedMatrix{T},
    η::Real,
    M::StridedMatrix{T},
    GV::StridedMatrix{T}
) where {T<:BlasFloat}

    n = size(V, 1)
    @assert size(V,2) == n
    @assert size(G) == (n,n)
    @assert size(M) == (n,n)
    @assert size(GV) == (n,n)
    @assert size(Vnew) == (n,n)

    halfη = η/2

    # GV := G*V
    mul!(GV, G, V)

    # Vnew := V - halfη*GV
    Vnew .= V .- halfη .* GV

    # M := I + halfη*G
    M .= halfη .* G
    M[1:size(M,1)+1:end] .+= one(eltype(M))

    # Solve M * Vnew = Vnew in-place
    F = lu!(M)
    ldiv!(F, Vnew)

    return Vnew
end
