using LinearAlgebra

"""
    commutator!(C, A, B, α)
    commutator!(C, A, B)

Compute the commutator
```math
    C = α(A B - B A)
```
in-place.

# Arguments
- `C`: preallocated matrix to store the commutator.
- `A`, `B`: input matrices.
- `α`: scalar multiplier (default 1).
"""
function commutator!(C, A, B, α)
    # C := α*A*B
    mul!(C, A, B, α, 0)
    # C := C - α*B*A = α[A, B]
    mul!(C, B, A, -α, 1)
end

commutator!(C, A, B) = commutator!(C, A, B, 1)

"""
    commutator(A, B, α)
    commutator(A, B)

Compute the commutator
```math
    α(A B - B A)
```

# Arguments
- `A`, `B`: input matrices.
- `α`: scalar multiplier (default 1).
"""
function commutator(A, B, α)
    C = similar(A)
    commutator!(C, A, B, α)
end

commutator(A, B) = commutator(A, B, 1)

"""
    conjugate!(Y, X, U, T)

Compute the unitary conjugation
```math
    Y = U X U'
```
in-place.

# Arguments
- `Y`: preallocated matrix to store the result.
- `X`: matrix to be conjugated.
- `U`: unitary matrix performing the conjugation.
- `T`: preallocated workspace for intermediate calculations.
"""
function conjugate!(Y, X, U, T)
    # T := U * X
    mul!(T, U, X)
    # Y := T * U' = U * X * U'
    mul!(Y, T, U')
    return Y
end

conjugate!(X, U, T) = conjugate!(X, X, U, T)

"""
    conjugate(X, U)

Compute the unitary conjugation
```math
    U * X * U'
```

# Arguments
- `X`: matrix to be conjugated.
- `U`: unitary matrix performing the conjugation.
"""
function conjugate(X, U)
    Y = similar(X)
    T = similar(X)
    conjugate!(Y, X, U, T)
    return Y
end
