using MKL
using MKLSparse
using LinearAlgebra
using LocalErgotropy

const E0 = 1.0
const r0 = 0.3
const n0 = 0.2
const ϕn0 = 1.0
const e1 = 0.4
const E1 = 2.0
const ν1 = 0.5
const ϕν1 = 1.6
const r1 = 0.4
const n1 = 0.3
const ϕn1 = 0.7

const H00 = ComplexF64[ 0 0 ; 0 E0 ]
const H11 = ComplexF64[ e1 ν1*cis(ϕν1) ; ν1*cis(-ϕν1) E1 ]
const ρ0 = ComplexF64[ r0 n0*cis(ϕn0) ; n0*cis(-ϕn0) 1 - r0]
const ρ1 = ComplexF64[ r1 n1*cis(ϕn1) ; n1*cis(-ϕn1) 1 - r1]
const Eavg = real(tr(ρ0 * H00)) + real(tr(ρ1 * H11))

const rtol, grtol = 1e-16, 1e-16
const maxiter = 10_000_000
W_local, V_opt = local_ergotropy_optimisation(ρ0, ρ1, H00, H11; rtol, grtol, maxiter)

println("===============================================")
println("Karen's qubit example.")
println("Average energy: $Eavg")
println("Optimal energy extracted via local operations: $W_local")
println("Local ergotropy objective: $(Eavg - W_local)")
println("Karen's ergotropy objective: 0.820597")
