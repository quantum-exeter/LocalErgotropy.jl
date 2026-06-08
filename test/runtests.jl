using LocalErgotropy
using Test

using LinearAlgebra
using SparseArrays
using LocalErgotropy: random_density_matrix, random_hamiltonian_spectrum,
                      random_density_matrix_eigen, random_hamiltonian

@testset "Random density matrix generation" begin
    niter = 100
    for iter in 1:niter
        d = rand(2:20)
        ρ = random_density_matrix(d)
        @test isapprox(ρ, ρ'; rtol=1e-10, atol=1e-10)
        @test isapprox(tr(ρ), 1.0; rtol=1e-10, atol=1e-10)
        evals = real(eigvals(ρ))
        @test all(evals .≥ 0.0)
    end
end

@testset "Random Hamiltonian generation" begin
    niter = 100
    for iter in 1:niter
        d = rand(2:20)
        Emin = rand() * 10.0
        Emax = Emin + rand() * 10.0
        E = random_hamiltonian_spectrum(d, Emin, Emax)
        @test length(E) == d
        @test all(E .≥ Emin)
        @test all(E .≤ Emax)
        @test issorted(E)
    end
end

@testset "Ergotropy calculation" begin
    niter = 100
    for iter in 1:niter
        d = rand(2:8)
        E = random_hamiltonian_spectrum(d, 0.0, 10.0)
        H = Diagonal(E)
        p, U = random_density_matrix_eigen(d)
        ρ = U * Diagonal(p) * U'
        pE = real(diag(ρ))

        W_analytical = ergotropy(p, pE, E)
        W_diagonal = ergotropy(ρ, H)
        W_sparse = ergotropy(ρ, sparse(H))
        @test isapprox(W_analytical, W_diagonal; rtol=1e-10, atol=1e-10)
        @test isapprox(W_analytical, W_sparse; rtol=1e-10, atol=1e-10)

        W_numerical, V_opt = ergotropy_optimisation(ρ, H; rtol=1e-14)
        W_extracted = extractable_work(ρ, H, V_opt)
        @test isapprox(W_analytical, W_numerical; rtol=1e-5, atol=1e-6)
        @test isapprox(W_numerical, W_extracted; rtol=1e-5, atol=1e-6)
    end
end

@testset "Local ergotropy optimisation" begin
    niter = 10
    for iter in 1:niter
        d = rand(2:8)
        ρ0 = random_density_matrix(d)
        ρ1 = random_density_matrix(d)
        p0 = rand()
        p1 = 1.0 - p0
        rmul!(ρ0, p0)
        rmul!(ρ1, p1)
        H0 = random_hamiltonian(d)
        H1 = random_hamiltonian(d)

        W_local, V_opt = local_ergotropy_optimisation(ρ0, ρ1, H0, H1; rtol=1e-12)
        W_extracted = local_extractable_work(ρ0, ρ1, H0, H1, V_opt)
        @test isapprox(W_local, W_extracted; rtol=1e-5, atol=1e-5)

        W_local, _ = local_ergotropy_optimisation(ρ0/tr(ρ0), zero(ρ0), H0, zero(H0); rtol=1e-12)
        W_single, _ = ergotropy_optimisation(ρ0/tr(ρ0), H0; rtol=1e-14)
        @test isapprox(W_local, W_single; rtol=1e-5, atol=1e-5)
        W_local, _ = local_ergotropy_optimisation(zero(ρ1), ρ1/tr(ρ1), zero(H1), H1; rtol=1e-12)
        W_single, _ = ergotropy_optimisation(ρ1/tr(ρ1), H1; rtol=1e-14)
        @test isapprox(W_local, W_single; rtol=1e-5, atol=1e-5)
    end

    # Test against Karen's qubit example

    E0 = 1.0
    r0 = 0.3
    n0 = 0.2
    ϕn0 = 1.0
    e1 = 0.4
    E1 = 2.0
    ν1 = 0.5
    ϕν1 = 1.6
    r1 = 0.4
    n1 = 0.3
    ϕn1 = 0.7

    H00 = ComplexF64[ 0 0 ; 0 E0 ]
    H11 = ComplexF64[ e1 ν1*cis(ϕν1) ; ν1*cis(-ϕν1) E1 ]
    ρ0 = ComplexF64[ r0 n0*cis(ϕn0) ; n0*cis(-ϕn0) 1 - r0]
    ρ1 = ComplexF64[ r1 n1*cis(ϕn1) ; n1*cis(-ϕn1) 1 - r1]
    Eavg = real(tr(ρ0 * H00)) + real(tr(ρ1 * H11))
    W_local, V_opt = local_ergotropy_optimisation(ρ0, ρ1, H00, H11; rtol=1e-16, grtol=1e-16, maxiter=10_000_000)
    @test isapprox(Eavg - W_local, 0.820597; rtol=1e-6, atol=1e-6)
end
