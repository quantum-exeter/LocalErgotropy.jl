module LocalErgotropy

public commutator, commutator!, conjugate, conjugate!
include("linalg.jl")

public random_density_matrix_eigen, random_density_matrix
public random_hamiltonian_spectrum, random_hamiltonian
include("random_operators.jl")

include("cayley.jl")

export extractable_work, ergotropy, ergotropy_optimisation
include("ergotropy.jl")

export local_extractable_work, local_ergotropy_optimisation
include("local_ergotropy.jl")

end
