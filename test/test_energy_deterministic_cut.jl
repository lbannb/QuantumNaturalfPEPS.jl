using QuantumNaturalfPEPS
using QuantumNaturalGradient
using LinearAlgebra
using Random
using Test
using ITensors, ITensorMPS
using Test

QNG = QuantumNaturalGradient
QNfP = QuantumNaturalfPEPS

#= 
    Settings
=#
params = Dict{Symbol, Any}(
    :T => Float64,
    :seed => 30062026, # random seed
    :Lx => 4,
    :Ly => 4,
    :bdim => 2,
    :sample_nr => 100,
    :lr => 0.05,
    :t => 1.0,
    :U => 0.2,
    :eigencut => 1e-4,
    :contract_cutoff => 1e-10,  # tight cut so ψ_cut is a faithful wavefunction — otherwise the
    :sample_cutoff  => 1e-10,   # sampler/energy approximation mismatch swamps the variationality
    :contract_dim => 10_000,    # we are trying to test (see deterministic-cut discussion)
    :maxiter => 10,
    :α_init => 2.0,
    :parity_sector => 0, # 0=even, 1=odd
    :target_state => 0 # target ground state, 0 for ground state, 1 for first excited state, etc.
)
N = params[:Lx] * params[:Ly]

@show params
@show params[:t], params[:U]

Random.seed!(params[:seed])

hilbert = siteinds("Fermion", params[:Lx], params[:Ly])
hilbert_peps = siteinds("S=1/2", params[:Lx], params[:Ly])

# create PEPS with the specified parameters
samplecut = marginalcut = bdim = params[:bdim]
peps = PEPS(params[:T], hilbert_peps; bond_dim=bdim, 
    contract_cutoff=params[:contract_cutoff], 
    contract_dim=params[:contract_dim], 
    sample_cutoff=params[:sample_cutoff])

# Multiply the spectrum of the PEPS by a power-law factor as described in arXiv/2503.12557
α = params[:α_init]
bs = collect(1:bdim)
spectrum = bs .^ (-α)
QNfP.multiply_spectrum!(peps, spectrum)

ham_op = QNfP.TensorOperatorSum(QNfP.hamiltonian_hubbard(params[:t], params[:U], params[:Lx], params[:Ly]), hilbert)

#= Gaussian state settings =#
#= 
    start with π-flux state
    ○ χᵢ,ᵢ₊ₓ = χᵢ,ᵢ₊y = χ*(-1)^i_x
    ○ Δᵢ,ⱼ = 0
    ○ a₀(i) = 0
=#
n_max_MF_params = QNfP.get_max_num_MF_params_NN(params[:Lx], params[:Ly])
n_max_hopping_params_x = QNfP.get_max_num_hopping_x_NN(params[:Lx], params[:Ly])
n_max_hopping_params_y = QNfP.get_max_num_hopping_y_NN(params[:Lx], params[:Ly])
η = zeros(n_max_MF_params)
χ = 1.0

# staggered onsite potential
m_cdw = 1.0 # staggered onsite potential strength
for y in 1:params[:Ly], x in 1:params[:Lx]
    idx = QuantumNaturalfPEPS.col_major_site(x, y, params[:Lx])
    η[idx] = -m_cdw * (-1)^(x + y)
end
η[N + 1: N + n_max_hopping_params_x + n_max_hopping_params_y] .= -0.05

trial_state = QNfP.GaussianState(QNfP.build_general_H_BdG_2D_NN, N; η=η, parity_sector=params[:parity_sector], target_state=params[:target_state])

# Generate Operators for QNG
Oks_and_Eks = QNfP.generate_Oks_and_Eks(peps, ham_op; trial_state=trial_state, slow_energy=true)

θ_PEPS = QNfP.vec(peps)
θ = Vector{eltype(θ_PEPS)}(vcat(θ_PEPS, η))

# Setup the Integrator and Solver
integrator = QuantumNaturalGradient.Euler(lr=params[:lr])
solver = QNG.EigenSolver(params[:eigencut])

function build_history_callback()
    E_history = Vector{Vector{Float64}}()

    callback = function (; state, misc, niter)
        push!(E_history, copy(misc["history"][!, :energy]))
    end

    return callback, E_history
end

callback, E_history = build_history_callback()

@time loss_value, trained_θ, misc = QNG.evolve(Oks_and_Eks, θ; 
    integrator, 
    verbosity=2,
    callback,
    solver,
    sample_nr=params[:sample_nr],
    maxiter=params[:maxiter])

@show loss_value

E_ED = -12.499494752891966

# very Energy must be above ED
for E in E_history[end]
    @test E_ED <= E
end