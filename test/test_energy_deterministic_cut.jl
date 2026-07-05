using QuantumNaturalfPEPS
using QuantumNaturalGradient
using Random
using Test
using ITensors, ITensorMPS

QNfP = QuantumNaturalfPEPS

#=
    Tests for the deterministic environment cut (slow_energy):
    a) argument routing of slow_energy / slow_energy_interval through generate_Oks_and_Eks()
    b) with the deterministic cut, the amplitude ψ(S) evaluates to the same value for all
       identical samples S — with APPROXIMATE environment contraction (contract_dim is
       deliberately not exact), since that is the regime the deterministic cut is for
=#

T = Float64
Lx = Ly = 4
N = Lx * Ly
bdim = 2

Random.seed!(30062026)

hilbert = siteinds("Fermion", Lx, Ly)
hilbert_peps = siteinds("S=1/2", Lx, Ly)

# contract_dim=2 is below the exact boundary bond dimension, so the environment
# contraction is truncated and ψ(S) is only single-valued if the cut is deterministic
peps = QNfP.PEPS(T, hilbert_peps; bond_dim=bdim, contract_dim=2, contract_cutoff=1e-6, sample_cutoff=1e-10)

# Multiply the spectrum of the PEPS by a power-law factor as described in arXiv/2503.12557
α = 2.0
QNfP.multiply_spectrum!(peps, collect(1:bdim) .^ (-α))

ham_op = QNfP.TensorOperatorSum(QNfP.hamiltonian_hubbard(1.0, 0.2, Lx, Ly), hilbert)

# π-flux Gaussian trial state with a staggered onsite potential; the peaked sampling
# distribution guarantees duplicate samples among the draws in test b)
n_max_MF_params = QNfP.get_max_num_MF_params_NN(Lx, Ly)
n_max_hopping_params_x = QNfP.get_max_num_hopping_x_NN(Lx, Ly)
n_max_hopping_params_y = QNfP.get_max_num_hopping_y_NN(Lx, Ly)
η = zeros(n_max_MF_params)
m_cdw = 1.0
for y in 1:Ly, x in 1:Lx
    idx = QNfP.col_major_site(x, y, Lx)
    η[idx] = -m_cdw * (-1)^(x + y)
end
η[N+1:N+n_max_hopping_params_x+n_max_hopping_params_y] .= -0.05

trial_state = QNfP.GaussianState(QNfP.build_general_H_BdG_2D_NN, N; η=η, parity_sector=0, target_state=0)

θ = Vector{T}(vcat(QNfP.vec(peps), η))

pos = (size(peps, 1) - 1) ÷ 2 # default cut position used by slow_energy

@testset "a) argument routing in generate_Oks_and_Eks" begin
    # default: slow_energy=false, fast path runs
    f_fast = QNfP.generate_Oks_and_Eks(peps, ham_op; trial_state=trial_state)
    res = f_fast(θ, 5)
    for k in (:Oks, :Eks, :logψs, :samples, :weights, :contract_dims)
        @test haskey(res, k)
    end
    @test length(res[:Eks]) == 5

    # slow_energy=true with interval=2: calls 1 and 3 use the deterministic cut, call 2 the fast path
    f_slow = QNfP.generate_Oks_and_Eks(peps, ham_op; trial_state=trial_state, slow_energy=true, slow_energy_interval=2)
    res1 = f_slow(θ, 5)
    res2 = f_slow(θ, 5)
    res3 = f_slow(θ, 5)
    @test length(res2[:Eks]) == 5 # fast-path call runs through

    # on the deterministic-cut calls, logψ must equal a from-scratch evaluation at the fixed cut
    logψ_det = QNfP.get_logψ_function(peps; pos=pos, trial_state=trial_state)
    for res_slow in (res1, res3)
        for (S, lψ) in zip(res_slow[:samples], res_slow[:logψs])
            @test lψ ≈ logψ_det(S)
        end
    end
end

@testset "b) deterministic cut gives single-valued ψ(S)" begin
    QNfP.update_double_layer_envs!(peps)
    res = QNfP.Oks_and_Eks_singlethread(peps, ham_op, 1000; trial_state=trial_state, slow_energy=true)

    groups = Dict{Matrix{Int}, Vector{ComplexF64}}()
    for (S, lψ) in zip(res[:samples], res[:logψs])
        push!(get!(groups, S, ComplexF64[]), lψ)
    end

    dup = [v for v in values(groups) if length(v) > 1]
    @test !isempty(dup) # otherwise this test would be vacuous
    for v in dup
        @test all(lψ ≈ v[1] for lψ in v)
    end
end