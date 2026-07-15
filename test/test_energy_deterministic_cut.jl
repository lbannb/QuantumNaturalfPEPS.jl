using QuantumNaturalfPEPS
using QuantumNaturalGradient
using Random
using Test
using ITensors, ITensorMPS

QNfP = QuantumNaturalfPEPS

@testset "Deterministic environment cut" begin

    #=
        Tests for the deterministic environment cut (slow_energy):
        a) argument routing of slow_energy / slow_energy_interval through generate_Oks_and_Eks()
        b) WITHOUT the cut, the truncated amplitude ψ(S) is context-dependent: the same flipped
           configuration S' gets a different value inside E_loc (evaluated through the parent
           sample's environments) than when evaluated from scratch at the fixed cut
        c) WITH slow_energy=true, on identical samples only the energies change (every amplitude
           in E_loc is evaluated at the fixed cut), while the drawn logψ stay the same
        Note: the sampled logψ(S) themselves are deterministic per S even on the fast path
        (in :full mode the sampler builds env_top with the full contract_dim), so comparing
        duplicate samples' logψs cannot discriminate the two paths — the inconsistency lives
        in the flipped amplitudes inside E_loc.
    =#

    T = Float64
    Lx = Ly = 4
    N = Lx * Ly
    bdim = 2
    contract_cutoff = 1e-6

    Random.seed!(30062026)

    hilbert = siteinds("Fermion", Lx, Ly)
    hilbert_peps = siteinds("S=1/2", Lx, Ly)

    # contract_dim=2 is below the exact boundary bond dimension, so the environment
    # contraction is truncated and ψ(S) is only single-valued if the cut is deterministic
    peps = QNfP.PEPS(T, hilbert_peps; bond_dim=bdim, contract_dim=2, contract_cutoff=contract_cutoff, sample_cutoff=1e-10)

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

    @testset "Argument routing in generate_Oks_and_Eks" begin
        sample_nr = 5
        # default: slow_energy=false, fast path runs
        f_fast = QNfP.generate_Oks_and_Eks(peps, ham_op; trial_state=trial_state)
        res = f_fast(θ, sample_nr)
        for k in (:Oks, :Eks, :logψs, :samples, :weights, :contract_dims)
            @test haskey(res, k)
        end
        @test length(res[:Eks]) == sample_nr

        # slow_energy=true with interval=2: calls 2 and 4 use the deterministic cut, calls 1 and 3 the fast path
        f_slow = QNfP.generate_Oks_and_Eks(peps, ham_op; trial_state=trial_state, slow_energy=true, slow_energy_interval=2)
        res1 = f_slow(θ, sample_nr)
        res2 = f_slow(θ, sample_nr)
        res3 = f_slow(θ, sample_nr)
        res4 = f_slow(θ, sample_nr)
        @test length(res1[:Eks]) == sample_nr # fast-path call runs through
        @test length(res3[:Eks]) == sample_nr

        # on the deterministic-cut calls, logψ must equal a from-scratch evaluation at the fixed cut
        logψ_det = QNfP.get_logψ_function(peps; pos=pos, trial_state=trial_state)
        for res_slow in (res2, res4)
            for (S, lψ) in zip(res_slow[:samples], res_slow[:logψs])
                @test lψ ≈ logψ_det(S)
            end
        end
    end

    @testset "Fast path: truncated ψ(S) is context-dependent" begin
        QNfP.update_double_layer_envs!(peps)
        logψ_det = QNfP.get_logψ_function(peps; pos=pos, trial_state=trial_state)

        # reconstruct the fast energy path for one sample, exactly as get_Ek does:
        # the flipped amplitudes ψ(S') are evaluated through THIS sample's truncated environments
        S, _, env_top = QNfP.get_sample(peps; trial_state=trial_state)
        logψ, env_top, env_down, _ = QNfP.get_logψ_and_envs(peps, S, env_top; overwrite=false)
        logψ += log(QNfP.get_amplitude(trial_state, collect(vec(S))))
        h_envs_r, h_envs_l = QNfP.get_all_horizontal_envs(peps, env_top, env_down, S)
        Ek_terms = QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, S; get_flip_sites=true)
        logψ_flipped = QNfP.get_logψ_flipped(peps, Ek_terms, env_top, env_down, S, logψ;
                                             trial_state=trial_state, h_envs_r=h_envs_r, h_envs_l=h_envs_l)

        # the drawn sample itself agrees with the fixed-cut evaluation ...
        @test logψ ≈ logψ_det(S)

        # ... but the SAME flipped configuration gets a different amplitude in E_loc than a
        # from-scratch evaluation at the fixed cut: under truncation ψ(S') is not single-valued
        diffs = Float64[]
        for (flip_term, lψ_f) in logψ_flipped
            flip_term == () && continue
            S_f = copy(S)
            for ((x, y), Sij) in flip_term
                S_f[x, y] = Sij
            end
            push!(diffs, abs(lψ_f - logψ_det(S_f)))
        end
        @test !isempty(diffs)
        @test maximum(diffs) > contract_cutoff # the truncation effect the deterministic cut removes
    end

    @testset "slow_energy=true evaluates E_loc at the fixed cut" begin
        sample_nr = 20
        QNfP.update_double_layer_envs!(peps)
        # identical RNG seed -> identical samples, only the energy path differs
        Random.seed!(1234)
        res_fast = QNfP.Oks_and_Eks_singlethread(peps, ham_op, sample_nr; trial_state=trial_state, slow_energy=false)
        Random.seed!(1234)
        res_slow = QNfP.Oks_and_Eks_singlethread(peps, ham_op, sample_nr; trial_state=trial_state, slow_energy=true)

        @test res_fast[:samples] == res_slow[:samples] # same configurations were drawn
        @test res_fast[:logψs] ≈ res_slow[:logψs] # drawn amplitudes are identical either way
        @test !(res_fast[:Eks] ≈ res_slow[:Eks]) # but the flipped amplitudes inside E_loc are not
    end
end