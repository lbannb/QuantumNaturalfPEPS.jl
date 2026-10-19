using Test
using LinearAlgebra
using Random
using QuantumNaturalfPEPS
using ITensors
using ITensorMPS

function _sampling_cache_hamiltonian(parameters, number_of_sites)
    hopping, pairing, chemical_potential = parameters
    particle_block = diagm(
        0 => fill(-chemical_potential, number_of_sites),
        1 => fill(-hopping, number_of_sites - 1),
        -1 => fill(-conj(hopping), number_of_sites - 1),
    )
    pairing_block = diagm(
        1 => fill(pairing, number_of_sites - 1),
        -1 => fill(-pairing, number_of_sites - 1),
    )
    return Hermitian([
        particle_block pairing_block
        pairing_block' -transpose(particle_block)
    ])
end

@testset "Projected parent-Slater Schur cache" begin
    Lx = Ly = 4
    number_of_sites = Lx * Ly
    gaussian_state = cs_state(
        Lx,
        Ly;
        η=[10.0],
        hopping_amplitude=0.2,
        stripe_delta=0.2,
        particle_number=number_of_sites,
    )
    projected_state = gutzwiller_project(gaussian_state)
    fixed_orbital_state = gutzwiller_project(projected_state.occupied_orbitals)
    @test fixed_orbital_state isa FixedGutzwillerProjectedState
    @test fixed_orbital_state.correlation_matrix ≈
        projected_state.correlation_matrix atol=1e-12
    changed_basis_state = gutzwiller_project(
        projected_state.occupied_orbitals * Diagonal(1:number_of_sites),
    )
    @test changed_basis_state.correlation_matrix ≈
        fixed_orbital_state.correlation_matrix atol=1e-12
    rank_deficient_orbitals = copy(projected_state.occupied_orbitals)
    rank_deficient_orbitals[:, end] .= rank_deficient_orbitals[:, 1]
    @test_throws ArgumentError gutzwiller_project(rank_deficient_orbitals)
    order = [x + (y - 1) * Lx for x in 1:Lx for y in 1:Ly]
    cache = QuantumNaturalfPEPS.ProjectedGaussianSchurCache(projected_state; order)
    fixed_orbital_cache = QuantumNaturalfPEPS.ProjectedGaussianSchurCache(
        fixed_orbital_state;
        order,
    )
    prefix = Dict{Int,Int}()
    prefix_probability = 1.0

    for site in order
        probabilities = projected_conditional_probabilities(cache)
        fixed_probabilities = projected_conditional_probabilities(
            fixed_orbital_cache,
        )
        @test fixed_probabilities ≈ probabilities atol=1e-10
        exact = map(0:1) do spin
            candidate = copy(prefix)
            candidate[site] = spin
            QuantumNaturalfPEPS.get_prob(projected_state, candidate) /
                prefix_probability
        end
        @test probabilities ≈ exact atol=1e-10
        spin = argmax(probabilities) - 1
        prefix[site] = spin
        prefix_probability = QuantumNaturalfPEPS.get_prob(projected_state, prefix)
        QuantumNaturalfPEPS.condition_projected_gaussian!(cache, spin)
        QuantumNaturalfPEPS.condition_projected_gaussian!(fixed_orbital_cache, spin)
    end

    target_Sz = 1.0
    target_Nup = Int(number_of_sites / 2 + target_Sz)
    fixed_state = gutzwiller_project(
        projected_state.occupied_orbitals;
        Nup=target_Nup,
    )
    @test fixed_state isa FixedGutzwillerProjectedState
    fixed_cache = QuantumNaturalfPEPS.ProjectedGaussianSchurCache(fixed_state; order)
    for position in eachindex(order)
        probabilities = projected_conditional_probabilities(fixed_cache)
        spin = if fixed_cache.measured_up < fixed_cache.target_Nup
            0
        else
            1
        end
        QuantumNaturalfPEPS.condition_projected_gaussian!(fixed_cache, spin)
        if position == length(order)
            @test fixed_cache.measured_up == fixed_cache.target_Nup
        end
        @test all(>=(0), probabilities)
    end

    hilbert = siteinds("S=1/2", Lx, Ly)
    peps = PEPS(ComplexF64, hilbert; bond_dim=1, show_warning=false)
    write!(peps, fill(ComplexF64(inv(sqrt(2))), length(peps)))
    sample, _, _ = QuantumNaturalfPEPS.get_sample(
        peps;
        trial_state=fixed_state,
        lookahead_depth=1,
    )
    sampled_Sz = sum(spin == 0 ? 0.5 : -0.5 for spin in sample)
    @test sampled_Sz == target_Sz
end

@testset "Gaussian low-rank sampling caches" begin
    number_of_sites = 4
    state = QuantumNaturalfPEPS.GaussianState(
        _sampling_cache_hamiltonian,
        number_of_sites;
        η=[1.0 + 0.2im, 0.7 - 0.1im, 0.2],
        parity_sector=0,
    )

    @testset "direct sampler defaults to Schur" begin
        Random.seed!(8723)
        sample, log_probability = QuantumNaturalfPEPS.get_sample(state)
        configuration = collect(vec(sample))
        @test exp(log_probability) ≈
            QuantumNaturalfPEPS.get_prob(state, configuration) atol=1e-10
        @test mod(sum(configuration), 2) == state.parity_sector
    end
end

@testset "Gaussian per-sample caches against the direct formulas" begin
    QNfP = QuantumNaturalfPEPS
    L = 4
    N = L^2
    rng = Xoshiro(1507)
    η = randn(rng, QNfP.get_max_num_MF_params_NN(L, L))
    order = [QNfP.col_major_site(i, j, L) for i in 1:L for j in 1:L]

    for parity_sector in (0, 1)
        GS = QNfP.GaussianState(QNfP.build_general_H_BdG_2D_NN, N; η, parity_sector)

        @testset "Schur sampling cache: conditional probabilities (parity $parity_sector)" begin
            cache = QNfP.GaussianSchurSamplingCache(GS; order)
            prefix = Dict{Int,Int}()
            for site in order
                p = QNfP.gaussian_conditional_probabilities(cache)
                joint = map(0:1) do n
                    candidate = copy(prefix)
                    candidate[site] = n
                    QNfP.get_prob(GS, candidate)
                end
                @test sum(p) ≈ 1 atol=1e-12
                @test p ./ sum(p) ≈ joint ./ sum(joint) atol=1e-12
                n = p[1] > 1e-6 && (p[2] <= 1e-6 || rand(rng) < 0.5) ? 0 : 1
                prefix[site] = n
                QNfP.condition_gaussian!(cache, n)
            end
            @test mod(sum(values(prefix)), 2) == parity_sector
        end

        @testset "sparse ∂H in the Majorana basis (parity $parity_sector)" begin
            c = GS.loggrad_cache
            for (i, dH) in enumerate(QNfP.build_H_BdG_derivatives(GS.H_BdG_func, GS.η, GS.N))
                dense = QNfP.transform_H_to_majorana_qq(Matrix(dH))
                from_entries = zeros(size(dense))
                for p in c.dH_ptr[i]:c.dH_ptr[i+1]-1
                    from_entries[c.dH_rows[p], c.dH_cols[p]] += c.dH_vals[p]
                end
                @test from_entries ≈ dense atol=1e-14
            end
        end

        @testset "log-derivatives against the dense ∂ηΓ (parity $parity_sector)" begin
            dΓs = QNfP.build_slater_loggradient_cache(GS).dΓs
            for _ in 1:5
                S, _ = QNfP.get_sample(GS)
                Ok = QNfP.get_Ok(GS, S, zeros(length(η)))
                F = 2 .* GS.Γ .+ QNfP.build_occ_projector_matrix(QNfP.to_occ_dict(collect(vec(S))), N)
                Ok_dense = [real(0.25 * tr(F \ dΓ)) for dΓ in dΓs]
                @test maximum(abs, Ok .- Ok_dense) <= 1e-10 * max(1, maximum(abs, Ok_dense))
            end
        end

        @testset "flipped amplitudes against the full Pfaffian (parity $parity_sector)" begin
            for _ in 1:5
                S, _ = QNfP.get_sample(GS)
                reference = collect(vec(S))
                amplitude = QNfP.get_amplitude(GS, reference)
                flips = QNfP.trial_amplitude_cache(GS, S)
                @test flips isa QNfP.GaussianFlipAmplitudeCache
                @test QNfP.get_amplitude(flips, reference) == amplitude
                # one to five changed sites: hops, pair creation/annihilation, odd changes (zero), full-Pfaffian fallback
                for _ in 1:100
                    sites = randperm(rng, N)[1:rand(rng, 1:5)]
                    flipped = copy(reference)
                    flipped[sites] .= 1 .- flipped[sites]
                    @test isapprox(QNfP.get_amplitude(flips, flipped), QNfP.get_amplitude(GS, flipped);
                                   atol=1e-10 * abs(amplitude))
                end
            end
        end

        @testset "joint sampler uses the Schur cache (parity $parity_sector)" begin
            @test QNfP._direct_sampling_cache(GS, order) isa QNfP.GaussianSchurSamplingCache
            @test QNfP._direct_sampling_cache(QNfP.FrozenTrialState(GS), order) isa QNfP.GaussianSchurSamplingCache
            # a uniform product PEPS leaves the Gaussian weights unchanged: same draws as get_sample(GS)
            hilbert = siteinds("S=1/2", L, L)
            peps = PEPS(Float64, hilbert; bond_dim=1, show_warning=false)
            write!(peps, fill(inv(sqrt(2)), length(peps)))
            for seed in 1:3
                Random.seed!(seed)
                S_joint, logpc_joint, _ = QNfP.get_sample(peps; trial_state=GS)
                Random.seed!(seed)
                S_gaussian, logpc_gaussian = QNfP.get_sample(GS)
                @test S_joint == S_gaussian
                @test logpc_joint ≈ logpc_gaussian atol=1e-10
                @test exp(logpc_gaussian) ≈ QNfP.get_prob(GS, collect(vec(S_gaussian))) rtol=1e-8
            end
        end
    end
end
