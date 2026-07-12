"""
ERGM network simulation.

All sampling is built on a single Metropolis–Hastings edge-toggle kernel
(`_mh_run!`), exposed publicly as [`mh_sample`](@ref) (one parameterized
chain) and wrapped by [`sample_networks`](@ref) / [`simulate_ergm`](@ref)
(multi-network simulation, parallelized over independent chains).

Every function takes an `rng::AbstractRNG` keyword; all random draws flow
from it, so runs with the same seed are exactly reproducible. Parallel
functions seed one RNG per chain deterministically from the caller's `rng`,
so results are also independent of the number of threads.
"""

"""
    mh_sample(model::ERGMModel, θ::Vector{Float64};
              n_samples::Int=100, burnin::Int=10000, interval::Int=1000,
              rng::AbstractRNG=Random.default_rng(),
              start_net::Union{Nothing,Network}=nothing,
              return_networks::Bool=false)
        -> (stats::Matrix{Float64}, networks::Union{Nothing,Vector{<:Network}})

Run one Metropolis–Hastings edge-toggle chain of the ERGM `exp(θ'g(y))`
defined by `model`'s terms, and return the sampled sufficient statistics
(and optionally the sampled networks).

This is the public single-chain sampling primitive that the estimation
routines (and downstream packages such as ERGMEgo.jl) build on. Each MH
step proposes toggling a uniformly random dyad and accepts with probability
`min(1, exp(±θ'Δ))`, where `Δ` is the add-direction change statistic
vector of the dyad.

Dyads masked as missing (`Network.set_missing_dyad!`) are never proposed:
they are held fixed at their face value (edge present/absent as stored)
throughout the chain, i.e. the sampler conditions on the face values of the
unobserved dyads. Throws `ArgumentError` if every dyad is masked.

# Arguments
- `model::ERGMModel`: Model specification (terms + network template). The
  chain runs on a copy; `model.network` is never mutated.
- `θ::Vector{Float64}`: Natural-parameter vector, one entry per term.

# Keywords
- `n_samples::Int=100`: Number of recorded samples.
- `burnin::Int=10000`: MH steps discarded before the first sample.
- `interval::Int=1000`: MH steps between recorded samples (thinning).
- `rng::AbstractRNG=Random.default_rng()`: Source of all random draws;
  same rng state ⇒ identical output.
- `start_net=nothing`: Starting network. Defaults to a copy of the observed
  `model.network` (attributes are preserved either way).
- `return_networks::Bool=false`: Also collect a copy of the network at
  every sampling point.

# Returns
A NamedTuple `(stats, networks)`:
- `stats`: `n_samples × p` matrix; row `k` holds `g(y⁽ᵏ⁾)`.
- `networks`: `Vector` of the sampled networks if `return_networks=true`,
  otherwise `nothing`.

# Example
```julia
model = ERGMModel(ERGMFormula([Edges(), Triangle()]), net)
out = mh_sample(model, [-1.5, 0.3]; n_samples=500, rng=Xoshiro(1))
mean(out.stats, dims=1)   # E[g] estimate at θ
```
"""
function mh_sample(model::ERGMModel{T}, θ::Vector{Float64};
                   n_samples::Int=100,
                   burnin::Int=10000,
                   interval::Int=1000,
                   rng::AbstractRNG=Random.default_rng(),
                   start_net::Union{Nothing,Network}=nothing,
                   return_networks::Bool=false) where T
    length(θ) == length(model.formula.terms) ||
        throw(ArgumentError("length(θ) = $(length(θ)) does not match the " *
                            "number of model terms ($(length(model.formula.terms)))"))
    net = _copy_network(isnothing(start_net) ? model.network : start_net)
    stats, networks = _mh_run!(rng, net, model.formula.terms, θ, model.directed,
                               n_samples, burnin, interval, return_networks)
    return (stats=stats, networks=return_networks ? networks : nothing)
end

# The shared MH edge-toggle kernel: mutates `net` in place, returns the
# sampled statistics matrix and (when `collect_networks`) network copies at
# each sampling point. A function barrier: `terms` is passed concretely so
# the loop is fully typed.
function _mh_run!(rng::AbstractRNG, net::Network, terms::TermSet,
                  θ::Vector{Float64}, directed::Bool,
                  n_samples::Int, burnin::Int, interval::Int,
                  collect_networks::Bool)
    p = length(terms)
    n = Int(nv(net))

    # Masked (unobserved) dyads are held fixed at their face value: they are
    # never proposed for toggling, so the chain conditions on them.
    n_free_dyads = (directed ? n * (n - 1) : n * (n - 1) ÷ 2) - n_missing_dyads(net)
    n_free_dyads > 0 || throw(ArgumentError(
        "every dyad of the network is masked as missing; the MH sampler has " *
        "no free dyads to toggle"))

    samples = Matrix{Float64}(undef, n_samples, p)
    networks = Vector{typeof(net)}()
    collect_networks && sizehint!(networks, n_samples)

    current_stats = compute_all(terms, net)
    delta = Vector{Float64}(undef, p)
    total_steps = burnin + n_samples * interval
    sample_idx = 0

    for step in 1:total_steps
        # Propose a uniformly random dyad toggle (never a masked dyad)
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        while i == j || (!directed && j < i) || is_missing_dyad(net, i, j)
            i = rand(rng, 1:n)
            j = rand(rng, 1:n)
        end

        # Add-direction change statistics; MH log-ratio is θ'Δ for an
        # addition and −θ'Δ for a removal
        change_stat_all!(delta, terms, net, i, j)
        log_accept = dot(θ, delta)
        if has_edge(net, i, j)
            log_accept = -log_accept
        end

        if log(rand(rng)) < log_accept
            if has_edge(net, i, j)
                rem_edge!(net, i, j)
                current_stats .-= delta
            else
                add_edge!(net, i, j)
                current_stats .+= delta
            end
        end

        # Record sample after burn-in, with thinning
        if step > burnin && (step - burnin) % interval == 0
            sample_idx += 1
            samples[sample_idx, :] = current_stats
            collect_networks && push!(networks, _copy_network(net))
        end
    end

    return samples, networks
end

"""
    simulate_ergm(result::ERGMResult; n_sim::Int=1, burnin::Int=10000,
                  interval::Int=1000, rng::AbstractRNG=Random.default_rng(),
                  n_chains::Int=min(n_sim, 4)) -> Vector{Network}

Simulate networks from a fitted ERGM (at `result.coefficients`).

# Arguments
- `result::ERGMResult`: Fitted ERGM result
- `n_sim::Int=1`: Number of networks to simulate
- `burnin::Int=10000`: MCMC burn-in steps (per chain)
- `interval::Int=1000`: Steps between samples
- `rng::AbstractRNG`: Source of all random draws (reproducible seeding)
- `n_chains::Int`: Number of independent chains (see [`sample_networks`](@ref))

# Returns
- Vector of simulated Network objects
"""
function simulate_ergm(result::ERGMResult{T};
                       n_sim::Int=1,
                       burnin::Int=10000,
                       interval::Int=1000,
                       rng::AbstractRNG=Random.default_rng(),
                       n_chains::Int=min(n_sim, 4)) where T
    return sample_networks(result.model, result.coefficients;
                           n_sim=n_sim, burnin=burnin, interval=interval,
                           rng=rng, n_chains=n_chains)
end

"""
    sample_networks(model::ERGMModel, θ::Vector{Float64};
                    n_sim::Int=1, burnin::Int=10000, interval::Int=1000,
                    start_net::Union{Nothing,Network}=nothing,
                    rng::AbstractRNG=Random.default_rng(),
                    n_chains::Int=min(n_sim, 4)) -> Vector{Network}

Sample networks from an ERGM specification.

The `n_sim` draws are split over `n_chains` independent MH chains run in
parallel (`Threads.@spawn`), each burned in separately and seeded
deterministically from `rng` — so for a fixed `rng` state and `n_chains`
the result is identical regardless of the number of threads. `n_chains`
deliberately does **not** default to `Threads.nthreads()`, precisely to
keep results thread-count-independent.

# Arguments
- `model::ERGMModel`: ERGM model specification
- `θ::Vector{Float64}`: Model coefficients
- `n_sim::Int=1`: Number of networks to sample
- `burnin::Int=10000`: MCMC burn-in steps (per chain)
- `interval::Int=1000`: Steps between samples
- `start_net`: Starting network for every chain (default: an independent
  Bernoulli(density) random network per chain, with the observed network's
  attributes)
- `rng::AbstractRNG`: Source of all random draws
- `n_chains::Int`: Number of independent chains
"""
function sample_networks(model::ERGMModel{T}, θ::Vector{Float64};
                         n_sim::Int=1,
                         burnin::Int=10000,
                         interval::Int=1000,
                         start_net::Union{Nothing,Network}=nothing,
                         rng::AbstractRNG=Random.default_rng(),
                         n_chains::Int=min(n_sim, 4)) where T
    n_sim <= 0 && return Network{T}[]
    n_chains = clamp(n_chains, 1, n_sim)

    # Per-chain sample counts and deterministic per-chain seeds drawn in
    # order from the caller's rng (thread-count-independent)
    counts = fill(n_sim ÷ n_chains, n_chains)
    for c in 1:(n_sim % n_chains)
        counts[c] += 1
    end
    seeds = rand(rng, UInt64, n_chains)

    terms = model.formula.terms
    chain_nets = Vector{Vector{<:Network}}(undef, n_chains)
    @sync for c in 1:n_chains
        Threads.@spawn begin
            chain_rng = Random.Xoshiro(seeds[c])
            start = isnothing(start_net) ?
                _random_network(model.network; rng=chain_rng) :
                _copy_network(start_net)
            _, nets = _mh_run!(chain_rng, start, terms, θ, model.directed,
                               counts[c], burnin, interval, true)
            chain_nets[c] = nets
        end
    end

    networks = Network{T}[]
    sizehint!(networks, n_sim)
    for c in 1:n_chains
        append!(networks, chain_nets[c])
    end
    return networks
end

"""
    _random_network(net; density=network_density(net),
                    rng=Random.default_rng()) -> Network

Create a random MCMC starting network from an observed network: an
attribute-preserving copy of `net` whose edge set is replaced by independent
Bernoulli(`density`) draws. Vertex/network attributes (and the `directed`,
`bipartite`, and `loops` settings) are inherited from `net`, so
attribute-based terms evaluate against the same covariates as on the
observed network.

Dyads masked as missing keep their observed face value instead of being
randomized: the sampler conditions on them, so every chain must start from
the same fixed state at those dyads.
"""
function _random_network(net::Network{T};
                         density::Float64=network_density(net),
                         rng::AbstractRNG=Random.default_rng()) where T
    start = copy(net)

    # Replace the copied edge set with random edges (masked dyads keep
    # their face value)
    for e in collect(edges(start))
        is_missing_dyad(start, src(e), dst(e)) && continue
        rem_edge!(start, src(e), dst(e))
    end

    n = Int(nv(start))
    directed = is_directed(start)
    for i in 1:n
        j_range = directed ? (1:n) : ((i+1):n)
        for j in j_range
            i == j && continue
            is_missing_dyad(start, i, j) && continue
            if rand(rng) < density
                add_edge!(start, i, j)
            end
        end
    end

    return start
end
