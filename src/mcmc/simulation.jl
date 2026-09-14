"""
ERGM network simulation.

All sampling is built on a single Metropolis toggle kernel,
[`mh_toggle!`](@ref) — the accept/reject arithmetic, burn-in and thinning,
with the proposal, the change statistics and the state mutation supplied as
callables — adapted to binary networks by `_mh_run!`, exposed publicly as
[`mh_sample`](@ref) (one parameterized chain) and wrapped by
[`sample_networks`](@ref) / [`simulate_ergm`](@ref) (multi-network
simulation, parallelized over independent chains). The ERGM variants adopt
the same kernel with their own move types (see the `mh_toggle!` docstring).

Every sampler's `burnin`/`interval` default to `nothing`, resolved by ONE
rule, [`_mcmc_defaults`](@ref) (`20 × n_dyads` / `max(100, n_dyads ÷ 10)`),
which `mcmle` and the MPLE bootstrap use as well.

Every function takes an `rng::AbstractRNG` keyword; all random draws flow
from it, so runs with the same seed are exactly reproducible. Parallel
functions seed one RNG per chain deterministically from the caller's `rng`,
so results are also independent of the number of threads.
"""

"""
    mh_sample(model::ERGMModel, θ::Vector{Float64};
              n_samples::Int=100, burnin=nothing, interval=nothing,
              rng::AbstractRNG=Random.default_rng(),
              start_net::Union{Nothing,Network}=nothing,
              return_networks::Bool=false, missing::Symbol=:error,
              toggleable::Symbol=:free)
        -> (stats::Matrix{Float64}, networks::Union{Nothing,Vector{<:Network}})

Run one Metropolis–Hastings edge-toggle chain of the ERGM `exp(θ'g(y))`
defined by `model`'s terms, and return the sampled sufficient statistics
(and optionally the sampled networks).

This is the public single-chain sampling primitive that the estimation
routines (and downstream packages such as ERGMEgo.jl) build on. Each MH
step proposes toggling a uniformly random dyad and accepts with probability
`min(1, exp(±θ'Δ))`, where `Δ` is the add-direction change statistic
vector of the dyad.

Which dyads the chain may toggle is `toggleable`:

- `:free` (default) — every dyad that is not masked as missing
  (`Networks.set_missing_dyad!`). A masked dyad is then *frozen* at its
  stored face value (edge present/absent as recorded), never proposed, so
  the chain conditions on it. Reinterpreting an unobserved tie as an
  observed one is never silent, so with `:free` a masked network is
  rejected unless the caller opts in with `missing=:condition_on_face`.
  Throws `ArgumentError` if every dyad is masked.
- `:all` — every dyad, masked ones included: the chain samples the
  *unconditional* model `exp(θ'g(y))` over the whole dyad set, as the free
  chain of the missing-data MCMLE ([`mcmle`](@ref) with `missing=:mle`)
  does. No unobserved value is read as an observation (the face values only
  seed the starting state, which burn-in washes out), so no `missing=`
  opt-in is needed; `missing` must be left at `:error`.
- `:masked` — only the masked dyads, every observed dyad held fixed at its
  observed value: the *constrained* chain of the missing-data MCMLE, whose
  mean estimates `E[g(Y) | Y_obs]` (Handcock & Gile 2010). Requires
  `n_missing_dyads(model.network) > 0`; `missing` must be left at `:error`.

# Arguments
- `model::ERGMModel`: Model specification (terms + network template). The
  chain runs on a copy; `model.network` is never mutated.
- `θ::Vector{Float64}`: Natural-parameter vector, one entry per term.

# Keywords
- `n_samples::Int=100`: Number of recorded samples.
- `burnin::Int`: MH steps discarded before the first sample. Defaults to
  `20 × n_dyads` (the shared dyad-scaled rule, [`_mcmc_defaults`](@ref)),
  where `n_dyads` is the number of dyads the chain may toggle.
- `interval::Int`: MH steps between recorded samples (thinning). Defaults
  to `max(100, n_dyads ÷ 10)`.
- `rng::AbstractRNG=Random.default_rng()`: Source of all random draws;
  same rng state ⇒ identical output.
- `start_net=nothing`: Starting network. Defaults to a copy of the observed
  `model.network` (attributes are preserved either way).
- `return_networks::Bool=false`: Also collect a copy of the network at
  every sampling point.
- `missing::Symbol=:error`: `:error` rejects a network with masked dyads
  (under `toggleable=:free`); `:condition_on_face` explicitly holds them at
  their face value.
- `toggleable::Symbol=:free`: `:free`, `:all` or `:masked` — see above.

# Returns
A NamedTuple `(stats, networks)`:
- `stats`: `n_samples × p` matrix; row `k` holds `g(y⁽ᵏ⁾)`.
- `networks`: `Vector` of the sampled networks if `return_networks=true`,
  otherwise `nothing`.

# Example
```julia
using ERGM, Random, Statistics
net = load_dataset(:florentine_marriage)
model = ERGMModel(ERGMFormula([Edges(), Triangle()]), net)
out = mh_sample(model, [-1.5, 0.3]; n_samples=500, rng=Xoshiro(1))
size(out.stats)            # (500, 2)
mean(out.stats, dims=1)    # E[g] estimate at θ

# The two chains of the missing-data MCMLE on a masked copy
masked = copy(net); set_missing_dyad!(masked, 3, 4); set_missing_dyad!(masked, 1, 9)
mm = ERGMModel(ERGMFormula([Edges(), Triangle()]), masked)
free = mh_sample(mm, [-1.5, 0.3]; n_samples=200, toggleable=:all, rng=Xoshiro(2))
cons = mh_sample(mm, [-1.5, 0.3]; n_samples=200, toggleable=:masked, rng=Xoshiro(3),
                 return_networks=true)
all(has_edge(s, 2, 6) for s in cons.networks)   # true: observed ties never move
```
"""
function mh_sample(model::ERGMModel{T,D}, θ::Vector{Float64};
                   n_samples::Int=100,
                   burnin::Union{Nothing,Int}=nothing,
                   interval::Union{Nothing,Int}=nothing,
                   rng::AbstractRNG=Random.default_rng(),
                   start_net::Union{Nothing,Network}=nothing,
                   return_networks::Bool=false,
                   missing::Symbol=:error,
                   toggleable::Symbol=:free) where {T,D}
    length(θ) == length(model.formula.terms) ||
        throw(ArgumentError("length(θ) = $(length(θ)) does not match the " *
                            "number of model terms ($(length(model.formula.terms)))"))
    src_net = isnothing(start_net) ? model.network : start_net
    _check_toggleable(src_net, toggleable, missing; context="mh_sample")
    burnin, interval = _resolve_mcmc_controls(model, burnin, interval;
                                              toggleable=toggleable)
    net = _copy_network(src_net)
    stats, networks = _mh_run!(rng, net, model.formula.terms, θ,
                               n_samples, burnin, interval, return_networks,
                               toggleable)
    return (stats=stats, networks=return_networks ? networks : nothing)
end

missing_policies(::typeof(mh_sample)) = _MISSING_POLICIES

const _TOGGLEABLE = (:free, :all, :masked)

# Validate the `toggleable` dyad set of a chain against the network it runs
# on. `:free` is the ordinary sampler and goes through the missing-data
# guard; `:all` and `:masked` are the two chains of the missing-data MCMLE,
# which read no unobserved value as an observation and therefore take no
# `missing=` opt-in (an opt-in would be meaningless, so it is refused).
function _check_toggleable(net, toggleable::Symbol, missing::Symbol;
                           context::AbstractString)
    toggleable in _TOGGLEABLE || throw(ArgumentError(
        "invalid toggleable dyad set $(repr(toggleable)) for $context; expected " *
        ":free (every unmasked dyad), :all (every dyad, masked ones included) " *
        "or :masked (only the masked dyads, observed dyads held fixed)"))
    if toggleable === :free
        _guard_missing(net, missing; context=context)
    else
        missing === :error || throw(ArgumentError(
            "$context: `missing=$(repr(missing))` is only meaningful for " *
            "toggleable=:free; the $(repr(toggleable)) chain of the missing-data " *
            "MCMLE reads no masked dyad at face value and takes no opt-in"))
        toggleable === :masked && n_missing_dyads(net) == 0 && throw(ArgumentError(
            "$context: toggleable=:masked needs at least one dyad masked as " *
            "missing (`set_missing_dyad!`), but the network has none — the " *
            "constrained chain would have nothing to toggle"))
    end
    return nothing
end

"""
    _mcmc_defaults(model::ERGMModel) -> (burnin, interval)
    _mcmc_defaults(n_dyads::Int) -> (burnin, interval)

THE dyad-scaled rule behind every sampler default in ERGM.jl (panel 2026-09,
item 24e): `burnin = 20 × n_dyads` toggles and `interval = max(100,
n_dyads ÷ 10)` toggles between recorded draws, where `n_dyads` is the number
of free (unmasked) dyads (`ERGM._n_dyads`). A Metropolis chain that
proposes one dyad per step needs a number of steps proportional to the
number of dyads to move every dyad a bounded number of times, so a fixed
budget (the pre-0.2 `10000`/`1000`) was both far too small for a 500-node
network and wastefully large for a 16-node one.

`mcmle`, the MPLE parametric bootstrap, `mh_sample`, `sample_networks`,
`simulate_ergm` and `gof` all resolve a `burnin=nothing`/`interval=nothing`
keyword through this one function, so the budgets cannot drift apart.

# Example
```julia
using ERGM
model = ERGMModel(ERGMFormula([Edges()]), load_dataset(:florentine_marriage))
ERGM._mcmc_defaults(model)     # (burnin = 2400, interval = 100): 120 dyads
ERGM._mcmc_defaults(124750)    # (burnin = 2495000, interval = 12475): n = 500
```
"""
_mcmc_defaults(model::ERGMModel) = _mcmc_defaults(_n_dyads(model))
_mcmc_defaults(n_dyads::Int) = (burnin = 20 * n_dyads,
                                interval = max(100, n_dyads ÷ 10))

# Resolve `burnin`/`interval` keywords that default to `nothing`: an explicit
# integer is honoured as given, `nothing` becomes the dyad-scaled default —
# scaled by the number of dyads the chain may toggle (`_n_toggleable`), so
# the constrained chain of the missing-data MCMLE, which moves only the
# masked dyads, is not burned in for the whole network.
function _resolve_mcmc_controls(model::ERGMModel, burnin, interval;
                                toggleable::Symbol=:free)
    if burnin === nothing || interval === nothing
        d = _mcmc_defaults(_n_toggleable(model.network, toggleable))
        burnin = something(burnin, d.burnin)
        interval = something(interval, d.interval)
    end
    return Int(burnin), Int(interval)
end

# Number of dyads a chain with the given `toggleable` set may propose
function _n_toggleable(net::Network{T,D}, toggleable::Symbol) where {T,D}
    n = Int(nv(net))
    total = D ? n * (n - 1) : n * (n - 1) ÷ 2
    return toggleable === :all ? total :
           toggleable === :masked ? n_missing_dyads(net) :
           total - n_missing_dyads(net)
end

"""
    mh_toggle!(rng::AbstractRNG, θ::AbstractVector, delta::Vector{Float64},
               propose, change!, apply!, on_sample;
               burnin::Int, interval::Int, n_samples::Int) -> Int

The Metropolis toggle kernel every ERGM-family sampler is built on (panel
2026-09, item 28). It owns exactly three things — the accept/reject
arithmetic, the burn-in and the thinning — and knows nothing about
networks: what a *move* is, how it changes the sufficient statistics and how
it is applied to the state are supplied as callables, so the same loop
samples binary networks (ERGM.jl's `_mh_run!`), formation/dissolution
networks (TERGM), multilayer networks (ERGMMulti) and rankings (ERGMRank).

For `burnin + n_samples × interval` steps the kernel does

    move    = propose(rng)                 # a proposal, drawn from `rng`
    removal = change!(delta, move)         # fills Δ, says whether the move removes
    accept  = log(rand(rng)) < (removal ? -1 : 1) * θ'Δ
    accept && apply!(move, removal)        # mutate the state

and after burn-in calls `on_sample(k)` at every `interval`-th step, `k`
running from 1 to `n_samples`. Returns the number of samples recorded.

# Callables
- `propose(rng) -> move`: draw the next proposal, consuming only `rng`.
  For ERGM it is a free dyad `(i, j)` — never a masked one, never a loop,
  `i < j` on undirected networks. Any immutable value can be a move (a
  tuple, a struct); the kernel only passes it on.
- `change!(delta, move) -> Bool`: fill `delta` with the **add-direction**
  change statistics `g(y⁺) − g(y⁻)` of the move (the state-independent
  convention every `change_stat` method follows) and return `true` when the
  move is a *removal* (the tie is currently present), so that the kernel
  negates the log-ratio. A sampler whose moves have no removal direction —
  ERGMRank's rank swaps, whose `delta` is already the signed difference —
  returns `false` always.
- `apply!(move, removal::Bool)`: mutate the state for an accepted move
  (toggle the tie, and keep any running statistics current).
- `on_sample(k)`: record sample number `k` (write a row of a statistics
  matrix, push a copy of the network, ...).

# Keywords
- `burnin::Int`: steps discarded before the first sample (≥ 0).
- `interval::Int`: steps between recorded samples (≥ 1).
- `n_samples::Int`: number of samples to record (≥ 0).

# Contract
`θ` and `delta` must have the same length; `delta` is the caller's
workspace and is overwritten at every step. The kernel draws from `rng` in a
fixed order — the proposal first (through `propose`), then one uniform for
the acceptance test — so for the same `rng` state, `propose` and `change!`
the sampled sequence is bit-identical to ERGM.jl's pre-0.2 hand-written
loop (pinned by the "MH kernel" testset). It allocates nothing per step
when the callables do not (`@allocated` is pinned to grow by 0 bytes
between `burnin=10_000` and `burnin=20_000`): keep the move a stack value
and the closures free of reassigned captured variables.

# Adopting the kernel in a variant

*TERGM `_sample_constrained`* — moves are dyads drawn from the `free` list
(non-edges of the previous network for formation, its edges for
dissolution), the change statistics are the temporal `_tchange`, no samples
are recorded (the state after `steps` toggles is the draw):

```jl
propose = rng -> free[rand(rng, 1:length(free))]
change! = (delta, move) -> begin
    i, j = move
    for (k, t) in enumerate(terms); delta[k] = _tchange(t, net, i, j, prev); end
    has_edge(net, i, j)
end
apply! = (move, removal) -> begin
    i, j = move
    removal ? rem_edge!(net, i, j) : add_edge!(net, i, j)
    nothing
end
mh_toggle!(rng, θ, delta, propose, change!, apply!, _ -> nothing;
           burnin=steps, interval=1, n_samples=0)
```
(A sketch in TERGM's own names — `free`, `net`, `prev`, `_tchange` — not a
standalone program, hence the `jl` fence; the runnable example is below.)

*ERGMMulti* — a move is `(layer, i, j)`; `change!` calls
`change_stat_layer(t, current, l, i, j)` and returns
`has_edge(current.layers[l], i, j)`; `apply!` toggles that layer's edge;
`on_sample` pushes a copy of the multilayer network. This is real code
now — `ERGMMulti._multi_mh_run`, pinned bit-identical to a hand-written
`mh_toggle!` call in ERGMMulti's tests.

*ERGMRank* — a move is a swap `(ego, j, k)`; `change!` fills `delta` with
`_swap_delta(terms, current, ego, j, k)` and returns `false` (a swap has no
removal direction); `apply!` is `swap_ranks!(current, ego, j, k)`;
`on_sample` pushes `copy(current)`.

*ERGMCount* does **not** adopt it: its sampler is a Gibbs sweep that
redraws each dyad's count from its full conditional, not a Metropolis
toggle.

# Example
A two-state toy chain — a single "tie" with statistic `g = y`, sampled at
`θ = log(3)`, so `P(y = 1) = 3/4`:

```julia
using ERGM, Random, Statistics
state = Ref(false)
draws = Float64[]
mh_toggle!(Xoshiro(1), [log(3.0)], [0.0],
           rng -> 1,                              # the only move
           (delta, move) -> (delta[1] = 1.0; state[]),
           (move, removal) -> (state[] = !removal),
           k -> push!(draws, state[]);
           burnin=100, interval=1, n_samples=20_000)
abs(mean(draws) - 0.75) < 0.02    # true
```
"""
function mh_toggle!(rng::AbstractRNG, θ::AbstractVector{<:Real},
                    delta::Vector{Float64},
                    propose::P, change!::C, apply!::A, on_sample::S;
                    burnin::Int, interval::Int, n_samples::Int) where {P, C, A, S}
    burnin >= 0 || throw(ArgumentError("mh_toggle!: burnin must be ≥ 0 (got $burnin)"))
    interval >= 1 || throw(ArgumentError("mh_toggle!: interval must be ≥ 1 (got $interval)"))
    n_samples >= 0 || throw(ArgumentError("mh_toggle!: n_samples must be ≥ 0 (got $n_samples)"))
    length(delta) == length(θ) || throw(ArgumentError(
        "mh_toggle!: delta has length $(length(delta)) but θ has length $(length(θ))"))

    total_steps = burnin + n_samples * interval
    k = 0
    for step in 1:total_steps
        move = propose(rng)
        # Add-direction change statistics; the MH log-ratio is θ'Δ for an
        # addition and −θ'Δ for a removal
        removal = change!(delta, move)::Bool
        log_accept = dot(θ, delta)
        removal && (log_accept = -log_accept)

        if log(rand(rng)) < log_accept
            apply!(move, removal)
        end

        # Record sample after burn-in, with thinning
        if step > burnin && (step - burnin) % interval == 0
            k += 1
            on_sample(k)
        end
    end
    return k
end

# The binary-network adapter over `mh_toggle!`: mutates `net` in place and
# returns the sampled statistics matrix and (when `collect_networks`)
# network copies at each sampling point. A function barrier: `terms` is
# passed concretely so the closures and the kernel loop are fully typed;
# the directedness is the network's type parameter `D`.
#
# `toggleable` selects the dyads the chain may propose, purely through the
# proposal closure (panel 2026-09, item 32): `:free` — every unmasked dyad
# (masked dyads frozen at their face value, so the chain conditions on
# them; the pre-0.2 behaviour, bit-identical); `:all` — every dyad, the
# unconditional model; `:masked` — only the masked dyads, drawn uniformly
# from their list, the observed dyads held fixed (the constrained chain of
# the missing-data MCMLE).
function _mh_run!(rng::AbstractRNG, net::Network{T,D}, terms::TermSet,
                  θ::Vector{Float64},
                  n_samples::Int, burnin::Int, interval::Int,
                  collect_networks::Bool, toggleable::Symbol=:free) where {T,D}
    toggleable in _TOGGLEABLE || throw(ArgumentError(
        "_mh_run!: toggleable must be :free, :all or :masked (got $(repr(toggleable)))"))
    n = Int(nv(net))
    _n_toggleable(net, toggleable) > 0 || throw(ArgumentError(
        toggleable === :masked ?
        "no dyad of the network is masked as missing; the constrained " *
        "(toggleable=:masked) MH chain has nothing to toggle" :
        "every dyad of the network is masked as missing; the MH sampler has " *
        "no free dyads to toggle"))

    if toggleable === :masked
        # The masked dyads, canonically ordered (i < j on undirected
        # networks), as a plain vector to draw from uniformly
        masked = Tuple{Int,Int}[(Int(i), Int(j)) for (i, j) in missing_dyads(net)]
        propose = function (rng)
            @inbounds return masked[rand(rng, 1:length(masked))]
        end
        return _mh_run_with_proposal!(rng, net, terms, θ, n_samples, burnin,
                                      interval, collect_networks, propose)
    elseif toggleable === :all
        # Any dyad (never a loop; i < j on undirected networks)
        propose = function (rng)
            i = rand(rng, 1:n)
            j = rand(rng, 1:n)
            while i == j || (!D && j < i)
                i = rand(rng, 1:n)
                j = rand(rng, 1:n)
            end
            return (i, j)
        end
        return _mh_run_with_proposal!(rng, net, terms, θ, n_samples, burnin,
                                      interval, collect_networks, propose)
    end

    # `:free`: a uniformly random free dyad (never a loop, never a masked
    # dyad; i < j on undirected networks)
    propose = function (rng)
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        while i == j || (!D && j < i) || is_missing_dyad(net, i, j)
            i = rand(rng, 1:n)
            j = rand(rng, 1:n)
        end
        return (i, j)
    end
    return _mh_run_with_proposal!(rng, net, terms, θ, n_samples, burnin,
                                  interval, collect_networks, propose)
end

# The rest of the adapter, shared by the three proposal closures: change
# statistics, state mutation and sample recording over `mh_toggle!`.
function _mh_run_with_proposal!(rng::AbstractRNG, net::Network{T,D}, terms::TermSet,
                                θ::Vector{Float64}, n_samples::Int, burnin::Int,
                                interval::Int, collect_networks::Bool,
                                propose::P) where {T,D,P}
    p = length(terms)
    samples = Matrix{Float64}(undef, n_samples, p)
    networks = Vector{typeof(net)}()
    collect_networks && sizehint!(networks, n_samples)

    current_stats = compute_all(terms, net)
    delta = Vector{Float64}(undef, p)

    change! = function (delta, move)
        i, j = move
        change_stat_all!(delta, terms, net, i, j)
        return has_edge(net, i, j)
    end
    apply! = function (move, removal)
        i, j = move
        if removal
            rem_edge!(net, i, j)
            current_stats .-= delta
        else
            add_edge!(net, i, j)
            current_stats .+= delta
        end
        return nothing
    end
    on_sample = function (k)
        @inbounds for c in 1:p
            samples[k, c] = current_stats[c]
        end
        collect_networks && push!(networks, _copy_network(net))
        return nothing
    end

    mh_toggle!(rng, θ, delta, propose, change!, apply!, on_sample;
               burnin=burnin, interval=interval, n_samples=n_samples)

    return samples, networks
end

"""
    simulate_ergm(result::ERGMResult; n_sim::Int=1, burnin=nothing,
                  interval=nothing, rng::AbstractRNG=Random.default_rng(),
                  n_chains::Int=min(n_sim, 4),
                  missing::Symbol=:error) -> Vector{Network}

Simulate networks from a fitted ERGM (at `result.coefficients`).

If the fitted network has dyads masked as missing, the simulation cannot
reinterpret them: the sampler would silently freeze each unobserved tie at
its stored face value and report it as a simulated tie. That requires the
explicit `missing=:condition_on_face` opt-in (a warning is emitted); the
default `missing=:error` refuses. This holds even for an MPLE fit, whose
*point estimate* excluded the masked dyads: simulation is a separate act.

# Arguments
- `result::ERGMResult`: Fitted ERGM result
- `n_sim::Int=1`: Number of networks to simulate
- `burnin::Int`: MCMC burn-in steps (per chain). Defaults to `20 × n_dyads`
  (the shared dyad-scaled rule, [`_mcmc_defaults`](@ref))
- `interval::Int`: Steps between samples. Defaults to
  `max(100, n_dyads ÷ 10)`
- `rng::AbstractRNG`: Source of all random draws (reproducible seeding)
- `n_chains::Int`: Number of independent chains (see [`sample_networks`](@ref))
- `missing::Symbol=:error`: Missing-dyad policy (`:error` or
  `:condition_on_face`)

# Returns
- Vector of simulated Network objects

# Example
```julia
using ERGM, Random
net = load_dataset(:florentine_marriage)
fit = fit_ergm(net, [Edges(), NodeCov(:wealth)])
sims = simulate_ergm(fit; n_sim=10, rng=Xoshiro(1))
length(sims)                       # 10
all(nv(s) == 16 for s in sims)     # true
```
"""
function simulate_ergm(result::ERGMResult{T,D};
                       n_sim::Int=1,
                       burnin::Union{Nothing,Int}=nothing,
                       interval::Union{Nothing,Int}=nothing,
                       rng::AbstractRNG=Random.default_rng(),
                       n_chains::Int=min(n_sim, 4),
                       missing::Symbol=:error) where {T,D}
    method = _guard_missing(result.model.network, missing;
                            context="simulate_ergm")
    method === :condition_on_face &&
        _warn_condition_on_face(result.model.network, "simulation")
    return sample_networks(result.model, result.coefficients;
                           n_sim=n_sim, burnin=burnin, interval=interval,
                           rng=rng, n_chains=n_chains, missing=missing)
end

missing_policies(::typeof(simulate_ergm)) = _MISSING_POLICIES

"""
    sample_networks(model::ERGMModel, θ::Vector{Float64};
                    n_sim::Int=1, burnin=nothing, interval=nothing,
                    start_net::Union{Nothing,Network}=nothing,
                    rng::AbstractRNG=Random.default_rng(),
                    n_chains::Int=min(n_sim, 4),
                    missing::Symbol=:error) -> Vector{Network{T,D}}

Sample networks from an ERGM specification. The returned vector is concretely
typed (`Vector{Network{T,D}}`, the model's own network type).

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
- `burnin::Int`: MCMC burn-in steps (per chain). Defaults to `20 × n_dyads`
  (the shared dyad-scaled rule, [`_mcmc_defaults`](@ref))
- `interval::Int`: Steps between samples. Defaults to
  `max(100, n_dyads ÷ 10)`
- `start_net`: Starting network for every chain (default: an independent
  Bernoulli(density) random network per chain, with the observed network's
  attributes)
- `rng::AbstractRNG`: Source of all random draws
- `n_chains::Int`: Number of independent chains
- `missing::Symbol=:error`: Missing-dyad policy. `:error` refuses a network
  with masked dyads; `:condition_on_face` explicitly freezes each masked
  dyad at its stored face value in every chain (`_random_network` starting
  states preserve them too).

# Example
```julia
using ERGM, Random
net = load_dataset(:florentine_marriage)
model = ERGMModel(ERGMFormula([Edges(), Triangle()]), net)
sims = sample_networks(model, [-1.6, 0.2]; n_sim=8, n_chains=2, rng=Xoshiro(1))
length(sims)                        # 8
sims isa Vector{Network{Int,false}} # true
```
"""
function sample_networks(model::ERGMModel{T,D}, θ::Vector{Float64};
                         n_sim::Int=1,
                         burnin::Union{Nothing,Int}=nothing,
                         interval::Union{Nothing,Int}=nothing,
                         start_net::Union{Nothing,Network}=nothing,
                         rng::AbstractRNG=Random.default_rng(),
                         n_chains::Int=min(n_sim, 4),
                         missing::Symbol=:error) where {T,D}
    _guard_missing(isnothing(start_net) ? model.network : start_net, missing;
                   context="sample_networks")
    burnin, interval = _resolve_mcmc_controls(model, burnin, interval)
    n_sim <= 0 && return Network{T,D}[]
    n_chains = clamp(n_chains, 1, n_sim)

    # Per-chain sample counts and deterministic per-chain seeds drawn in
    # order from the caller's rng (thread-count-independent)
    counts = fill(n_sim ÷ n_chains, n_chains)
    for c in 1:(n_sim % n_chains)
        counts[c] += 1
    end
    seeds = rand(rng, UInt64, n_chains)

    terms = model.formula.terms
    chain_nets = Vector{Vector{Network{T,D}}}(undef, n_chains)
    @sync for c in 1:n_chains
        Threads.@spawn begin
            chain_rng = Random.Xoshiro(seeds[c])
            # A chain runs on the model's own network type: a user-supplied
            # start network of another directedness would not be that model.
            start = isnothing(start_net) ?
                _random_network(model.network; rng=chain_rng) :
                convert(Network{T,D}, _copy_network(start_net))
            _, nets = _mh_run!(chain_rng, start, terms, θ,
                               counts[c], burnin, interval, true)
            chain_nets[c] = nets
        end
    end

    networks = Network{T,D}[]
    sizehint!(networks, n_sim)
    for c in 1:n_chains
        append!(networks, chain_nets[c])
    end
    return networks
end

missing_policies(::typeof(sample_networks)) = _MISSING_POLICIES

"""
    _random_network(net; density=network_density(net; missing=:face),
                    rng=Random.default_rng(), randomize_masked=false) -> Network

Create a random MCMC starting network from an observed network: an
attribute-preserving copy of `net` whose edge set is replaced by independent
Bernoulli(`density`) draws. Vertex/network attributes (and the `directed`,
`bipartite`, and `loops` settings) are inherited from `net`, so
attribute-based terms evaluate against the same covariates as on the
observed network.

By default dyads masked as missing keep their observed face value instead
of being randomized: a `toggleable=:free` chain conditions on them, so every
chain must start from the same fixed state at those dyads. With
`randomize_masked=true` the masked dyads are Bernoulli draws like every
other — the starting state of a `toggleable=:all` chain, which samples the
unconditional model and must not be seeded with an unobserved value it
could otherwise carry through a short burn-in. (The mask itself is kept
either way.) The default starting density is read at face value
(`missing=:face`): `network_density` refuses a masked network by default,
and a starting density is a tuning choice that burn-in washes out, not a
statistic.
"""
function _random_network(net::Network{T,D};
                         density::Float64=network_density(net; missing=:face),
                         rng::AbstractRNG=Random.default_rng(),
                         randomize_masked::Bool=false) where {T,D}
    start = copy(net)

    # Replace the copied edge set with random edges (masked dyads keep
    # their face value unless asked otherwise)
    for e in collect(edges(start))
        !randomize_masked && is_missing_dyad(start, src(e), dst(e)) && continue
        rem_edge!(start, src(e), dst(e))
    end

    n = Int(nv(start))
    directed = is_directed(start)
    for i in 1:n
        j_range = directed ? (1:n) : ((i+1):n)
        for j in j_range
            i == j && continue
            !randomize_masked && is_missing_dyad(start, i, j) && continue
            if rand(rng) < density
                add_edge!(start, i, j)
            end
        end
    end

    return start
end
