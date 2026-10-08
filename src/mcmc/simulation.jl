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
rule, [`mcmc_defaults`](@ref ERGM.Extension.mcmc_defaults) (`20 × n_dyads` / `max(100, n_dyads ÷ 10)`),
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
              toggleable::Symbol=:free, proposal::Symbol=:tnt)
        -> (stats::Matrix{Float64}, networks::Union{Nothing,Vector{<:Network}})

Run one Metropolis–Hastings edge-toggle chain of the ERGM `exp(θ'g(y))`
defined by `model`'s terms, and return the sampled sufficient statistics
(and optionally the sampled networks).

This is the public single-chain sampling primitive that the estimation
routines (and downstream packages such as ERGMEgo.jl) build on. Each MH
step proposes toggling one dyad and accepts with probability
`min(1, exp(±θ'Δ) · H)`, where `Δ` is the add-direction change statistic
vector of the dyad and `H` the proposal's Hastings ratio.

**The proposal** (`proposal=`) is statnet's default **tie/no-tie** (`:tnt`):
with probability ½ a uniformly chosen existing tie is proposed for removal,
otherwise a uniformly chosen dyad is toggled, and the acceptance ratio
carries the exact Hastings correction (R ergm's `MH_TNT`). On sparse
networks — the usual case — a uniformly random dyad is nearly always an
absent tie whose addition the model rejects, so TNT mixes several times
faster per toggle (about 7× in effective sample size on faux.mesa.high).
`proposal=:random` is the uniformly random dyad toggle (symmetric, no
correction), the pre-0.2 sampler, kept for reproducing earlier results.
Both are exact: their stationary distribution is `exp(θ'g(y))/Z(θ)`
(verified against enumeration in the test suite). The `:masked` chain
always proposes uniformly among the masked dyads.

Which dyads the chain may toggle is `toggleable`:

- `:free` (default) — every dyad that is not masked as missing
  (`NetworkCore.set_missing_dyad!`). A masked dyad is then *frozen* at its
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
  `20 × n_dyads` (the shared dyad-scaled rule, [`mcmc_defaults`](@ref ERGM.Extension.mcmc_defaults)),
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
- `proposal::Symbol=:tnt`: `:tnt` (tie/no-tie) or `:random` — see above.

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
                   toggleable::Symbol=:free,
                   proposal::Symbol=:tnt) where {T,D}
    length(θ) == length(model.formula.terms) ||
        throw(ArgumentError("length(θ) = $(length(θ)) does not match the " *
                            "number of model terms ($(length(model.formula.terms)))"))
    src_net = isnothing(start_net) ? model.network : start_net
    _check_toggleable(src_net, toggleable, missing; context="mh_sample")
    burnin, interval = _resolve_mcmc_controls(model, burnin, interval;
                                              toggleable=toggleable)
    net = _copy_network(src_net)
    _check_proposal(proposal; context="mh_sample")
    stats, networks = _mh_run!(rng, net, model.formula.terms, θ,
                               n_samples, burnin, interval, return_networks,
                               toggleable; proposal=proposal)
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
    mcmc_defaults(model::ERGMModel) -> (burnin, interval)
    mcmc_defaults(n_dyads::Int) -> (burnin, interval)

THE dyad-scaled rule behind every sampler default in ERGM.jl: `burnin = 20 × n_dyads` toggles and `interval = max(100,
n_dyads ÷ 10)` toggles between recorded draws, where `n_dyads` is the number
of free (unmasked) dyads ([`n_observed_dyads`](@ref ERGM.Extension.n_observed_dyads)). A Metropolis chain that
proposes one dyad per step needs a number of steps proportional to the
number of dyads to move every dyad a bounded number of times, so a fixed
budget (the pre-0.2 `10000`/`1000`) was both far too small for a 500-node
network and wastefully large for a 16-node one.

`mcmle`, the MPLE parametric bootstrap, `mh_sample`, `sample_networks`,
`simulate_ergm` and `gof` all resolve a `burnin=nothing`/`interval=nothing`
keyword through this one function, so the budgets cannot drift apart. Part
of [`ERGM.Extension`](@ref): every variant's sampler resolves its defaults
through it too (ERGMCount converts the toggles to sweeps, ERGMRank counts
swaps), so the family burns in alike.

# Example
```julia
using ERGM
model = ERGMModel(ERGMFormula([Edges()]), load_dataset(:florentine_marriage))
ERGM.Extension.mcmc_defaults(model)     # (burnin = 2400, interval = 100): 120 dyads
ERGM.Extension.mcmc_defaults(124750)    # (burnin = 2495000, interval = 12475): n = 500
```
"""
mcmc_defaults(model::ERGMModel) = mcmc_defaults(n_observed_dyads(model))
mcmc_defaults(n_dyads::Int) = (burnin = 20 * n_dyads,
                               interval = max(100, n_dyads ÷ 10))

# Resolve `burnin`/`interval` keywords that default to `nothing`: an explicit
# integer is honoured as given, `nothing` becomes the dyad-scaled default —
# scaled by the number of dyads the chain may toggle (`_n_toggleable`), so
# the constrained chain of the missing-data MCMLE, which moves only the
# masked dyads, is not burned in for the whole network.
function _resolve_mcmc_controls(model::ERGMModel, burnin, interval;
                                toggleable::Symbol=:free)
    # Every sampler entry point (mcmle, the MPLE bootstrap, mh_sample,
    # sample_networks/simulate_ergm, gof) resolves its controls here, so this
    # is also where the one data requirement of the sampler is checked
    _refuse_live_edge_attributes(model)
    if burnin === nothing || interval === nothing
        d = mcmc_defaults(_n_toggleable(model.network, toggleable))
        burnin = something(burnin, d.burnin)
        interval = something(interval, d.interval)
    end
    return Int(burnin), Int(interval)
end

# The samplers toggle ties with `rem_edge!`/`add_edge!`, and `rem_edge!`
# deletes the tie's edge attributes: a term that reads an edge attribute LIVE
# from the network loses it the first time the chain removes the tie, so the
# chain would silently target a different model (a weighted-edges term decays
# to its default weight). Built-in terms never read edge attributes (EdgeCov
# takes a matrix); a term from another package is probed: its statistic and
# its change statistics at up to 25 ties and 5 non-ties are compared on the
# network and on a copy with every edge attribute removed. Any difference is
# an ArgumentError naming the remedy — snapshot the attribute at model
# construction, as ERGMUserterms' `WeightedEdges` does. The MPLE, which never
# toggles a tie, is unaffected.
function _refuse_live_edge_attributes(model::ERGMModel)
    net = model.network
    isempty(list_edge_attributes(net)) && return nothing
    terms = model.formula.terms.terms
    foreign = [k for k in eachindex(terms)
               if parentmodule(typeof(_probe_term(terms[k]))) !== (@__MODULE__)]
    isempty(foreign) && return nothing
    bare = copy(net)
    for attr in list_edge_attributes(bare)
        delete_edge_attribute!(bare, attr)
    end
    ties = [(Int(src(e)), Int(dst(e))) for e in Iterators.take(edges(net), 25)]
    n = Int(nv(net))
    nonties = Tuple{Int,Int}[]
    for i in 1:n, j in 1:n
        length(nonties) >= 5 && break
        (i == j || (!is_directed(net) && i > j) || has_edge(net, i, j)) && continue
        push!(nonties, (i, j))
    end
    same(a, b) = a == b || isapprox(a, b; rtol=1e-12, atol=1e-12)
    for k in foreign
        t = terms[k]
        live = !same(compute(t, net), compute(t, bare)) ||
               any(((i, j),) -> !same(change_stat(t, net, i, j), change_stat(t, bare, i, j)),
                   [ties; nonties])
        live || continue
        label = model.formula.terms.names[k]
        throw(ArgumentError(
            "term '$label' ($(nameof(typeof(_probe_term(t))))) reads an edge attribute " *
            "of the network live: its statistic changes when the network's edge " *
            "attributes are removed. ERGM.jl's samplers toggle ties with " *
            "rem_edge!/add_edge!, which delete a tie's attributes, so MCMLE, " *
            "simulation, gof and the MPLE bootstrap would silently target a " *
            "different model. Snapshot the attribute when the model is built: " *
            "define `ERGM.Extension.materialize(term, net)` to return a twin holding it as " *
            "a matrix (as ERGMUserterms' WeightedEdges does), or pass the dyadic " *
            "covariate as a matrix (`EdgeCov(W)`). `mple` (method=:mple), which " *
            "never toggles a tie, is unaffected."))
    end
    return nothing
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
               burnin::Int, interval::Int, n_samples::Int,
               hastings=nothing) -> Int

The Metropolis toggle kernel every ERGM-family sampler is built on. It owns exactly three things — the accept/reject
arithmetic, the burn-in and the thinning — and knows nothing about
networks: what a *move* is, how it changes the sufficient statistics and how
it is applied to the state are supplied as callables, so the same loop
samples binary networks (ERGM.jl's `_mh_run!`), formation/dissolution
networks (TERGM), multilayer networks (ERGMMulti) and rankings (ERGMRank).

For `burnin + n_samples × interval` steps the kernel does

    move    = propose(rng)                 # a proposal, drawn from `rng`
    removal = change!(delta, move)         # fills Δ, says whether the move removes
    accept  = log(rand(rng)) < (removal ? -1 : 1) * θ'Δ + hastings(move, removal)
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
- `hastings(move, removal) -> Float64` (keyword, optional): the log
  Hastings ratio `log q(y | y*) − log q(y* | y)` of an *asymmetric*
  proposal, evaluated on the current state before the move is applied.
  Omitted (`nothing`, the default) for a symmetric proposal — a uniformly
  random dyad, a rank swap — and then compiled out entirely. ERGM.jl's
  tie/no-tie (TNT) proposal passes its correction here (see
  [`mh_sample`](@ref)).

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
                    burnin::Int, interval::Int, n_samples::Int,
                    hastings::H=nothing) where {P, C, A, S, H}
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
        # Hastings correction of an asymmetric proposal (TNT), read from the
        # state BEFORE the move is applied; compiled out when absent, so a
        # symmetric proposal's chain is bit-identical to the pre-0.2 loop
        if !(hastings === nothing)
            log_accept += hastings(move, removal)::Float64
        end

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

# Every threaded loop (the chains of `sample_networks` and `_mcmc_sample`,
# the bridge rungs) runs through NetworkCore's `spawn_all(f, n)`: it waits for
# every task and rethrows the first failure as the task's ORIGINAL exception,
# never wrapped in a `TaskFailedException`, so `catch e; e isa ArgumentError`
# works on a threaded sampler exactly as on a serial one.

# ----------------------------------------------------------------------------
# Proposals
#
# Two proposals for the binary-network sampler, chosen by `proposal=`:
#
# - `:tnt` (the default) — statnet's tie/no-tie proposal (`MH_TNT`, the
#   proposal behind R ergm's default `~sparse`): with probability ½ toggle a
#   uniformly chosen existing tie (a removal), otherwise a uniformly chosen
#   dyad. On a sparse network a uniformly random dyad is almost always an
#   absent tie, so most random-toggle proposals are additions the model
#   rejects; TNT spends half its proposals on removals and mixes several
#   times faster per toggle (≈7× in effective sample size on faux.mesa.high,
#   density 1 %). The proposal is asymmetric, so the acceptance ratio
#   carries its Hastings correction `_tnt_log_hastings` (statnet's
#   `TNT_LR_E`/`TNT_LR_DE`/`TNT_LR_DN`, including the E = 0 ↔ 1 boundary
#   cases).
# - `:random` — a uniformly random dyad (symmetric, no correction): the
#   pre-0.2 sampler, kept bit-identical for reproducing old results.
#
# Ties are drawn uniformly in O(log n) from a Fenwick (binary indexed) tree
# over the per-vertex out-neighbour counts (degrees on an undirected network,
# where each tie is stored in both adjacency lists): a uniform slot among
# the Σ counts is a uniform tie. The tree is allocated once per chain and
# updated in place on every accepted toggle, so a step allocates nothing.
# ----------------------------------------------------------------------------

const _PROPOSALS = (:tnt, :random, :spdyad)

function _check_proposal(proposal::Symbol; context::AbstractString)
    proposal in _PROPOSALS || throw(ArgumentError(
        "$context: unknown proposal $(repr(proposal)); expected :tnt (tie/no-tie, " *
        "statnet's default — half the proposals remove an existing tie), " *
        ":spdyad (TNT mixed with a shared-partner-focused proposal, R ergm's " *
        "choice for triadic models) or :random (a uniformly random dyad, the " *
        "pre-0.2 sampler)"))
    return proposal
end

# The state of a TNT chain: a Fenwick tree over the per-vertex adjacency
# counts (`slots` in total — the number of ties on a directed network, twice
# it on an undirected one), the number of ties the chain may toggle
# (`n_edges`; a tie on a masked dyad frozen by a `:free` chain does not
# count), and the number of dyads it may toggle (`n_dyads`).
mutable struct _TNTState
    tree::Vector{Int}
    topbit::Int
    slots::Int
    n_edges::Int
    n_dyads::Int
end

function _TNTState(net::Network{T,D}, toggleable::Symbol) where {T,D}
    n = Int(nv(net))
    tree = Int[length(outneighbors(net, v)) for v in 1:n]
    slots = sum(tree; init=0)
    for i in 1:n                                  # O(n) Fenwick build
        k = i + (i & -i)
        k <= n && (tree[k] += tree[i])
    end
    n_edges = Int(ne(net))
    if toggleable === :free && n_missing_dyads(net) > 0
        # Masked dyads are frozen at their face value: their ties are never
        # proposed, so they are not part of the toggleable tie count
        n_edges -= count(d -> has_edge(net, d[1], d[2]), missing_dyads(net))
    end
    return _TNTState(tree, n == 0 ? 0 : prevpow(2, n), slots, n_edges,
                     _n_toggleable(net, toggleable))
end

# The vertex holding slot `r` (1 ≤ r ≤ slots) and the slot's rank among that
# vertex's own slots: one binary descent of the Fenwick tree
@inline function _fw_find(tree::Vector{Int}, topbit::Int, r::Int)
    pos = 0
    step = topbit
    n = length(tree)
    @inbounds while step > 0
        nxt = pos + step
        if nxt <= n && tree[nxt] < r
            pos = nxt
            r -= tree[nxt]
        end
        step >>= 1
    end
    return pos + 1, r
end

@inline function _fw_add!(tree::Vector{Int}, i::Int, v::Int)
    n = length(tree)
    @inbounds while i <= n
        tree[i] += v
        i += i & -i
    end
    return nothing
end

# Keep the TNT state in step with an accepted toggle of the free dyad (i, j)
@inline function _tnt_toggled!(s::_TNTState, i::Int, j::Int, removal::Bool, directed::Bool)
    d = removal ? -1 : 1
    s.n_edges += d
    _fw_add!(s.tree, i, d)
    s.slots += d
    if !directed
        _fw_add!(s.tree, j, d)
        s.slots += d
    end
    return nothing
end

"""
    _tnt_log_hastings(n_edges, n_dyads, removal) -> Float64

The log Hastings ratio `log q(y | y*) − log q(y* | y)` of the tie/no-tie
proposal for toggling one dyad of a state with `n_edges` toggleable ties
among `n_dyads` toggleable dyads (statnet's `MH_TNT`, P = Q = ½). With ties
present a dyad is proposed with probability `½/n_edges + ½/n_dyads` if it
is a tie and `½/n_dyads` otherwise; with none, every proposal is a uniform
dyad (`1/n_dyads`) — the boundary case statnet corrected after Goudie.
"""
@inline function _tnt_log_hastings(E::Int, N::Int, removal::Bool)
    if removal                                    # E ≥ 1: the dyad is a tie
        fwd = 0.5 / E + 0.5 / N
        rev = E == 1 ? 1.0 / N : 0.5 / N
    else
        fwd = E == 0 ? 1.0 / N : 0.5 / N
        rev = 0.5 / (E + 1) + 0.5 / N
    end
    return log(rev / fwd)
end

# The binary-network adapter over `mh_toggle!`: mutates `net` in place and
# returns the sampled statistics matrix and (when `collect_networks`)
# network copies at each sampling point. A function barrier: `terms` is
# passed concretely so the closures and the kernel loop are fully typed;
# the directedness is the network's type parameter `D`.
#
# `toggleable` selects the dyads the chain may propose: `:free` — every unmasked dyad (masked dyads frozen at their face
# value, so the chain conditions on them); `:all` — every dyad, the
# unconditional model; `:masked` — only the masked dyads, drawn uniformly
# from their list, the observed dyads held fixed (the constrained chain of
# the missing-data MCMLE). `proposal` (`:tnt` or `:random`, see above)
# applies to `:free` and `:all`; the `:masked` chain always draws uniformly
# from the (short) list of masked dyads.
function _mh_run!(rng::AbstractRNG, net::Network{T,D}, terms::TermSet,
                  θ::Vector{Float64},
                  n_samples::Int, burnin::Int, interval::Int,
                  collect_networks::Bool, toggleable::Symbol=:free;
                  proposal::Symbol=:tnt) where {T,D}
    toggleable in _TOGGLEABLE || throw(ArgumentError(
        "_mh_run!: toggleable must be :free, :all or :masked (got $(repr(toggleable)))"))
    _check_proposal(proposal; context="_mh_run!")
    θ = _sampler_theta(θ)          # a -Inf offset: never raise its statistic
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
    end

    if proposal === :spdyad && !(toggleable === :free && n_missing_dyads(net) > 0)
        return _mh_run_spdyad!(rng, net, terms, θ, n_samples, burnin, interval,
                               collect_networks, toggleable)
    end
    if proposal === :tnt || proposal === :spdyad   # (:spdyad with frozen masked dyads: TNT)
        return _mh_run_tnt!(rng, net, terms, θ, n_samples, burnin, interval,
                            collect_networks, toggleable)
    end

    if toggleable === :all
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
    # dyad; i < j on undirected networks) — the pre-0.2 sampler, bit-identical
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

# The tie/no-tie chain over the `:free` or `:all` dyad set
function _mh_run_tnt!(rng::AbstractRNG, net::Network{T,D}, terms::TermSet,
                      θ::Vector{Float64}, n_samples::Int, burnin::Int, interval::Int,
                      collect_networks::Bool, toggleable::Symbol) where {T,D}
    n = Int(nv(net))
    s = _TNTState(net, toggleable)
    # Under `:free` a masked dyad is never proposed; under `:all` every dyad is
    check_mask = toggleable === :free && n_missing_dyads(net) > 0
    propose = function (rng)
        if rand(rng) < 0.5 && s.n_edges > 0
            # A uniformly chosen toggleable tie: a uniform adjacency slot,
            # rejected (and redrawn) when it is a frozen masked dyad
            while true
                v, k = _fw_find(s.tree, s.topbit, rand(rng, 1:s.slots))
                u = @inbounds Int(outneighbors(net, v)[k])
                i, j = D ? (v, u) : (min(v, u), max(v, u))
                (check_mask && is_missing_dyad(net, i, j)) && continue
                return (i, j)
            end
        end
        # A uniformly chosen toggleable dyad
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        while i == j || (!D && j < i) || (check_mask && is_missing_dyad(net, i, j))
            i = rand(rng, 1:n)
            j = rand(rng, 1:n)
        end
        return (i, j)
    end
    return _mh_run_with_proposal!(rng, net, terms, θ, n_samples, burnin,
                                  interval, collect_networks, propose, s)
end

# What an asymmetric proposal's state supplies to the adapter: its update
# after an accepted toggle (the network is already toggled) and its log
# Hastings ratio for a proposed toggle (the network not yet toggled)
@inline _proposal_toggled!(s::_TNTState, net::Network{T,D}, i::Int, j::Int,
                           removal::Bool) where {T,D} = _tnt_toggled!(s, i, j, removal, D)
@inline _proposal_log_hastings(s::_TNTState, net, i::Int, j::Int, removal::Bool) =
    _tnt_log_hastings(s.n_edges, s.n_dyads, removal)

# ----------------------------------------------------------------------------
# SPDyad: TNT mixed with a shared-partner-focused proposal (R ergm ≥ 4.6's
# `SPDyad`, the proposal its default `~sparse + .triadic` hint selects for
# models with triadic terms; after Wang & Atchadé 2013). With probability
# `focus` (R's `triFocus`, 0.25) the proposed dyad is drawn uniformly from
# L(y), the dyads with at least one shared partner — where toggles change
# the triadic statistics — and otherwise by TNT. The proposal probability of
# toggling dyad d from y is the mixture
#
#     q(d | y) = focus·1[d ∈ L(y)]/|L(y)| + (1 − focus)·q_TNT(d | y)
#
# (q_TNT alone when L(y) is empty), and the Hastings ratio is
# q(d | y*)/q(d | y). Toggling d never changes d's own shared-partner count,
# so membership of d is the same in y and y*; |L(y*)| is read off the
# shared-partner counts of the dyads the toggle touches, in O(degree).
#
# L(y) is kept as an indexable set: `keys` (the dyads, for a uniform draw),
# `cnt` (their shared-partner counts) and `pos` (dyad → index), updated on
# every accepted toggle. Undirected: common neighbours. Directed: outgoing
# two-paths i→k→j of the ordered dyad (R's default type, OTP).
# ----------------------------------------------------------------------------

mutable struct _SPDyadState
    keys::Vector{Tuple{Int,Int}}
    cnt::Vector{Int32}
    pos::Dict{Tuple{Int,Int},Int}
    tnt::_TNTState
    focus::Float64
end

@inline _sp_key(::Val{true}, a::Int, b::Int) = (a, b)
@inline _sp_key(::Val{false}, a::Int, b::Int) = a < b ? (a, b) : (b, a)

@inline function _sp_count(s::_SPDyadState, key::Tuple{Int,Int})
    idx = get(s.pos, key, 0)
    return idx == 0 ? Int32(0) : @inbounds s.cnt[idx]
end

function _sp_inc!(s::_SPDyadState, key::Tuple{Int,Int})
    idx = get(s.pos, key, 0)
    if idx == 0
        push!(s.keys, key)
        push!(s.cnt, Int32(1))
        s.pos[key] = length(s.keys)
    else
        @inbounds s.cnt[idx] += Int32(1)
    end
    return nothing
end

function _sp_dec!(s::_SPDyadState, key::Tuple{Int,Int})
    idx = s.pos[key]
    @inbounds if s.cnt[idx] > 1
        s.cnt[idx] -= Int32(1)
    else
        # swap-remove, keeping `keys` dense for the uniform draw
        last = length(s.keys)
        if idx != last
            moved = s.keys[last]
            s.keys[idx] = moved
            s.cnt[idx] = s.cnt[last]
            s.pos[moved] = idx
        end
        pop!(s.keys); pop!(s.cnt)
        delete!(s.pos, key)
    end
    return nothing
end

function _SPDyadState(net::Network{T,D}, toggleable::Symbol; focus::Float64=0.25) where {T,D}
    s = _SPDyadState(Tuple{Int,Int}[], Int32[], Dict{Tuple{Int,Int},Int}(),
                     _TNTState(net, toggleable), focus)
    sizehint!(s.pos, 4 * max(Int(ne(net)), 16))
    for k in 1:Int(nv(net))
        if D
            for a in inneighbors(net, k), b in outneighbors(net, k)
                a == b || _sp_inc!(s, (Int(a), Int(b)))
            end
        else
            nb = neighbors(net, k)
            for x in eachindex(nb), y in (x + 1):length(nb)
                _sp_inc!(s, _sp_key(Val(false), Int(nb[x]), Int(nb[y])))
            end
        end
    end
    return s
end

# The dyads whose shared-partner count a toggle of (i, j) changes by one are,
# undirected, (i,k) for k ~ j and (j,k) for k ~ i; directed (OTP), (i→k) for
# j→k and (k→j) for k→i — the dyad's own endpoints skipped, so the lists are
# the same before and after the toggle. `_sp_delta_size` counts how many of
# them enter (an addition: those without a shared partner) or leave (a
# removal: those with exactly one) L; `_proposal_toggled!` applies the change.
function _sp_delta_size(s::_SPDyadState, net::Network{T,D}, i::Int, j::Int,
                        removal::Bool) where {T,D}
    hit = removal ? Int32(1) : Int32(0)
    c = 0
    if D
        for k in outneighbors(net, j)
            k == i || (c += _sp_count(s, (i, Int(k))) == hit)
        end
        for k in inneighbors(net, i)
            k == j || (c += _sp_count(s, (Int(k), j)) == hit)
        end
    else
        for k in neighbors(net, j)
            k == i || (c += _sp_count(s, _sp_key(Val(false), i, Int(k))) == hit)
        end
        for k in neighbors(net, i)
            k == j || (c += _sp_count(s, _sp_key(Val(false), j, Int(k))) == hit)
        end
    end
    return removal ? -c : c
end

function _proposal_toggled!(s::_SPDyadState, net::Network{T,D}, i::Int, j::Int,
                            removal::Bool) where {T,D}
    _tnt_toggled!(s.tnt, i, j, removal, D)
    if D
        for k in outneighbors(net, j)
            k == i && continue
            removal ? _sp_dec!(s, (i, Int(k))) : _sp_inc!(s, (i, Int(k)))
        end
        for k in inneighbors(net, i)
            k == j && continue
            removal ? _sp_dec!(s, (Int(k), j)) : _sp_inc!(s, (Int(k), j))
        end
    else
        for k in neighbors(net, j)
            k == i && continue
            key = _sp_key(Val(false), i, Int(k))
            removal ? _sp_dec!(s, key) : _sp_inc!(s, key)
        end
        for k in neighbors(net, i)
            k == j && continue
            key = _sp_key(Val(false), j, Int(k))
            removal ? _sp_dec!(s, key) : _sp_inc!(s, key)
        end
    end
    return nothing
end

# TNT's probability of proposing a given dyad (a tie or not) when there are
# E toggleable ties among N toggleable dyads
@inline _tnt_prob(E::Int, N::Int, tie::Bool) =
    E == 0 ? 1.0 / N : (tie ? 0.5 / E + 0.5 / N : 0.5 / N)

function _proposal_log_hastings(s::_SPDyadState, net::Network{T,D}, i::Int, j::Int,
                                removal::Bool) where {T,D}
    E, N = s.tnt.n_edges, s.tnt.n_dyads
    L = length(s.keys)
    L2 = L + _sp_delta_size(s, net, i, j, removal)      # |L(y*)|
    inL = _sp_count(s, _sp_key(Val(D), i, j)) > 0
    f = s.focus
    fwd = L == 0 ? _tnt_prob(E, N, removal) :
          f * (inL ? 1.0 / L : 0.0) + (1 - f) * _tnt_prob(E, N, removal)
    E2 = removal ? E - 1 : E + 1
    rev = L2 == 0 ? _tnt_prob(E2, N, !removal) :
          f * (inL ? 1.0 / L2 : 0.0) + (1 - f) * _tnt_prob(E2, N, !removal)
    return log(rev / fwd)
end

function _mh_run_spdyad!(rng::AbstractRNG, net::Network{T,D}, terms::TermSet,
                         θ::Vector{Float64}, n_samples::Int, burnin::Int, interval::Int,
                         collect_networks::Bool, toggleable::Symbol) where {T,D}
    n = Int(nv(net))
    s = _SPDyadState(net, toggleable)
    t = s.tnt
    propose = function (rng)
        if !isempty(s.keys) && rand(rng) < s.focus
            @inbounds return s.keys[rand(rng, 1:length(s.keys))]
        end
        if rand(rng) < 0.5 && t.n_edges > 0
            v, k = _fw_find(t.tree, t.topbit, rand(rng, 1:t.slots))
            u = @inbounds Int(outneighbors(net, v)[k])
            return D ? (v, u) : (min(v, u), max(v, u))
        end
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        while i == j || (!D && j < i)
            i = rand(rng, 1:n)
            j = rand(rng, 1:n)
        end
        return (i, j)
    end
    return _mh_run_with_proposal!(rng, net, terms, θ, n_samples, burnin,
                                  interval, collect_networks, propose, s)
end

# The rest of the adapter, shared by the proposal closures: change
# statistics, state mutation and sample recording over `mh_toggle!`. `tnt`
# is the TNT chain's state (`nothing` for a symmetric proposal): it supplies
# the Hastings correction and is updated on every accepted toggle.
function _mh_run_with_proposal!(rng::AbstractRNG, net::Network{T,D}, terms::TermSet,
                                θ::Vector{Float64}, n_samples::Int, burnin::Int,
                                interval::Int, collect_networks::Bool,
                                propose::P, tnt::S=nothing) where {T,D,P,S}
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
        tnt === nothing || _proposal_toggled!(tnt, net, i, j, removal)
        return nothing
    end
    on_sample = function (k)
        @inbounds for c in 1:p
            samples[k, c] = current_stats[c]
        end
        collect_networks && push!(networks, _copy_network(net))
        return nothing
    end
    hastings = tnt === nothing ? nothing :
        (move, removal) -> _proposal_log_hastings(tnt, net, move[1], move[2], removal)

    mh_toggle!(rng, θ, delta, propose, change!, apply!, on_sample;
               burnin=burnin, interval=interval, n_samples=n_samples,
               hastings=hastings)

    return samples, networks
end

"""
    simulate_ergm(result::ERGMResult; n_sim::Int=1, burnin=nothing,
                  interval=nothing, rng::AbstractRNG=Random.default_rng(),
                  n_chains::Int=min(n_sim, 4),
                  missing::Symbol=:error, proposal::Symbol=:tnt) -> Vector{Network}

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
  (the shared dyad-scaled rule, [`mcmc_defaults`](@ref ERGM.Extension.mcmc_defaults))
- `interval::Int`: Steps between samples. Defaults to
  `max(100, n_dyads ÷ 10)`
- `rng::AbstractRNG`: Source of all random draws (reproducible seeding)
- `n_chains::Int`: Number of independent chains (see [`sample_networks`](@ref))
- `missing::Symbol=:error`: Missing-dyad policy (`:error` or
  `:condition_on_face`)
- `proposal::Symbol=:tnt`: MH proposal, `:tnt` (tie/no-tie, statnet's
  default) or `:random` (see [`mh_sample`](@ref))

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
                       missing::Symbol=:error,
                       proposal::Symbol=:tnt) where {T,D}
    method = _guard_missing(result.model.network, missing;
                            context="simulate_ergm")
    method === :condition_on_face &&
        _warn_condition_on_face(result.model.network, "simulation")
    return sample_networks(result.model, _simulation_theta(result);
                           n_sim=n_sim, burnin=burnin, interval=interval,
                           rng=rng, n_chains=n_chains, missing=missing,
                           proposal=proposal)
end

# The coefficients a fitted model is simulated at. A coefficient reported as
# NaN has no identification: a dyad-independent statistic that is constant,
# or a linear combination of the others, on every network — any value gives
# the same model, so 0 is used. A NaN on a dyad-dependent statistic (an MPLE
# whose design was singular at the observed network) is not such a
# statistic, and there is nothing to simulate from.
function _simulation_theta(result::ERGMResult)
    θ = result.coefficients
    any(isnan, θ) || return θ
    terms = result.model.formula.terms
    bad = [terms.names[k] for k in eachindex(θ) if isnan(θ[k]) && is_dyad_dependent(terms.terms[k])]
    isempty(bad) || throw(ArgumentError(
        "the coefficient(s) of $(join(bad, ", ")) are NaN (not identifiable at the " *
        "observed network), so the fitted model cannot be simulated; refit without " *
        "the term(s)"))
    return Float64[isnan(x) ? 0.0 : x for x in θ]
end

missing_policies(::typeof(simulate_ergm)) = _MISSING_POLICIES

"""
    sample_networks(model::ERGMModel, θ::Vector{Float64};
                    n_sim::Int=1, burnin=nothing, interval=nothing,
                    start_net::Union{Nothing,Network}=nothing,
                    rng::AbstractRNG=Random.default_rng(),
                    n_chains::Int=min(n_sim, 4),
                    missing::Symbol=:error,
                    proposal::Symbol=:tnt) -> Vector{Network{T,D}}

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
  (the shared dyad-scaled rule, [`mcmc_defaults`](@ref ERGM.Extension.mcmc_defaults))
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
- `proposal::Symbol=:tnt`: MH proposal, `:tnt` (tie/no-tie, statnet's
  default) or `:random` (see [`mh_sample`](@ref))

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
                         missing::Symbol=:error,
                         proposal::Symbol=:tnt) where {T,D}
    _guard_missing(isnothing(start_net) ? model.network : start_net, missing;
                   context="sample_networks")
    _check_proposal(proposal; context="sample_networks")
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
    spawn_all(n_chains) do c
        chain_rng = Random.Xoshiro(seeds[c])
        # A chain runs on the model's own network type: a user-supplied
        # start network of another directedness would not be that model.
        # (a -Inf offset forbids ties a random start could contain: such a
        # chain starts from the observed network, which has none)
        start = !isnothing(start_net) ? convert(Network{T,D}, _copy_network(start_net)) :
                any(isinf, θ) ? _copy_network(model.network) :
                _random_network(model.network; rng=chain_rng)
        _, nets = _mh_run!(chain_rng, start, terms, θ,
                           counts[c], burnin, interval, true;
                           proposal=proposal)
        chain_nets[c] = nets
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
