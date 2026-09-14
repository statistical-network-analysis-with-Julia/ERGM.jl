"""
Base types and interface for ERGM terms.

Defines the abstract term hierarchy and the interface that all
ERGM terms must implement.
"""

"""
    AbstractERGMTerm

Abstract base type for all ERGM terms.

All terms must implement:
- `compute(term, net) -> Float64`: Compute the term statistic for the network
- `change_stat(term, net, i, j) -> Float64`: The add-direction change statistic
  `g(y⁺ᵢⱼ) − g(y⁻ᵢⱼ)`, independent of the dyad's current state
- `name(term) -> String`: Return the term name

and may declare the term-trait protocol (`required_vertex_attributes`,
`requires_directed`, `is_dyad_dependent`, ...; see `src/terms/traits.jl`).

# Example
```julia
using ERGM
struct Density <: AbstractERGMTerm end          # a third-party term
ERGM.name(::Density) = "density"
ERGM.compute(::Density, net) = ne(net) / (nv(net) * (nv(net) - 1) / 2)
ERGM.change_stat(::Density, net, i, j) = 1 / (nv(net) * (nv(net) - 1) / 2)
ERGM.is_dyad_dependent(::Density) = false
net = load_dataset(:florentine_marriage)
compute(Density(), net)                       # 0.1667
Density() isa AbstractERGMTerm                # true
```
"""
abstract type AbstractERGMTerm end

"""
    StructuralTerm <: AbstractERGMTerm

Terms based purely on network structure (edges, triangles, etc.).
"""
abstract type StructuralTerm <: AbstractERGMTerm end

"""
    NodalTerm <: AbstractERGMTerm

Terms based on vertex attributes (nodefactor, nodecov, etc.).
"""
abstract type NodalTerm <: AbstractERGMTerm end

"""
    DyadicTerm <: AbstractERGMTerm

Terms based on dyad-level attributes or combinations of vertex attributes.
"""
abstract type DyadicTerm <: AbstractERGMTerm end

"""
    ConstraintTerm <: AbstractERGMTerm

Terms that represent constraints rather than model terms.
"""
abstract type ConstraintTerm <: AbstractERGMTerm end

# Interface functions

"""
    compute(term::AbstractERGMTerm, net) -> Float64

Compute the term statistic `g(y)` for the given network — the statnet
`summary(net ~ term)` value. `compute` is the shared Networks.jl statistic
generic (`Networks.compute`); ERGM adds one method per term type.

A masked (missing) dyad is read at its stored face value: the term layer
knows nothing about the mask (see `supports_missing(::AbstractERGMTerm)`),
the estimators do.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
compute(Edges(), net)            # 20.0
compute(Triangle(), net)         # 3.0
compute(NodeCov(:wealth), net)   # 2168.0
```
"""
function compute(term::AbstractERGMTerm, net)
    error("compute() not implemented for $(typeof(term))")
end

"""
    change_stat(term::AbstractERGMTerm, net, i::Int, j::Int) -> Float64

Compute the add-direction change statistic for dyad (i,j):
`g(y⁺ᵢⱼ) − g(y⁻ᵢⱼ)`, i.e. the statistic with edge (i,j) present minus the
statistic with it absent, holding all other dyads at their current values.

The result must not depend on whether edge (i,j) currently exists — this is
the convention required by both the MPLE design matrix and the
Metropolis–Hastings sampler (which negates it for removal proposals).

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
change_stat(Edges(), net, 1, 2)          # 1.0 — always, present or absent
change_stat(Triangle(), net, 2, 9)       # 0.0 — 2 and 9 share no neighbour
change_stat(Triangle(), net, 6, 7)       # 1.0 — 2 is tied to both 6 and 7
has_edge(net, 2, 9)                      # true; the values above do not depend on it
```
"""
function change_stat(term::AbstractERGMTerm, net, i::Int, j::Int)
    error("change_stat() not implemented for $(typeof(term))")
end

"""
    name(term::AbstractERGMTerm) -> String

The R ergm-style coefficient name of the term (`"edges"`, `"nodematch.sex"`,
`"gwesp.fixed.0.5"`, ...), as printed in the coefficient table. `name` is
the shared Networks.jl statistic generic.

# Example
```julia
using ERGM
name(Edges())            # "edges"
name(GWESP(0.5))         # "gwesp.fixed.0.5"
name(NodeMatch(:sex))    # "nodematch.sex"
```
"""
function name(term::AbstractERGMTerm)
    return string(typeof(term))
end

"""
    name(term::AbstractERGMTerm, net) -> String

The coefficient name `term` carries in a model built on `net` — the label
R ergm prints for that term *on that network*. For most terms this is
`name(term)`; the geometrically weighted shared-partner terms
([`GWESP`](@ref), [`GWDSP`](@ref)) are the exception, because R's label
depends on the network's directedness (`gwesp.fixed.0.5` on an undirected
network, `gwesp.OTP.fixed.0.5` on a directed one). `ERGMModel` and
[`summary_stats`](@ref) name their statistics with this method, so
`fit.model.formula.terms.names` matches `names(coef(fit))` of a statnet fit
of the same formula on either kind of network.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
name(GWESP(0.5), net)                # "gwesp.fixed.0.5"
dnet = network(3; directed=true)
name(GWESP(0.5), dnet)               # "gwesp.OTP.fixed.0.5"
name(Edges(), dnet)                  # "edges"
```
"""
name(term::AbstractERGMTerm, net) = name(term)

"""
    TermSet

A collection of ERGM terms.

Terms are stored as a tuple so that `compute_all`/`change_stat_all` compile
to statically dispatched calls per term instead of dynamic dispatch through
an abstractly-typed vector (which would dominate the MCMC inner loop).
`TermSet(::AbstractVector)` accepts a vector of terms (nested vectors are
spliced, as `fit_ergm` does).

# Example
```julia
using ERGM
ts = TermSet([Edges(), Triangle()])
length(ts)                         # 2
ts.names                           # ["edges", "triangle"]
compute_all(ts, load_dataset(:florentine_marriage))   # [20.0, 3.0]
```
"""
struct TermSet{T<:Tuple}
    terms::T
    names::Vector{String}

    function TermSet(terms::T) where {T<:Tuple}
        all(t -> t isa AbstractERGMTerm, terms) ||
            throw(ArgumentError("all elements must be AbstractERGMTerms"))
        names = [name(t) for t in terms]
        new{T}(terms, names)
    end

    # Names resolved against a network (`name(term, net)`): what
    # `_materialize(ts, net)` builds for `ERGMModel`, so a directed model's
    # `gwesp` coefficient is labelled `gwesp.OTP.fixed.<decay>` as in R
    function TermSet(terms::T, names::Vector{String}) where {T<:Tuple}
        all(t -> t isa AbstractERGMTerm, terms) ||
            throw(ArgumentError("all elements must be AbstractERGMTerms"))
        length(names) == length(terms) ||
            throw(ArgumentError("TermSet: $(length(names)) names for $(length(terms)) terms"))
        new{T}(terms, names)
    end
end

TermSet(terms::Vector{<:AbstractERGMTerm}) = TermSet(Tuple(terms))
TermSet(terms::AbstractVector) = TermSet(Tuple(_collect_terms(terms)))

"""
    _collect_terms(terms) -> Vector{AbstractERGMTerm}

Normalize a user-supplied term collection into a flat `Vector{AbstractERGMTerm}`.

Accepts a single term, a vector of terms, and — for callers that build term
lists programmatically — a vector that mixes terms with *vectors of terms*
(`[Edges(), [NodeMatch(:g; diff=true, level=l) for l in levels]]`); such
nested vectors are spliced in place instead of raising a raw `MethodError`.
(`Degree(0:2)` is a single expanding term since 0.2 and needs no splicing.)

Anything else throws an `ArgumentError` naming the offending element, its
position and its type — with a hint when it is a term *type* rather than an
instance (`Edges` instead of `Edges()`), the most common slip. `public`: the
variants (TERGM's `stergm`) accept exactly what `fit_ergm` accepts by
running their sides through it.

# Example
```julia
using ERGM
ERGM._collect_terms(Edges())                                        # [Edges()]
length(ERGM._collect_terms(Any[Edges(), [NodeMatch(:g), Triangle()]]))   # 3: spliced
try ERGM._collect_terms([Edges, Triangle()]) catch e; e isa ArgumentError end   # true: a term TYPE
```
"""
_collect_terms(term::AbstractERGMTerm) = AbstractERGMTerm[term]
function _collect_terms(terms::AbstractVector)
    out = AbstractERGMTerm[]
    for (k, t) in enumerate(terms)
        if t isa AbstractERGMTerm
            push!(out, t)
        elseif t isa AbstractVector && !isempty(t) && all(x -> x isa AbstractERGMTerm, t)
            append!(out, t)
        else
            hint = t isa Type && t <: AbstractERGMTerm ?
                " — `$(nameof(t))` is a term type, not a term; did you mean `$(nameof(t))()`?" :
                t isa AbstractVector ?
                " — a vector of terms may only contain terms (`[Edges(), [1, 2]]` " *
                "is not)" : ""
            throw(ArgumentError(
                "element $k of the term list is not an ERGM term: got " *
                "$(repr(t)) of type $(typeof(t))$hint. Every element must be an " *
                "`AbstractERGMTerm` (e.g. `Edges()`, `Triangle()`, `NodeMatch(:attr)`)."))
        end
    end
    isempty(out) && throw(ArgumentError(
        "the term list is empty; an ERGM needs at least one term (e.g. `[Edges()]`)"))
    return out
end

Base.length(ts::TermSet) = length(ts.terms)
Base.iterate(ts::TermSet, state=1) = state > length(ts) ? nothing : (ts.terms[state], state + 1)
Base.getindex(ts::TermSet, i) = ts.terms[i]

"""
    compute_all(ts::TermSet, net) -> Vector{Float64}

Compute all term statistics for the network — the vector `g(y)` of the
model's sufficient statistics, in term order. `compute_all` is the shared
Networks.jl statistic generic.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
compute_all(TermSet([Edges(), NodeCov(:wealth)]), net)   # [20.0, 2168.0]
```
"""
function compute_all(ts::TermSet, net)
    return collect(_compute_tuple(ts, net))
end

# The per-term evaluations of a TermSet as an NTuple{p,Float64}, emitted
# term by term by a generated function so the code is the same straight-line
# sequence of statically dispatched calls for ANY number of terms. `Base.map`
# over a tuple is unrolled only below 32 elements; from 32 on it falls back
# to a Vector{Any} of boxed Float64s plus a splat (~30 KB per call), which
# put a per-step allocation cliff under every model with ≥ 32 statistics —
# a NodeMix on an 8-level attribute alone has 35 cells (panel 2026-09
# round-2 blocker; pinned by the 36-statistic allocation tests).
@generated function _compute_tuple(ts::TermSet{T}, net) where {T<:Tuple}
    p = length(T.parameters)
    calls = [:(compute(ts.terms[$k], net)) for k in 1:p]
    return :(tuple($(calls...)))
end

@generated function _change_stat_tuple(ts::TermSet{T}, net, i::Int, j::Int) where {T<:Tuple}
    p = length(T.parameters)
    calls = [:(change_stat(ts.terms[$k], net, i, j)) for k in 1:p]
    return :(tuple($(calls...)))
end

"""
    change_stat_all(ts::TermSet, net, i::Int, j::Int) -> Vector{Float64}

Compute add-direction change statistics for all terms for dyad (i,j) — one
row of the MPLE design matrix.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
change_stat_all(TermSet([Edges(), NodeCov(:wealth)]), net, 1, 2)   # [1.0, 46.0]
```
"""
function change_stat_all(ts::TermSet, net, i::Int, j::Int)
    return collect(_change_stat_tuple(ts, net, i, j))
end

"""
    change_stat_all!(dest, ts::TermSet, net, i::Int, j::Int) -> dest

In-place version of [`change_stat_all`](@ref) for use in sampling loops.
"""
@generated function change_stat_all!(dest::AbstractVector{Float64}, ts::TermSet{T}, net,
                                     i::Int, j::Int) where {T<:Tuple}
    p = length(T.parameters)
    body = [:(dest[$k] = change_stat(ts.terms[$k], net, i, j)) for k in 1:p]
    return quote
        $(body...)
        return dest
    end
end

"""
    summary_stats(net, terms; missing=:error) -> NamedTuple

Compute the observed value of every term on `net` without fitting anything —
R ergm's `summary(net ~ edges + triangle)`. Returns a `NamedTuple` keyed by
the terms' names as they appear in a model on `net` (see
[`name`](@ref)`(term, net)`: a directed network's `GWESP(0.5)` is
`gwesp.OTP.fixed.0.5`, as in R); `terms` is a vector of terms, in the same
shapes [`fit_ergm`](@ref) accepts. A multi-degree count term (`Degree(0:2)`)
expands into one entry per degree, as it does in a fit.

**Masked dyads** (`set_missing_dyad!`) are refused by default — a
statistic is a number computed from every dyad, and an unobserved dyad has
no value to contribute (the ecosystem missing-data contract,
`Networks.require_observed`). `missing=:face` is the explicit opt-in: every
masked dyad is read at its stored face value (a masked tie counts as a
tie). This is a *descriptive* statistic, so the opt-in is legitimate — but
note that R's `summary()` treats an NA dyad as *absent*, so to reproduce
R's number remove the masked ties first (the provenanced masked-flomarriage
fixture does exactly that). `missing_policies(summary_stats) == (:error,
:face)`. The protocol-level `compute`/`compute_all` are raw evaluations
with no policy of their own (they are what the estimators, which handle
the mask themselves, call).

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
summary_stats(net, [Edges(), Triangle()])   # (edges = 20.0, triangle = 3.0)
summary_stats(net, [Degree(0:1)])           # (degree0 = 1.0, degree1 = 4.0)
masked = copy(net); set_missing_dyad!(masked, 1, 9)   # a masked tie
try summary_stats(masked, [Edges()]) catch e; e isa ArgumentError end   # true
summary_stats(masked, [Edges()]; missing=:face)       # (edges = 20.0,)
```
"""
function summary_stats(net, terms::AbstractVector; missing::Symbol=:error)
    require_observed(net, missing; context="summary_stats")
    terms = _collect_terms(terms)
    expanded = AbstractERGMTerm[]
    for t in terms
        e = _expand_degrees(t)
        e isa AbstractVector ? append!(expanded, e) : push!(expanded, e)
    end
    terms = expanded
    names = [Symbol(name(t, net)) for t in terms]
    values = [compute(t, net) for t in terms]
    return NamedTuple{Tuple(names)}(values)
end

# A descriptive statistic: `:face` is a legitimate opt-in (as for the SNA
# measures), and the default refuses (the shared `require_observed` text,
# which names `missing=:face` because this routine really takes it)
missing_policies(::typeof(summary_stats)) = (:error, :face)

# ============================================================================
# Model and Result Types
# ============================================================================

"""
    ERGMFormula(terms; constraints=ConstraintTerm[])

Represents an ERGM model specification: the right-hand side of R ergm's
`net ~ edges + triangle + nodematch("attr")`.

`terms` is a [`TermSet`](@ref) or any vector of terms (the shapes
[`fit_ergm`](@ref) accepts, including `[Edges(), Degree(0:2)]`).

**Constraints are not implemented.** `ConstraintTerm` is exported as a
reserved abstract type, but ERGM.jl fits unconstrained models only: passing
a non-empty `constraints=` vector throws an `ArgumentError` rather than
storing constraints that no sampler or estimator would honour (statnet's
`constraints=` — `edges`, `degrees`, `blockdiag`, `observed`, … — has no
counterpart yet).

# Fields
- `terms::TermSet`: Model terms
- `constraints::Vector{ConstraintTerm}`: always empty (see above)

# Example
```julia
using ERGM
formula = ERGMFormula([Edges(), Triangle()])
length(formula.terms)    # 2
```
"""
struct ERGMFormula
    terms::TermSet
    constraints::Vector{ConstraintTerm}

    function ERGMFormula(terms::TermSet;
                         constraints::AbstractVector=ConstraintTerm[])
        isempty(constraints) || throw(ArgumentError(
            "constraints are not implemented; ERGM.jl fits unconstrained models " *
            "only (statnet `constraints=` — edges, degrees, blockdiag, observed, " *
            "… — has no counterpart yet). Remove the `constraints=` keyword; " *
            "`ConstraintTerm` is reserved for a future release."))
        new(terms, ConstraintTerm[])
    end
end

ERGMFormula(terms::AbstractVector; constraints::AbstractVector=ConstraintTerm[]) =
    ERGMFormula(TermSet(terms); constraints=constraints)

# The formula prints as R prints it — the right-hand side — never as the
# struct (a materialized attribute term holds a per-vertex vector)
Base.show(io::IO, f::ERGMFormula) = print(io, "ERGMFormula: ", join(f.terms.names, " + "))

"""
    ERGMModel{T,D}
    ERGMModel(formula::ERGMFormula, net::Network{T,D}; reference=:bernoulli)

An ERGM model specification with observed network. `T` is the network's
vertex-index type and `D` its directedness (`Graphs.is_directed(model) == D`,
known at the type level, so every hot loop reading `model.network` is
statically typed).

Construction validates the formula against the network — every
attribute-based term's vertex attribute must exist on the network, and
intrinsically directed terms (e.g. `Mutual`) are rejected on undirected
networks — throwing an `ArgumentError` otherwise. Attribute-based nodal
terms are then *materialized*: their attribute values are snapshotted into
dense typed vectors (see `src/terms/materialize.jl`) so that change
statistics in the estimation and sampling hot loops avoid the untyped
attribute storage. Materialized terms keep the original term names and
semantics.

**One-mode networks only.** A two-mode (bipartite-flagged) network — created
with `network(n; bipartite=k)`, or a `BipartiteNetwork` — is refused with an
`ArgumentError`: the bipartite ERGM terms of statnet (`b1degree`, `b2degree`,
`b1factor`, `b1nodematch`, …) and the two-mode proposal kernel are not
implemented, and fitting the one-mode terms would silently count the
structurally impossible within-mode dyads as observations. See the README's
"Not implemented" section.

**No self-loops.** A network that *contains* a self-loop (`has_edge(net,
v, v)`; only possible with `network(n; loops=true)`) is refused with an
`ArgumentError` too: the term statistics would count the loop, but the
pseudo-likelihood, `nobs`, the MH proposal and every simulation range over
the `n(n−1)` (or `n(n−1)/2`) off-diagonal dyads only, so a loop would
inflate the observed statistics against a model that can never reproduce
it (R ergm warns "This network contains loops" and fits regardless). A
`loops=true` network with no loop present is accepted and modelled as the
loop-free network it is — the diagonal is structurally absent.

`show(model)` prints the network's size and directedness and the formula
(`ERGMModel{Int64,false}: 16 vertices, 20 edges (undirected); terms: edges
+ nodecov.wealth`), never the materialized attribute vectors.

# Fields
- `formula::ERGMFormula`: Model formula (with materialized terms)
- `network::Network{T,D}`: Observed network
- `reference::Symbol`: Reference measure (:bernoulli by default)

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
model = ERGMModel(ERGMFormula([Edges(), Triangle()]), net)
is_directed(model)                  # false
model.formula.terms.names           # ["edges", "triangle"]
```
"""
struct ERGMModel{T,D}
    formula::ERGMFormula
    network::Network{T,D}
    reference::Symbol

    function ERGMModel(formula::ERGMFormula, net::Network{T,D};
                       reference::Symbol=:bernoulli) where {T,D}
        _refuse_two_mode(net)
        _refuse_self_loops(net)
        _validate_formula(formula.terms, net)
        mformula = ERGMFormula(_materialize(formula.terms, net);
                               constraints=formula.constraints)
        new{T,D}(mformula, net, reference)
    end
end

# A `BipartiteNetwork` is two-mode by construction; refuse it with the same
# message instead of a MethodError.
ERGMModel(formula::ERGMFormula, net::BipartiteNetwork; kwargs...) =
    _refuse_two_mode(net)

"""
    _refuse_two_mode(net) -> net

The refusal `ERGMModel` gives a two-mode (bipartite) network: an
`ArgumentError` saying that bipartite terms and the two-mode proposal kernel
are not implemented and that fitting the one-mode terms would count the
impossible within-mode dyads as observations. Returns `net` unchanged when
it is one-mode. `public` so a harness (ERGMUserterms) refuses up front with
ERGM's own sentence rather than a mirrored copy of it.

# Example
```julia
using ERGM
ERGM._refuse_two_mode(Network(4)) === Network                          # false (it returns the network)
ERGM._refuse_two_mode(Network(4)) isa Network                          # true
try ERGM._refuse_two_mode(network(5; bipartite=2)) catch e; e isa ArgumentError end   # true
```
"""
function _refuse_two_mode(net)
    is_two_mode(net) && throw(ArgumentError(
        "ERGM.jl fits one-mode networks only; this network is two-mode " *
        "(bipartite). Bipartite ERGM terms (statnet b1degree, b2degree, " *
        "b1factor, b1nodematch, …) and the two-mode proposal kernel are not " *
        "implemented — see README 'Not implemented'. Fitting the one-mode " *
        "terms would silently count the impossible within-mode dyads as " *
        "observations, so the model is refused instead."))
    return net
end

"""
    _refuse_self_loops(net) -> net

The refusal `ERGMModel` gives a network containing a self-loop: an
`ArgumentError` naming the looped vertices. A self-loop is refused rather
than counted by the statistics and ignored by the estimators and samplers
(which range over the off-diagonal dyads only) — the two halves of the
package would disagree about the data (R ergm warns "This network contains
loops"). Returns `net` unchanged when it has none. `public` for the same
reason as [`_refuse_two_mode`](@ref).

# Example
```julia
using ERGM
ERGM._refuse_self_loops(Network(4)) isa Network                        # true
loopy = network(3; loops=true); add_edge!(loopy, 2, 2)
try ERGM._refuse_self_loops(loopy) catch e; occursin("at vertex 2", e.msg) end   # true
```
"""
function _refuse_self_loops(net)
    loops = [Int(v) for v in vertices(net) if has_edge(net, v, v)]
    isempty(loops) && return net
    shown = length(loops) > 10 ? join(loops[1:10], ", ") * ", …" : join(loops, ", ")
    throw(ArgumentError(
        "the network contains $(length(loops)) self-loop$(length(loops) == 1 ? "" : "s") " *
        "(at vertex $shown). ERGM.jl models the off-diagonal dyads only: the term " *
        "statistics would count a loop, but the pseudo-likelihood, nobs, the MH " *
        "proposal and every simulation never touch the diagonal, so the observed " *
        "statistics would be compared against a model that cannot reproduce " *
        "them (R ergm warns \"This network contains loops\" here). Remove the " *
        "loops (`rem_edge!(net, v, v)`) or build the network with `loops=false`."))
end

# Directedness is a type parameter: `is_directed(model)` is a compile-time
# constant, and there is no separate `directed` field to fall out of sync.
Graphs.is_directed(::ERGMModel{T,D}) where {T,D} = D
Graphs.is_directed(::Type{<:ERGMModel{T,D}}) where {T,D} = D

function Base.show(io::IO, m::ERGMModel{T,D}) where {T,D}
    net = m.network
    n_masked = n_missing_dyads(net)
    print(io, "ERGMModel{$T,$D}: $(nv(net)) vertices, $(ne(net)) edges ",
          D ? "(directed)" : "(undirected)",
          n_masked > 0 ? ", $n_masked masked dyad$(n_masked == 1 ? "" : "s")" : "",
          "; terms: ", join(m.formula.terms.names, " + "))
    m.reference === :bernoulli || print(io, "; reference: ", m.reference)
    return nothing
end

"""
    MCMLEConvergence

The convergence report an MCMLE fit carries in `fit.mcmc_convergence`
(`nothing` for MPLE): a NamedTuple with

- `iterations::Int`: MCMLE iterations run (≤ `maxiter`);
- `step_length::Float64`: the last Hummel step length γ (1.0 at convergence);
- `t_ratios::Vector{Float64}`: per-statistic convergence t-ratios
  `|g_obs − ḡ| / sd(g)` on the final sample at the returned coefficients;
- `hotelling_p::Float64`: p-value of the Hotelling T² test on that sample;
- `n_eff::Float64`: the effective sample size behind it (Geyer, summed over
  chains).

See [`mcmc_convergence`](@ref), which computes the last three.

# Example
```julia
using ERGM, Random
net = load_dataset(:florentine_marriage)
fit = mcmle(ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), net);
            n_samples=300, bridge_rungs=0, rng=Xoshiro(1))
fit.mcmc_convergence isa ERGM.MCMLEConvergence   # true
fit.mcmc_convergence.step_length                 # 1.0 at convergence
```
"""
const MCMLEConvergence = NamedTuple{(:iterations, :step_length, :t_ratios,
                                     :hotelling_p, :n_eff),
                                    Tuple{Int, Float64, Vector{Float64},
                                          Float64, Float64}}

"""
    ERGMResult{T,D}

Results from fitting an ERGM. The type parameters are those of the fitted
[`ERGMModel`](@ref) (vertex-index type and directedness).

# Fields
- `model::ERGMModel{T,D}`: The fitted model
- `coefficients::Vector{Float64}`: Estimated coefficients
- `std_errors::Vector{Float64}`: Standard errors
- `z_values::Vector{Float64}`: Z-statistics
- `p_values::Vector{Float64}`: Two-sided p-values
- `vcov::Matrix{Float64}`: Estimated covariance matrix of the coefficients
- `loglik::Float64`: Log-likelihood (or pseudo-log-likelihood)
- `aic::Float64`: AIC
- `bic::Float64`: BIC
- `method::Symbol`: Estimation method (:mple or :mcmle)
- `converged::Bool`: Convergence status
- `mcmc_samples::Union{Nothing, Matrix{Float64}}`: Statistics sampled at the
  final coefficient values (for MCMLE)
- `se_type::Symbol`: How the standard errors were obtained (`:hessian` for
  inverse observed information, `:bootstrap` for parametric bootstrap,
  `:mcmc` for the inverse Fisher information estimated from MCMC samples)
- `missing_method::Symbol`: How masked (unobserved) dyads were treated —
  `:none` if the network had none, `:available_case` if they were dropped
  from the pseudo-likelihood (MPLE), `:mle` if they were integrated out of
  the likelihood by the constrained chain of the missing-data MCMLE
  (`missing=:mle`), `:condition_on_face` if they were held fixed at their
  stored face value throughout MCMC (`missing=:condition_on_face`). See
  `src/missing.jl`.
- `vcov_fisher::Matrix{Float64}`: For an MCMLE fit, the inverse Fisher
  information `Σ̂⁻¹` from the final MCMC sample alone — `vcov` is this plus
  the Monte-Carlo component `V·Σ_mc·V` (see [`mcmle`](@ref)). For an MPLE
  fit it is `vcov` itself.
- `mcmc_se::Vector{Float64}`: The Monte-Carlo standard error of each
  coefficient, `sqrt(diag(V·Σ_mc·V))` — the part of `std_errors` that comes
  from the finite MCMC sample rather than from the data (R's "MCMC %").
  Zeros for an MPLE fit. Accessor: [`mcmc_se`](@ref).
- `mcmc_convergence::Union{Nothing, MCMLEConvergence}`: For an MCMLE fit,
  the convergence report recomputed on the final sample at the returned
  coefficients — a NamedTuple `(iterations, step_length, t_ratios,
  hotelling_p, n_eff)` — so an unconverged fit can say *how* unconverged it
  is (`converged` is the loop's verdict; the warning quotes these numbers).
  `nothing` for an MPLE fit.
- `chain_lengths::Vector{Int}`: For an MCMLE fit, the number of rows of
  `mcmc_samples` each of the `n_chains` independent chains contributed, in
  order (the sample is their concatenation), so [`mcmc_diagnostics`](@ref)
  can work chain by chain instead of across the seams. Empty for MPLE.
- `boot_replicates::Union{Nothing, Matrix{Float64}}`: For `se=:bootstrap`,
  the `n_boot × p` matrix of refitted coefficients the covariance was taken
  over; a replicate on which the MPLE did not exist (a boundary statistic
  or a separated design in the simulated network) is a row of `NaN`s and
  was excluded. `nothing` otherwise.

The full ecosystem StatsAPI surface is available on a result: `coef`,
`stderror`, `vcov`, `confint`, `loglikelihood`, `nobs`, `dof`, `aic`, `bic`
and `coeftable` (a `Networks.CoefficientTable`), plus the result-metadata
protocol (`Networks.fit_metadata`).

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
fit = fit_ergm(net, [Edges(), NodeCov(:wealth)])
coef(fit)             # 2-vector
confint(fit)          # 2×2 matrix of normal-theory 95% limits
coeftable(fit)        # the R-style table `show(fit)` prints
```
"""
struct ERGMResult{T,D}
    model::ERGMModel{T,D}
    coefficients::Vector{Float64}
    std_errors::Vector{Float64}
    z_values::Vector{Float64}
    p_values::Vector{Float64}
    vcov::Matrix{Float64}
    loglik::Float64
    aic::Float64
    bic::Float64
    method::Symbol
    converged::Bool
    mcmc_samples::Union{Nothing, Matrix{Float64}}
    se_type::Symbol
    missing_method::Symbol
    vcov_fisher::Matrix{Float64}
    mcmc_se::Vector{Float64}
    mcmc_convergence::Union{Nothing, MCMLEConvergence}
    chain_lengths::Vector{Int}
    boot_replicates::Union{Nothing, Matrix{Float64}}
end

"""
    mcmc_se(result::ERGMResult) -> Vector{Float64}

The Monte-Carlo component of each coefficient's standard error: for an MCMLE
fit, `sqrt(diag(V·Σ_mc·V))` where `V = vcov_fisher` is the inverse Fisher
information and `Σ_mc` the Geyer initial-sequence covariance of the sampled
statistics' mean (Hunter & Handcock 2006 §3.3) — the uncertainty that comes
from the finite MCMC sample rather than from the data. `stderror(result)`
already includes it (`se² = se_fisher² + mcmc_se²`); `show` prints R's
`summary.ergm` "MCMC %" column, which is the share of the *standard error*
(not of its variance) that the Monte-Carlo term adds:
`round(100 · (se − se_fisher) / se)` with `se_fisher =
sqrt.(diag(result.vcov_fisher))` — R's `100 * (tot.se - mod.se) / tot.se`.
A large share says the sample, not the data, limits the precision: raise
`n_samples`/`n_chains`. Zeros for an MPLE fit (no MCMC sample).

# Example
```julia
using ERGM, Random
net = load_dataset(:florentine_marriage)
fit = fit_ergm(net, [Edges(), GWESP(0.5)]; method=:mcmle, n_samples=1000, rng=Xoshiro(1))
all(mcmc_se(fit) .< stderror(fit))            # true
all(mcmc_se(fit_ergm(net, [Edges()])) .== 0)  # true: MPLE has no MC term
```
"""
mcmc_se(result::ERGMResult) = result.mcmc_se

"""
    has_dyad_dependent(model::ERGMModel) -> Bool

Whether the model's formula contains any dyad-dependent term (see
[`is_dyad_dependent`](@ref); unknown term types count as dependent).

This is THE predicate that decides whether a pseudo-likelihood fit is an
approximation: for a dyad-independent formula the MPLE *is* the MLE and the
inverse-Hessian standard errors are the exact ML ones. It is defined once and
used by both `show(::ERGMResult)` (which prints the prose caveat) and
`is_exact(::ERGMResult)` (which reports the same fact to a machine), so the two
can never disagree. The ERGM variants (ERGMCount, ERGMMulti, TERGM) add methods
for their own model types rather than defining same-named privates.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
has_dyad_dependent(ERGMModel(ERGMFormula([Edges(), NodeCov(:wealth)]), net))  # false
has_dyad_dependent(ERGMModel(ERGMFormula([Edges(), Triangle()]), net))        # true
```
"""
has_dyad_dependent(model::ERGMModel) =
    any(is_dyad_dependent(t) for t in model.formula.terms)

"""
    _has_dyad_dependent

Deprecated `const` alias of [`has_dyad_dependent`](@ref) — the same
function, reached through the private name the variants (TERGM, ERGMMulti)
used before 0.2, kept `public` so a method added to either name lands on the
one predicate `show`/`is_exact` dispatch on. New code uses the public name.

# Example
```julia
using ERGM
ERGM._has_dyad_dependent === has_dyad_dependent   # true
```
"""
const _has_dyad_dependent = has_dyad_dependent

function Base.show(io::IO, result::ERGMResult)
    println(io, "ERGM Results")
    println(io, "============")
    println(io, "Method: $(result.method)")

    # Missing dyads are only mentioned when there are any: how they were
    # treated is part of the estimand, so it must never be invisible.
    n_masked = n_missing_dyads(result.model.network)
    if n_masked > 0
        treatment = result.missing_method === :available_case ?
            "available-case (masked dyads dropped from the pseudo-likelihood)" :
            result.missing_method === :mle ?
            "missing-data maximum likelihood (masked dyads integrated out by a " *
            "constrained chain)" :
            result.missing_method === :condition_on_face ?
            "conditioned on face values (masked dyads held fixed during MCMC)" :
            string(result.missing_method)
        println(io, "Missing dyads: $n_masked masked; $treatment")
    end

    if result.method === :mcmle && isnan(result.loglik)
        # `bridge_rungs=0`: the path-sampling estimate was skipped on request
        println(io, "Log-likelihood: not estimated (bridge_rungs=0)")
        println(io, "AIC: not estimated, BIC: not estimated")
    else
        println(io, "Log-likelihood: $(round(result.loglik, digits=4))")
        println(io, "AIC: $(round(result.aic, digits=2)), BIC: $(round(result.bic, digits=2))")
    end
    println(io, "Converged: $(result.converged)")
    if !result.converged
        # The caveat sits right under the verdict: an unconverged fit must
        # never look like a fit with a footnote
        println(io, "  ", _nonconvergence_caveat(result))
    end
    println(io)
    println(io, "Coefficients:")

    # Shared ecosystem presentation layer: the printed table IS
    # `coeftable(result)` (a Networks.CoefficientTable, rendered through
    # `print_coeftable`: Estimate / Std.Error / z value / Pr(>|z|) with
    # significance codes), so what is shown and what is inspected agree.
    show(io, coeftable(result))

    # MCMLE: R's `summary.ergm` "MCMC %" column — the percentage of each
    # standard error that the Monte-Carlo term adds over the Fisher part
    if result.method === :mcmle
        shares = _mcmc_percent(result)
        names = result.model.formula.terms.names
        println(io)
        println(io, "MCMC % of the standard error (100·(se − se_fisher)/se): ",
                join(("$(names[k]) $(shares[k])" for k in eachindex(names)), ", "))
    end

    fixed = _fixed_coefficient_note(result)
    if fixed !== nothing
        println(io)
        println(io, "Note: ", fixed)
    end
    excluded = _boot_exclusion_note(result)
    if excluded !== nothing
        println(io)
        println(io, "Note: ", excluded)
    end

    # Honest-uncertainty caveat: pseudo-likelihood fits of dyad-dependent
    # models have suspect inverse-Hessian standard errors (statnet prints an
    # analogous warning). Dyad-independent formulas need no caveat — there
    # the pseudo-likelihood is the likelihood.
    if result.method == :mple && has_dyad_dependent(result.model)
        println(io)
        if result.se_type == :bootstrap
            println(io, "Note: this model contains dyad-dependent terms and was fit by maximum")
            println(io, "pseudolikelihood (MPLE). Standard errors are parametric-bootstrap")
            println(io, "estimates; the MPLE point estimates may still be biased. Consider")
            println(io, "refitting with method=:mcmle.")
        else
            println(io, "Warning: this model contains dyad-dependent terms and was fit by")
            println(io, "maximum pseudolikelihood (MPLE). The standard errors are based on the")
            println(io, "naive pseudolikelihood and are suspect (typically anticonservative);")
            println(io, "the p-values should not be trusted. Refit with method=:mcmle, or use")
            println(io, "se=:bootstrap for parametric-bootstrap standard errors.")
        end
    end
end

# R's "MCMC %": `ergm:::summary.ergm` computes
# `round(100 * (tot.se - mod.se) / tot.se)` — the share of the TOTAL
# standard error that the Monte-Carlo term adds over the model (Fisher)
# part, rounded to an integer. NaN SEs give "NaN". (The variance share
# `100·mcmc_se²/se²` is a different, larger number; do not print it under
# R's name.)
function _mcmc_percent(result::ERGMResult)
    se_fisher = sqrt.(max.(diag(result.vcov_fisher), 0.0))
    return [isfinite(se) && isfinite(f) && se > 0 ? round(Int, 100 * (se - f) / se) : NaN
            for (f, se) in zip(se_fisher, result.std_errors)]
end

# THE non-convergence sentence: printed by `show` under `Converged: false`
# and listed by `approximations`, from the same numbers, so they cannot
# disagree
function _nonconvergence_caveat(result::ERGMResult)
    if result.method === :mcmle
        c = result.mcmc_convergence
        detail = c === nothing ? "" :
            " (max t-ratio $(_fmt3(maximum(c.t_ratios))), Hotelling p " *
            "$(_fmt3(c.hotelling_p)), step length γ $(_fmt3(c.step_length)) " *
            "after $(c.iterations) iteration$(c.iterations == 1 ? "" : "s"))"
        return "MCMLE did not converge$detail: point estimates and standard " *
               "errors unreliable — increase maxiter/n_samples, or refit with " *
               "init=coef(fit)"
    else
        return "the maximum pseudo-likelihood estimate does not exist or was not " *
               "reached: the pseudo-likelihood has no finite maximum (perfect " *
               "separation — R ergm: \"The MPLE does not exist!\") or Newton " *
               "hit maxiter; the coefficients are the last iterate and the " *
               "standard errors and p-values are meaningless"
    end
end

# Bootstrap replicates without a finite MPLE (a boundary statistic or a
# separated design in the SIMULATED network) were excluded from the
# covariance: said in `show` and in `approximations`, from the replicate
# matrix itself
function _boot_exclusion_note(result::ERGMResult)
    reps = result.boot_replicates
    reps === nothing && return nothing
    n_boot = size(reps, 1)
    n_dropped = count(b -> !all(isfinite, view(reps, b, :)), 1:n_boot)
    n_dropped == 0 && return nothing
    return "$n_dropped of the $n_boot parametric-bootstrap refits had no finite " *
           "MPLE (a statistic at the boundary of its attainable range, or a " *
           "separated design, in the simulated network) and were excluded from " *
           "the standard errors, which are the empirical covariance of the " *
           "remaining $(n_boot - n_dropped) refits (fit.boot_replicates)"
end

# ============================================================================
# The shared result-metadata protocol (Networks.jl `src/results.jl`)
# ============================================================================
#
# `fit_metadata(fit)` collects these seven accessors, so what the fit actually
# did is programmatically inspectable instead of being a sentence in `show`.
# The prose caveat printed above and the values reported here are derived from
# the SAME predicate (`has_dyad_dependent`), so they cannot drift apart.

estimand(::ERGMResult) = :ergm

"""
    objective(result::ERGMResult) -> Symbol

`:pseudolikelihood` for an MPLE fit (dyadwise conditionals multiplied as if
independent) and `:mc_likelihood` for MCMLE (a Monte-Carlo approximation to the
likelihood). Part of the shared result-metadata protocol.

# Example
```julia
using ERGM
fit = fit_ergm(load_dataset(:florentine_marriage), [Edges()])
objective(fit)      # :pseudolikelihood
```
"""
objective(result::ERGMResult) =
    result.method === :mple  ? :pseudolikelihood :
    result.method === :mcmle ? :mc_likelihood    : :unspecified

"""
    is_exact(result::ERGMResult) -> Bool

`true` only for a **converged** MPLE fit of a **dyad-independent** formula:
there the dyadwise conditionals are the model's conditionals, so the
pseudo-likelihood is the likelihood and the MPLE is the exact MLE. The very
same estimator applied to a formula containing any dyad-dependent term
(`Triangle`, `GWESP`, `Mutual`, ...) reports `false`, and so does a fit whose
pseudo-likelihood has no finite maximum — perfect separation (`converged ==
false`: the coefficients are a point on an asymptote, not an estimate) or a
coefficient fixed at `±Inf` by a boundary statistic (R's `drop`): an
extended-value estimate is not the exact MLE. MCMLE is always `false` — it
maximizes a Monte-Carlo approximation to the likelihood.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
is_exact(fit_ergm(net, [Edges(), NodeCov(:wealth)]))   # true — MPLE is the MLE here
is_exact(fit_ergm(net, [Edges(), Triangle()]))         # false — dyad-dependent
```
"""
is_exact(result::ERGMResult) =
    result.method === :mple && result.converged && all(isfinite, result.coefficients) &&
    !has_dyad_dependent(result.model)

"""
    se_method(result::ERGMResult) -> Symbol

`:hessian` (inverse observed information of the objective), `:bootstrap`
(parametric bootstrap), or `:fisher` — the `se_type == :mcmc` case, where the
covariance is the inverse Fisher information estimated from the MCMC sample
**plus** the Monte-Carlo component of the estimate (`V + V·Σ_mc·V`, Hunter &
Handcock 2006 §3.3; the pure Fisher part is `result.vcov_fisher`, the added
standard error is [`mcmc_se`](@ref)).

# Example
```julia
using ERGM
fit = fit_ergm(load_dataset(:florentine_marriage), [Edges()])
se_method(fit)      # :hessian
```
"""
se_method(result::ERGMResult) =
    result.se_type === :mcmc ? :fisher : result.se_type

missing_method(result::ERGMResult) = result.missing_method

function approximations(result::ERGMResult)
    out = String[]
    if result.method === :mple && has_dyad_dependent(result.model)
        push!(out, "maximum pseudo-likelihood of a dyad-dependent formula: the " *
                   "dyad conditionals are multiplied as if independent, so the " *
                   "point estimates are biased in finite samples")
        result.se_type === :hessian &&
            push!(out, "inverse-Hessian standard errors of the naive " *
                       "pseudo-likelihood: expected anticonservative under " *
                       "dyadic dependence")
    end
    if result.method === :mcmle
        push!(out, "MCMLE: the likelihood is approximated by an MCMC sample, so " *
                   "the estimates carry Monte-Carlo error (included in the " *
                   "standard errors; see mcmc_se)")
        if isnan(result.loglik)
            push!(out, "log-likelihood not estimated (bridge_rungs=0): AIC/BIC " *
                       "are NaN")
        else
            push!(out, "the reported log-likelihood (and AIC/BIC) is a " *
                       "path-sampling bridge estimate from a dyad-independent " *
                       "reference model")
        end
    end
    result.missing_method === :condition_on_face &&
        push!(out, "masked (unobserved) dyads were held fixed at their stored " *
                   "face values throughout MCMC — a different estimand from " *
                   "missing-data maximum likelihood")
    result.missing_method === :mle &&
        push!(out, "missing-data MLE: the target E[g(Y) | Y_obs] is itself a " *
                   "Monte-Carlo estimate from the constrained chain over the " *
                   "masked dyads, so it carries Monte-Carlo error of its own " *
                   "(included in the standard errors; see mcmc_se)")
    # Non-convergence is part of what the fit actually did, so it is reported
    # here as well as warned about at fit time (never only in a log line).
    result.converged || push!(out, _nonconvergence_caveat(result))
    fixed = _fixed_coefficient_note(result)
    fixed === nothing || push!(out, fixed)
    excluded = _boot_exclusion_note(result)
    excluded === nothing || push!(out, excluded)
    return out
end

# A coefficient fixed at ∓Inf by a boundary statistic (R's `drop`): said in
# `show` and in `approximations`, from the coefficients themselves
function _fixed_coefficient_note(result::ERGMResult)
    names = result.model.formula.terms.names
    lo = [names[k] for k in eachindex(names) if result.coefficients[k] == -Inf]
    hi = [names[k] for k in eachindex(names) if result.coefficients[k] == Inf]
    isempty(lo) && isempty(hi) && return nothing
    parts = String[]
    isempty(lo) || push!(parts, "$(join(lo, ", ")) fixed at -Inf (observed statistic " *
                                "at its smallest attainable value)")
    isempty(hi) || push!(parts, "$(join(hi, ", ")) fixed at +Inf (observed statistic " *
                                "at its largest attainable value)")
    return "coefficient(s) " * join(parts, "; ") * ": no finite estimate exists; " *
           "the other coefficients are estimated on the dyads these terms do not " *
           "touch, as R ergm does (drop=TRUE)"
end

# StatsAPI interface: methods on the shared statistics generics, so results
# interoperate with StatsBase/GLM-style tooling (`coef(fit)`, `aic(fit)`, ...)

"""
    _n_dyads(model::ERGMModel) -> Int

Number of observed free dyads in the model's network (ordered pairs for
directed networks, unordered pairs otherwise), excluding dyads masked as
missing via `Networks.set_missing_dyad!` — their tie status is unobserved,
so they are not observations. `public`: the variants size their samplers
with it; it is what `nobs(fit)` returns.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
model = ERGMModel(ERGMFormula([Edges()]), net)
ERGM._n_dyads(model)          # 120
set_missing_dyad!(net, 3, 4)
ERGM._n_dyads(model)          # 119
```
"""
function _n_dyads(model::ERGMModel)
    n = Int(nv(model.network))
    total = is_directed(model) ? n * (n - 1) : n * (n - 1) ÷ 2
    return total - n_missing_dyads(model.network)
end

"""
    coef(result::ERGMResult) -> Vector{Float64}

Estimated coefficients of the fitted model (a method of `StatsAPI.coef`),
in the order of `result.model.formula.terms.names`.

# Example
```julia
using ERGM
fit = fit_ergm(load_dataset(:florentine_marriage), [Edges(), NodeCov(:wealth)])
round.(coef(fit); digits=3)      # [-2.595, 0.011]
```
"""
StatsAPI.coef(result::ERGMResult) = result.coefficients

"""
    stderror(result::ERGMResult) -> Vector{Float64}

Standard errors of the coefficient estimates (a method of
`StatsAPI.stderror`); their type is recorded in `result.se_type`
(`:hessian`, `:bootstrap`, or `:mcmc` — the last includes the Monte-Carlo
component [`mcmc_se`](@ref)).

# Example
```julia
using ERGM
fit = fit_ergm(load_dataset(:florentine_marriage), [Edges(), NodeCov(:wealth)])
round.(stderror(fit); digits=3)      # [0.536, 0.005]
```
"""
StatsAPI.stderror(result::ERGMResult) = result.std_errors

"""
    vcov(result::ERGMResult) -> Matrix{Float64}

Variance-covariance matrix of the coefficient estimates (a method of
`StatsAPI.vcov`); `sqrt.(diag(vcov(fit))) == stderror(fit)`.

# Example
```julia
using ERGM, LinearAlgebra
fit = fit_ergm(load_dataset(:florentine_marriage), [Edges(), NodeCov(:wealth)])
size(vcov(fit))                                   # (2, 2)
sqrt.(diag(vcov(fit))) ≈ stderror(fit)            # true
```
"""
StatsAPI.vcov(result::ERGMResult) = result.vcov
StatsAPI.loglikelihood(result::ERGMResult) = result.loglik
StatsAPI.aic(result::ERGMResult) = result.aic
StatsAPI.bic(result::ERGMResult) = result.bic
StatsAPI.nobs(result::ERGMResult) = _n_dyads(result.model)
# R's `logLik.ergm` df: a coefficient fixed at ∓Inf by a boundary statistic
# (R's drop) is not an estimated parameter
StatsAPI.dof(result::ERGMResult) = count(isfinite, result.coefficients)

"""
    confint(result::ERGMResult; level=0.95) -> Matrix{Float64}

Normal-theory (Wald) confidence limits `θ̂ ± z_{(1+level)/2} · se`, one row
per coefficient with the lower limit in column 1 and the upper in column 2
(a method of `StatsAPI.confint`). The standard errors are the ones the fit
reports — `result.se_type` says whether they are inverse-Hessian,
parametric-bootstrap or MCMC-based — so for an MPLE fit of a dyad-dependent
model the intervals inherit the anticonservative pseudo-likelihood SEs
(see [`mple`](@ref)).

# Example
```julia
using ERGM
fit = fit_ergm(load_dataset(:florentine_marriage), [Edges(), NodeCov(:wealth)])
ci = confint(fit)               # 2×2
all(ci[:, 1] .< coef(fit) .< ci[:, 2])   # true
confint(fit; level=0.9)         # narrower
```
"""
function StatsAPI.confint(result::ERGMResult; level::Real=0.95)
    0 < level < 1 || throw(ArgumentError("confint: level must be in (0, 1) (got $level)"))
    q = quantile(Normal(), 1 - (1 - level) / 2)
    θ, se = result.coefficients, result.std_errors
    return hcat(θ .- q .* se, θ .+ q .* se)
end

"""
    coeftable(result::ERGMResult) -> Networks.CoefficientTable

The R-style coefficient table (`Estimate`, `Std.Error`, `z value`,
`Pr(>|z|)`) as an inspectable `Networks.CoefficientTable` — exactly the table
`show(result)` prints, built from the same vectors (a method of
`StatsAPI.coeftable`). Rows can be read by index or by term name.

# Example
```julia
using ERGM
fit = fit_ergm(load_dataset(:florentine_marriage), [Edges(), NodeCov(:wealth)])
tbl = coeftable(fit)
tbl["edges"].estimate == coef(fit)[1]     # true
tbl[2].p_value == fit.p_values[2]         # true
```
"""
StatsAPI.coeftable(result::ERGMResult) =
    CoefficientTable(result.model.formula.terms.names, result.coefficients,
                     result.std_errors; z_values=result.z_values,
                     p_values=result.p_values)
