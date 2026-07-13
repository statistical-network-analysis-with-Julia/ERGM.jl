"""
The public term-trait protocol.

A term is more than `compute`/`change_stat`/`name`: it also makes *claims*
about the data it needs and the models it belongs in. Those claims are
declared as traits — small, queryable, single-method extension points — and
ERGM.jl acts on them at model construction (see `src/terms/materialize.jl`):

| trait                          | default | acted on by                        |
|:-------------------------------|:--------|:-----------------------------------|
| [`required_vertex_attributes`](@ref) | `()`    | formula validation, materialization |
| [`required_edge_attributes`](@ref)   | `()`    | formula validation                 |
| [`requires_directed`](@ref)          | `false` | formula validation                 |
| [`requires_undirected`](@ref)        | `false` | formula validation                 |
| [`is_dyad_dependent`](@ref)          | `true`  | MPLE caveat, MCMLE bridge reference |
| `Networks.supports_missing`          | `false` | missing-data contract (advisory)   |

The protocol is *public and extensible*: a term defined in another package
(TERGM.jl, ERGMMulti.jl, ERGMUserterms.jl, or a third-party package) declares
its requirements by adding methods, and thereby participates in exactly the
same formula validation as ERGM.jl's own built-in terms — no fields are
introspected and no term type is special-cased.

```julia
struct RichClub <: AbstractERGMTerm
    attr::Symbol
end

ERGM.required_vertex_attributes(t::RichClub) = (t.attr,)
ERGM.requires_directed(::RichClub)           = true
ERGM.is_dyad_dependent(::RichClub)           = true
Networks.supports_missing(::RichClub)        = false
```

`ERGMUserterms.validate_term` exercises these declarations against a network
(it checks that the attributes a term declares are the ones it actually
reads, that a direction requirement is honoured, that a dyad-independence
claim holds, and that a missing-data claim holds).

The traits were internal (`_vertex_attribute`, `_requires_directed`,
`_requires_undirected`) until v0.5. The private names remain as
backward-compatible aliases — TERGM.jl declared `ERGM._requires_directed`
methods against them — but new code should use the public names.
"""

# ============================================================================
# Attribute requirements
# ============================================================================

"""
    required_vertex_attributes(term::AbstractERGMTerm) -> Tuple{Vararg{Symbol}}

The vertex attributes the term needs on the network, as a tuple of symbols
(empty for terms that read none — the default).

A term whose statistic reads a vertex attribute **must** declare it. ERGM.jl
checks the declaration against `list_vertex_attributes(net)` when an
[`ERGMModel`](@ref) is built and throws an `ArgumentError` if the attribute
is absent, because the alternative is silent nonsense: `get_vertex_attribute`
returns an empty `Dict` for an unknown attribute, so an undeclared
attribute-based term would quietly evaluate to an all-zero design column and
a meaningless coefficient.

Declared attributes are also what makes a term eligible for attribute
*materialization* (dense typed snapshots; ERGM's own nodal terms only).

# Example
```julia
struct Homophily <: AbstractERGMTerm
    attr::Symbol
end
ERGM.required_vertex_attributes(t::Homophily) = (t.attr,)
```

See also [`required_edge_attributes`](@ref), [`requires_directed`](@ref),
[`is_dyad_dependent`](@ref).
"""
required_vertex_attributes(::AbstractERGMTerm) = ()

required_vertex_attributes(term::NodeFactor) = (term.attr,)
required_vertex_attributes(term::NodeCov) = (term.attr,)
required_vertex_attributes(term::NodeMatch) = (term.attr,)
required_vertex_attributes(term::NodeMismatch) = (term.attr,)
required_vertex_attributes(term::AbsDiff) = (term.attr,)
required_vertex_attributes(term::NodeMix) = (term.attr,)

"""
    required_edge_attributes(term::AbstractERGMTerm) -> Tuple{Vararg{Symbol}}

The edge (dyad) attributes the term needs on the network, as a tuple of
symbols (empty by default). Validated against `list_edge_attributes(net)` at
model construction exactly like [`required_vertex_attributes`](@ref).

Declare an edge attribute only when its *absence* is an error. A term that
falls back to a default weight for edges without the attribute (as
`ERGMUserterms.WeightedEdges` does) genuinely does not require it and must
not declare it — otherwise a perfectly meaningful model would be rejected.

None of ERGM.jl's built-in terms declare one: `EdgeCov` snapshots the dyadic
covariate into a matrix at construction time, so by the time the model is
built the attribute is no longer needed.
"""
required_edge_attributes(::AbstractERGMTerm) = ()

# Backward-compatible private accessor: the single vertex attribute a term
# reads, or `nothing`. Superseded by `required_vertex_attributes` (which
# supports terms needing several); kept because TERGM.jl calls it.
function _vertex_attribute(term::AbstractERGMTerm)
    attrs = required_vertex_attributes(term)
    return isempty(attrs) ? nothing : first(attrs)
end

# ============================================================================
# Direction requirements
# ============================================================================

"""
    requires_directed(term::AbstractERGMTerm) -> Bool

Whether the term is only defined for **directed** networks (`false` by
default). Model construction rejects such a term on an undirected network
with an `ArgumentError`, as R ergm does: there its statistic would be a
structural zero or a silent duplicate of another term, not a model.

Built-in terms declaring `true`: `Mutual`, `IDegree`, `ODegree`,
`GWIDegree`, `GWODegree`. TERGM.jl declares it for `Delrecip`.

See also [`requires_undirected`](@ref).
"""
requires_directed(::AbstractERGMTerm) = false
requires_directed(::Mutual) = true
requires_directed(::IDegree) = true
requires_directed(::ODegree) = true
requires_directed(::GWIDegree) = true
requires_directed(::GWODegree) = true

"""
    requires_undirected(term::AbstractERGMTerm) -> Bool

Whether the term is only defined for **undirected** networks (`false` by
default). Model construction rejects such a term on a directed network with
an `ArgumentError`, pointing at the in-/out- variant to use instead.

Built-in term declaring `true`: `Degree` (use `IDegree`/`ODegree` when the
network is directed).

See also [`requires_directed`](@ref).
"""
requires_undirected(::AbstractERGMTerm) = false
requires_undirected(::Degree) = true

# Backward-compatible private aliases. `const` (not wrapper methods) so that
# a downstream method definition on the private name — TERGM.jl has shipped
# `ERGM._requires_directed(::Delrecip) = true` since v0.1 — adds a method to
# the *same* generic the public name and formula validation dispatch on.
const _requires_directed = requires_directed
const _requires_undirected = requires_undirected

# ============================================================================
# Dependence classification
# ============================================================================

"""
    is_dyad_dependent(term::AbstractERGMTerm) -> Bool

Whether the term's change statistic depends on the state of other dyads
(`true` for e.g. `Mutual`, `Triangle`, `Kstar`, `TwoPath`, `GWESP`,
`GWDegree`) or only on exogenous covariates of the dyad itself (`false` for
`Edges` and all nodal/dyadic covariate terms).

For dyad-independent models the pseudo-likelihood *is* the likelihood, so
MPLE is exact; for models with any dyad-dependent term MPLE point estimates
can be biased and its inverse-Hessian standard errors are typically
anticonservative (see [`mple`](@ref)).

The fallback for term types this package does not know about is `true`
(the conservative answer). Packages defining their own dyad-independent
terms should add a method returning `false`.
"""
is_dyad_dependent(::AbstractERGMTerm) = true
is_dyad_dependent(::Edges) = false
is_dyad_dependent(::NodalTerm) = false
is_dyad_dependent(::DyadicTerm) = false

# ============================================================================
# Missing-data behaviour
# ============================================================================

"""
    supports_missing(term::AbstractERGMTerm) -> Bool

Whether the term's statistic **honours the missing-dyad mask**: that is,
whether `compute(term, net)` is invariant to the face value of any dyad
marked with `Networks.set_missing_dyad!`.

This is the term-level method of the ecosystem missing-data trait
(`Networks.supports_missing`; see `src/missing.jl` for how ERGM's
*estimators* declare it). It is `false` for every built-in term, and that is
the honest answer: `compute(Edges(), net)` counts a masked dyad's stored
edge at face value. The estimators are where ERGM's principled treatment
lives — MPLE drops masked dyads from the pseudo-likelihood — so a `false`
declaration here is not a defect, merely a statement of fact.

A term declares `true` only if it consults the mask itself, e.g.

```julia
function compute(t::ObservedEdges, net)
    total = 0.0
    for e in edges(net)
        is_missing_dyad(net, src(e), dst(e)) && continue
        total += 1.0
    end
    return total
end
Networks.supports_missing(::ObservedEdges) = true
```

`ERGMUserterms.validate_term` tests the declaration by masking a dyad,
flipping its face value, and requiring the statistic not to move.
"""
supports_missing(::AbstractERGMTerm) = false
