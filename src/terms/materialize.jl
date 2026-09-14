"""
Model-construction-time term processing.

Two concerns that run once, when an `ERGMModel` is built (the term traits
they act on are the public protocol in `src/terms/traits.jl`):

1. **Formula validation** — fail loudly at model construction on user errors
   that would otherwise produce silently wrong fits: terms whose declared
   vertex/edge attributes ([`required_vertex_attributes`](@ref),
   [`required_edge_attributes`](@ref)) do not exist on the network, declared
   vertex attributes that are not set on EVERY vertex (statnet refuses NA
   attribute values; a silently zero-filled vertex is a wrong design
   column), and terms declaring a direction requirement
   ([`requires_directed`](@ref), [`requires_undirected`](@ref)) that the
   network does not meet. Because the checks read *traits*, a term from any
   package — not just ERGM's built-ins — participates in them.
2. **Attribute snapshotting** — attribute-based nodal terms are converted
   into "materialized" twins that hold dense, concretely typed vectors of
   the attribute values, so change statistics evaluated millions of times in
   the MPLE/MCMC hot loops no longer read the untyped
   `Dict{Symbol, Dict{T, Any}}` attribute storage.
"""

# ============================================================================
# Formula validation
# ============================================================================

function _missing_attribute_error(t, kind::String, attr::Symbol, available)
    avail_str = isempty(available) ? "(none)" :
        join(sort!([":" * String(a) for a in available]), ", ")
    throw(ArgumentError(
        "term '$(name(t))' refers to $kind attribute :$attr, which does " *
        "not exist on the network. Available $kind attributes: $avail_str."))
end

# The vertices (1:nv) with no value for `attr` — absent from the attribute
# Dict, or stored as `missing`/`nothing`, which is what an NA becomes on the
# way in from R/DataFrames.
function _vertices_without_value(net, attr::Symbol)
    raw = get_vertex_attribute(net, attr)
    return [v for v in 1:Int(nv(net)) if _no_value(get(raw, v, nothing))]
end
_no_value(val) = val === nothing || val === missing

function _incomplete_attribute_error(t, attr::Symbol, absent::Vector{Int}, n::Int)
    shown = length(absent) > 10 ?
        join(absent[1:10], ", ") * ", … ($(length(absent) - 10) more)" :
        join(absent, ", ")
    throw(ArgumentError(
        "term '$(name(t))' needs vertex attribute :$attr on every vertex, but " *
        "$(length(absent)) of $n vertices have no value (vertices $shown). " *
        "statnet refuses NA attribute values; set a value for every vertex or " *
        "drop the term."))
end

"""
    _validate_formula(terms::TermSet, net) -> nothing

Validate every term of a formula against the observed network, through the
term-trait protocol alone: a declared vertex/edge attribute must exist (and
a vertex attribute must have a value on every vertex — statnet refuses NA),
and a direction requirement must be met. Throws an `ArgumentError` naming
the term, the attribute/direction and the fix; called by the `ERGMModel`
constructor so user errors surface before any fitting. `public`: TERGM.jl
and the other variants validate their panels/layers with it rather than
re-implementing the checks.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
ERGM._validate_formula(TermSet([Edges(), NodeCov(:wealth)]), net)   # nothing
try
    ERGM._validate_formula(TermSet([NodeCov(:welth)]), net)
catch e
    occursin(":welth", e.msg)   # true — names the missing attribute
end
```
"""
function _validate_formula(terms::TermSet, net)
    available = list_vertex_attributes(net)
    available_edge = list_edge_attributes(net)
    n = Int(nv(net))
    for t in terms
        for attr in required_vertex_attributes(t)
            attr in available ||
                _missing_attribute_error(t, "vertex", attr, available)
            absent = _vertices_without_value(net, attr)
            isempty(absent) || _incomplete_attribute_error(t, attr, absent, n)
        end
        for attr in required_edge_attributes(t)
            attr in available_edge ||
                _missing_attribute_error(t, "edge", attr, available_edge)
        end
        if requires_directed(t) && !is_directed(net)
            throw(ArgumentError(
                "term '$(name(t))' is only defined for directed networks, but the " *
                "network is undirected. Remove the term or use a directed network " *
                "(R ergm raises the same error)."))
        end
        if requires_undirected(t) && is_directed(net)
            throw(ArgumentError(_undirected_only_message(t)))
        end
        _check_covariate_size(t, net)
    end
    return nothing
end

# A dyadic covariate matrix built for another network (wrong size, a subset,
# a transposed data frame) used to surface as a raw BoundsError at fit time;
# it is a formula error like any other and gets the same ArgumentError.
_check_covariate_size(t, net) = nothing
function _check_covariate_size(t::EdgeCov, net)
    n = Int(nv(net))
    r, c = size(t.covariate)
    r == n || throw(ArgumentError(
        "term '$(name(t))' has a $(r)×$(c) covariate matrix but the network has " *
        "$n vertices; EdgeCov needs an n×n matrix in vertex order (row i, column " *
        "j = the covariate of dyad (i, j))."))
    return nothing
end

# ============================================================================
# Materialized nodal terms
# ============================================================================
#
# Each materialized term wraps its source term (for the name and parameters)
# plus a dense typed snapshot of the attribute taken at model construction.
# Statistics and change statistics reproduce the wrapped term's semantics
# exactly — including its treatment of vertices with missing values — and
# are verified against the originals in the test suite.

# Integer coding of an arbitrary-valued categorical attribute: equal values
# (by `isequal`) share a code. A vertex without a value would get code 0 —
# and the `ci != 0` guards in `_matches`/`_mismatches`/`_mix_matches` below
# then keep it out of every count — but that path is UNREACHABLE through
# `ERGMModel`: `_validate_formula` has already refused a formula whose
# attribute is not set on every vertex (statnet refuses NA values; see
# `_incomplete_attribute_error`). The guards stay as defensive code only.
function _attribute_codes(net, attr::Symbol)
    raw = get_vertex_attribute(net, attr)
    n = Int(nv(net))
    codes = zeros(Int, n)
    code_of = Dict{Any, Int}()
    for v in 1:n
        val = get(raw, v, nothing)
        val === nothing && continue
        codes[v] = get!(code_of, val, length(code_of) + 1)
    end
    return codes, code_of
end

# Dense Float64 snapshot of a numeric attribute. No `default=`: a vertex
# without a value is refused by `_validate_formula` before this runs, and a
# zero-fill here would be exactly the silently wrong design column that check
# exists to prevent.
function _float_attribute_vector(term, net, attr::Symbol)
    try
        return vertex_attribute_vector(net, attr, Float64)
    catch err
        # Name the offending value and its type instead of leaking the inner
        # convert MethodError: the canonical slip is a String-valued
        # attribute (`:name`) in a numeric-covariate term.
        raw = get_vertex_attribute(net, attr)
        bad = findfirst(v -> !(get(raw, v, nothing) isa Real), 1:Int(nv(net)))
        example = bad === nothing ? "" :
            " — e.g. vertex $bad has $(repr(raw[bad])) of type $(typeof(raw[bad]))"
        throw(ArgumentError(
            "term '$(name(term))' needs a NUMERIC vertex attribute, but :$attr " *
            "has values that cannot be converted to Float64$example. Use a " *
            "numeric covariate here; for a categorical attribute use NodeFactor, " *
            "NodeMatch or NodeMix instead."))
    end
end

"Materialized [`NodeFactor`](@ref): per-vertex level-match flags."
struct MaterializedNodeFactor <: NodalTerm
    base::NodeFactor
    active::Vector{Bool}
end

"""
    _materialize(term, net) -> term or Vector of terms

Turn a term into its model-construction form: attribute terms snapshot the
vertex attribute into typed vectors (`MaterializedNodeCov`, ...) so the hot
loops never touch the attribute Dicts; multi-level `NodeFactor`/`NodeMix`
and multi-degree `Degree`/`IDegree`/`ODegree` expand into one statistic per
level/cell/degree (a `Vector`); every other term is returned unchanged.
Names and values are preserved (verified against the raw terms in the test
suite). `public`: the variants materialize their own layers/panels with it.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
ERGM._materialize(Edges(), net) === Edges()                       # true
m = ERGM._materialize(NodeCov(:wealth), net)
compute(m, net) == compute(NodeCov(:wealth), net)                 # true
length(ERGM._materialize(Degree(0:2), net))                       # 3
```
"""
_materialize(term::AbstractERGMTerm, net) = term

function _materialize(term::NodeFactor, net)
    # A multi-level term expands into one materialized statistic per
    # included level (statnet's nodefactor semantics: levels are resolved
    # from the network at model construction, first level dropped by
    # default — see the NodeFactor docstring)
    if term.level === nothing
        return [_materialize(NodeFactor(term.attr; level=l), net)
                for l in _nodefactor_levels(term, net)]
    end
    raw = get_vertex_attribute(net, term.attr)
    active = [begin
                  val = get(raw, v, nothing)
                  !isnothing(val) && val == term.level
              end
              for v in 1:Int(nv(net))]
    return MaterializedNodeFactor(term, active)
end

function compute(term::MaterializedNodeFactor, net)
    count = 0.0
    directed = is_directed(net)
    for v in vertices(net)
        term.active[v] || continue
        if directed
            count += length(inneighbors(net, v)) + length(outneighbors(net, v))
        else
            count += length(neighbors(net, v))
        end
    end
    return count
end

change_stat(term::MaterializedNodeFactor, net, i::Int, j::Int) =
    Float64(term.active[i] + term.active[j])

"Materialized [`NodeCov`](@ref): pre-transformed Float64 values per vertex."
struct MaterializedNodeCov <: NodalTerm
    base::NodeCov
    vals::Vector{Float64}
end

function _materialize(term::NodeCov, net)
    vals = _float_attribute_vector(term, net, term.attr)
    return MaterializedNodeCov(term, _transform_value.(vals, term.transform))
end

function compute(term::MaterializedNodeCov, net)
    total = 0.0
    # Two separate accumulations, matching NodeCov's compute exactly
    # (floating-point summation order included)
    for e in edges(net)
        total += term.vals[src(e)]
        total += term.vals[dst(e)]
    end
    return total
end

change_stat(term::MaterializedNodeCov, net, i::Int, j::Int) =
    term.vals[i] + term.vals[j]

"""
Materialized [`NodeMatch`](@ref): integer-coded attribute values
(0 = missing). `target == 0` counts every within-level match (uniform
homophily); `target > 0` counts matches on that level only; `target == -1`
is a `diff=true` level absent from the network (statistic constantly 0).
"""
struct MaterializedNodeMatch <: NodalTerm
    base::NodeMatch
    codes::Vector{Int}
    target::Int
end

function _materialize(term::NodeMatch, net)
    codes, code_of = _attribute_codes(net, term.attr)
    target = term.diff ? get(code_of, term.level, -1) : 0
    return MaterializedNodeMatch(term, codes, target)
end

@inline function _matches(term::MaterializedNodeMatch, i::Int, j::Int)
    ci = term.codes[i]
    return ci != 0 && ci == term.codes[j] && (term.target == 0 || ci == term.target)
end

function compute(term::MaterializedNodeMatch, net)
    count = 0.0
    for e in edges(net)
        _matches(term, Int(src(e)), Int(dst(e))) && (count += 1.0)
    end
    return count
end

change_stat(term::MaterializedNodeMatch, net, i::Int, j::Int) =
    _matches(term, i, j) ? 1.0 : 0.0

"Materialized [`NodeMismatch`](@ref): integer-coded values (0 = missing)."
struct MaterializedNodeMismatch <: NodalTerm
    base::NodeMismatch
    codes::Vector{Int}
end

function _materialize(term::NodeMismatch, net)
    codes, _ = _attribute_codes(net, term.attr)
    return MaterializedNodeMismatch(term, codes)
end

@inline function _mismatches(term::MaterializedNodeMismatch, i::Int, j::Int)
    ci, cj = term.codes[i], term.codes[j]
    return ci != 0 && cj != 0 && ci != cj
end

function compute(term::MaterializedNodeMismatch, net)
    count = 0.0
    for e in edges(net)
        _mismatches(term, Int(src(e)), Int(dst(e))) && (count += 1.0)
    end
    return count
end

change_stat(term::MaterializedNodeMismatch, net, i::Int, j::Int) =
    _mismatches(term, i, j) ? 1.0 : 0.0

"""
Materialized [`NodeMix`](@ref) cell: integer-coded attribute values
(0 = missing) plus the codes of the cell's two levels (−1 for a level absent
from the network, whose statistic is constantly 0).
"""
struct MaterializedNodeMix <: NodalTerm
    base::NodeMix
    codes::Vector{Int}
    c1::Int
    c2::Int
end

function _materialize(term::NodeMix, net)
    # A multi-cell term expands into one materialized statistic per selected
    # mixing cell (statnet's nodemix semantics: cells are resolved from the
    # network at model construction, first cell dropped by default — see the
    # NodeMix docstring)
    if !_nodemix_resolved(term)
        return [_materialize(NodeMix(term.attr, l1, l2), net)
                for (l1, l2) in _nodemix_cells(term, net)]
    end
    codes, code_of = _attribute_codes(net, term.attr)
    return MaterializedNodeMix(term, codes,
                               get(code_of, term.l1, -1),
                               get(code_of, term.l2, -1))
end

@inline function _mix_matches(term::MaterializedNodeMix, directed::Bool, i::Int, j::Int)
    ci, cj = term.codes[i], term.codes[j]
    (ci == 0 || cj == 0) && return false
    ci == term.c1 && cj == term.c2 && return true
    return !directed && ci == term.c2 && cj == term.c1
end

function compute(term::MaterializedNodeMix, net)
    count = 0.0
    directed = is_directed(net)
    for e in edges(net)
        _mix_matches(term, directed, Int(src(e)), Int(dst(e))) && (count += 1.0)
    end
    return count
end

change_stat(term::MaterializedNodeMix, net, i::Int, j::Int) =
    _mix_matches(term, is_directed(net), i, j) ? 1.0 : 0.0

"Materialized [`AbsDiff`](@ref): Float64 values per vertex."
struct MaterializedAbsDiff <: NodalTerm
    base::AbsDiff
    vals::Vector{Float64}
end

_materialize(term::AbsDiff, net) =
    MaterializedAbsDiff(term, _float_attribute_vector(term, net, term.attr))

function compute(term::MaterializedAbsDiff, net)
    total = 0.0
    for e in edges(net)
        total += abs(term.vals[src(e)] - term.vals[dst(e)])^term.base.pow
    end
    return total
end

change_stat(term::MaterializedAbsDiff, net, i::Int, j::Int) =
    abs(term.vals[i] - term.vals[j])^term.base.pow

# Names, traits, and dependence classification delegate to the source term
const _MaterializedTerm = Union{MaterializedNodeFactor, MaterializedNodeCov,
                                MaterializedNodeMatch, MaterializedNodeMismatch,
                                MaterializedNodeMix, MaterializedAbsDiff}

name(term::_MaterializedTerm) = name(term.base)
name(term::_MaterializedTerm, net) = name(term.base, net)
is_dyad_dependent(term::_MaterializedTerm) = is_dyad_dependent(term.base)
requires_directed(term::_MaterializedTerm) = requires_directed(term.base)
requires_undirected(term::_MaterializedTerm) = requires_undirected(term.base)
supports_missing(term::_MaterializedTerm) = supports_missing(term.base)

# A materialized term carries its own dense snapshot of the attribute, so it
# no longer *needs* one from the network — but it still names it, so that
# re-validating a materialized formula (e.g. against a simulated copy) keeps
# the same contract as the raw term.
required_vertex_attributes(term::_MaterializedTerm) =
    required_vertex_attributes(term.base)
required_edge_attributes(term::_MaterializedTerm) =
    required_edge_attributes(term.base)

# (Terms without a materialized twin pass through unchanged — the generic
# fallback documented above. Re-materializing an already-materialized term
# is also a no-op: its snapshot stays valid for any network sharing the
# vertex set and attributes, e.g. simulated copies.)

# A multi-degree count term (statnet's `degree(0:2)`) is a specification that
# expands into one single-degree statistic per degree; a one-degree term is
# already a statistic and passes through. Nothing is snapshotted — degree
# counts read the graph, not attributes — so `_expand_degrees` is also what
# `summary_stats` uses to evaluate the specification without a model.
_materialize(term::Degree, net) = _expand_degrees(term)
_materialize(term::IDegree, net) = _expand_degrees(term)
_materialize(term::ODegree, net) = _expand_degrees(term)

_expand_degrees(term::AbstractERGMTerm) = term
for T in (:Degree, :IDegree, :ODegree)
    @eval _expand_degrees(term::$T) =
        length(term.degrees) == 1 ? term : [$T(d) for d in term.degrees]
end

# Materializing a term set flattens: terms whose statistics are resolved
# against the network (multi-level NodeFactor, multi-cell NodeMix,
# multi-degree Degree/IDegree/ODegree) expand into one materialized term per
# statistic.
# The names are resolved against the network (`name(term, net)`), so a
# directed model's `GWESP`/`GWDSP` coefficients carry R's directed labels.
function _materialize(ts::TermSet, net)
    out = AbstractERGMTerm[]
    for t in ts.terms
        m = _materialize(t, net)
        m isa AbstractVector ? append!(out, m) : push!(out, m)
    end
    return TermSet(Tuple(out), [name(t, net) for t in out])
end

"""
    _expand_terms(terms, net) -> Vector{AbstractERGMTerm}

The *specification* of a term list against `net`: the same expansion
[`_materialize`](@ref) performs — a multi-level `NodeFactor` / multi-cell
`NodeMix` into one term per level or cell, a multi-degree `Degree` /
`IDegree` / `ODegree` into one per degree, levels resolved from `net` — but
as plain single-statistic terms (`NodeFactor(:g; level=l)`,
`NodeMix(:g, a, b)`, `Degree(d)`, every other term unchanged), WITHOUT the
attribute snapshot a materialized twin carries. `terms` may be a `TermSet` or
anything [`_collect_terms`](@ref) accepts.

This is what a model that evaluates its terms on *several* networks needs:
a STERGM scores T−1 auxiliary networks (TERGM), a multilayer model each
layer (ERGMMulti). One snapshot taken at construction would freeze the
first network's attribute onto every other, so such a model stores the
specification and materializes per network. `public` for exactly those
callers; the twin's `base` field is ERGM's own layout, read here and nowhere
else.

# Example
```julia
using ERGM
net = load_dataset(:faux_mesa_high)          # :Grade has six levels
spec = ERGM._expand_terms([Edges(), NodeFactor(:Grade), Degree(0:2)], net)
length(spec)                                   # 9: 1 + 5 non-base levels + 3 degrees
all(t -> !startswith(string(nameof(typeof(t))), "Materialized"), spec)   # true: no snapshot
spec[2] isa NodeFactor && spec[2].level !== nothing   # true: one level each
[name(t) for t in ERGM._expand_terms(ERGM.TermSet([Degree(0:1)]), net)]   # ["degree0", "degree1"]
```
"""
function _expand_terms(ts::TermSet, net)
    return AbstractERGMTerm[_specification(t) for t in _materialize(ts, net).terms]
end
_expand_terms(terms, net) = _expand_terms(TermSet(_collect_terms(terms)), net)

# The plain term behind a materialized twin (its `base`), or the term itself
_specification(t::AbstractERGMTerm) =
    (hasfield(typeof(t), :base) && fieldtype(typeof(t), :base) <: AbstractERGMTerm) ?
        getfield(t, :base) : t
