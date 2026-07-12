"""
Nodal attribute ERGM terms.

Terms based on vertex-level attributes: nodefactor, nodecov, etc.

All `change_stat` methods return the state-independent add-direction change
statistic `g(y⁺ᵢⱼ) − g(y⁻ᵢⱼ)` (see `src/terms/structural.jl`).
"""

# Sorted distinct (non-missing) values of a vertex attribute — the levels of
# a categorical attribute in statnet's ordering (`sort(unique(...))`). Values
# that cannot be `<`-compared fall back to sorting by their string form.
function _sorted_levels(net, attr::Symbol)
    lv = unique(v for v in values(get_vertex_attribute(net, attr)) if v !== nothing)
    try
        sort!(lv)
    catch
        sort!(lv; by=string)
    end
    return lv
end

"""
    NodeFactor <: NodalTerm
    NodeFactor(attr; levels=nothing, base=1, level=nothing)

Main effect of a categorical vertex attribute (R ergm's `nodefactor`): for
each included attribute level, the number of times a vertex with that level
appears as an endpoint of an edge. As in statnet, an edge within the level
counts twice, and one statistic is produced *per included level* (named
`"nodefactor.<attr>.<level>"`); the levels are resolved from the network at
model construction, where the term expands into its per-level statistics.

Which levels are included follows statnet:

- By default the **first level is dropped** (`base=1`): the levels are the
  sorted distinct attribute values, and the lowest acts as the reference
  category. Including all levels would make the term collinear with
  [`Edges`](@ref).
- `base` gives the indices (into the sorted levels) to drop; `base=0` keeps
  every level.
- `levels` gives the included attribute *values* explicitly, in order,
  overriding `base`.
- `level=x` constructs a single-level term directly (one statistic, no
  expansion), equivalent to `levels=[x]`.

!!! warning "Changed in 0.2"
    `NodeFactor(attr)` previously produced a *single* statistic counting the
    endpoint appearances of every vertex with the attribute — collinear with
    `Edges` by construction. It now matches statnet: one statistic per
    attribute level, with the first (sorted) level dropped as the reference.
    Pass `base=0` to keep all levels.

Before expansion (e.g. `compute` on a term that has not been through model
construction), a multi-level term evaluates to the *sum* of its per-level
statistics.

# Fields
- `attr::Symbol`: Vertex attribute name
- `level::Any`: Single level counted (`nothing` for a multi-level term)
- `levels::Union{Nothing, Vector{Any}}`: Explicit included levels
- `base::Vector{Int}`: Sorted-level indices dropped (default `[1]`)
"""
struct NodeFactor <: NodalTerm
    attr::Symbol
    level::Any
    levels::Union{Nothing, Vector{Any}}
    base::Vector{Int}

    function NodeFactor(attr::Symbol; level=nothing, levels=nothing, base=1)
        if level !== nothing && levels !== nothing
            throw(ArgumentError("give either `level` (a single-level term) or " *
                                "`levels`, not both"))
        end
        base_vec = base isa Integer ? (base == 0 ? Int[] : Int[base]) :
                   Int[b for b in base]
        all(b -> b >= 1, base_vec) ||
            throw(ArgumentError("base indices must be positive (use base=0 to " *
                                "keep all levels)"))
        new(attr, level,
            levels === nothing ? nothing : collect(Any, levels), base_vec)
    end
end

name(term::NodeFactor) = isnothing(term.level) ? "nodefactor.$(term.attr)" : "nodefactor.$(term.attr).$(term.level)"

# The included levels of a multi-level NodeFactor, resolved against the
# network: the explicit `levels` if given (validated to exist), otherwise
# the sorted attribute levels minus the `base` indices.
function _nodefactor_levels(term::NodeFactor, net)
    lv = _sorted_levels(net, term.attr)
    if term.levels !== nothing
        unknown = [l for l in term.levels if !(l in lv)]
        isempty(unknown) ||
            throw(ArgumentError("nodefactor.$(term.attr): levels $(unknown) do " *
                                "not occur on the network (available: $(lv))"))
        return collect(Any, term.levels)
    end
    keep = Any[l for (k, l) in enumerate(lv) if !(k in term.base)]
    isempty(keep) &&
        throw(ArgumentError("nodefactor.$(term.attr): no levels remain after " *
                            "dropping base level(s) $(term.base) of $(lv); pass " *
                            "base=0 or levels=[...] to keep levels"))
    return keep
end

# Whether a vertex value counts toward the term (single level, or any
# included level of a multi-level term); `included` is the pre-resolved
# level set for multi-level terms, `nothing` for single-level terms.
function _nodefactor_active(term::NodeFactor, included, val)
    val === nothing && return false
    term.level === nothing || return val == term.level
    return val in included
end

_nodefactor_included(term::NodeFactor, net) =
    term.level === nothing ? Set{Any}(_nodefactor_levels(term, net)) : nothing

function compute(term::NodeFactor, net)
    attr_vals = get_vertex_attribute(net, term.attr)
    included = _nodefactor_included(term, net)
    count = 0.0

    for v in vertices(net)
        val = get(attr_vals, v, nothing)
        if _nodefactor_active(term, included, val)
            # Endpoint appearances of v: total degree (in + out for directed)
            if is_directed(net)
                count += length(inneighbors(net, v)) + length(outneighbors(net, v))
            else
                count += length(neighbors(net, v))
            end
        end
    end

    return count
end

function change_stat(term::NodeFactor, net, i::Int, j::Int)
    attr_vals = get_vertex_attribute(net, term.attr)
    included = _nodefactor_included(term, net)

    # Adding edge (i,j) adds one endpoint appearance for each of i and j
    matches_i = _nodefactor_active(term, included, get(attr_vals, i, nothing))
    matches_j = _nodefactor_active(term, included, get(attr_vals, j, nothing))

    return Float64(matches_i + matches_j)
end

"""
    NodeCov <: NodalTerm

Main effect of a continuous vertex attribute.
Sum of attribute values for endpoints of each edge.

# Fields
- `attr::Symbol`: Vertex attribute name
- `transform::Symbol`: Optional transform (:none, :log, :sqrt)
"""
struct NodeCov <: NodalTerm
    attr::Symbol
    transform::Symbol

    NodeCov(attr::Symbol; transform::Symbol=:none) = new(attr, transform)
end

name(term::NodeCov) = "nodecov.$(term.attr)"

function _transform_value(val, transform::Symbol)
    if transform == :log
        return log(val + 1)
    elseif transform == :sqrt
        return sqrt(val)
    else
        return Float64(val)
    end
end

function compute(term::NodeCov, net)
    attr_vals = get_vertex_attribute(net, term.attr)
    total = 0.0

    # edges(net) yields each edge exactly once for both directed and
    # undirected networks, so each edge contributes x_i + x_j once
    for e in edges(net)
        i, j = src(e), dst(e)
        val_i = get(attr_vals, i, 0.0)
        val_j = get(attr_vals, j, 0.0)

        total += _transform_value(val_i, term.transform)
        total += _transform_value(val_j, term.transform)
    end

    return total
end

function change_stat(term::NodeCov, net, i::Int, j::Int)
    attr_vals = get_vertex_attribute(net, term.attr)

    val_i = _transform_value(get(attr_vals, i, 0.0), term.transform)
    val_j = _transform_value(get(attr_vals, j, 0.0), term.transform)

    return val_i + val_j
end

"""
    NodeMatch <: NodalTerm
    NodeMatch(attr; diff=false, level=nothing)

Homophily effect: the number of edges whose endpoints have matching values
of the vertex attribute `attr` (R ergm's `nodematch`).

With `diff=false` (uniform homophily, the default) this is a single
statistic counting all matched edges, named `"nodematch.<attr>"`.

With `diff=true` (differential homophily, matching R's
`nodematch(attr, diff=TRUE)`) there is one statistic *per attribute level*,
each counting only the edges matched on that level and named
`"nodematch.<attr>.<level>"`. Since this package's term system is
one-statistic-per-term (like [`NodeFactor`](@ref)), construct one `NodeMatch`
per level:

```julia
levels = sort(unique(values(get_vertex_attribute(net, :group))))
terms = [NodeMatch(:group; diff=true, level=l) for l in levels]
```

!!! warning "Changed in 0.2"
    `diff=true` previously counted *mismatched* edges — a different model
    from R's `nodematch(diff=TRUE)` under the same keyword. That statistic
    is now available as [`NodeMismatch`](@ref).

# Fields
- `attr::Symbol`: Vertex attribute name
- `diff::Bool`: Differential (per-level) homophily as in R
- `level::Any`: The attribute level counted (required when `diff=true`)
"""
struct NodeMatch <: NodalTerm
    attr::Symbol
    diff::Bool
    level::Any

    function NodeMatch(attr::Symbol; diff::Bool=false, level=nothing)
        if diff && level === nothing
            throw(ArgumentError(
                "NodeMatch(attr; diff=true) is differential (per-level) homophily, as in " *
                "R's nodematch(diff=TRUE): one statistic per attribute level. The term " *
                "system is one-statistic-per-term, so construct one term per level, e.g. " *
                "[NodeMatch(:group; diff=true, level=l) for l in levels]. " *
                "For the count of mismatched edges use NodeMismatch(attr)."))
        end
        if !diff && level !== nothing
            throw(ArgumentError("level is only meaningful with diff=true"))
        end
        new(attr, diff, level)
    end
end

name(term::NodeMatch) =
    term.diff ? "nodematch.$(term.attr).$(term.level)" : "nodematch.$(term.attr)"

# Whether an i–j edge counts toward this (possibly per-level) match statistic
function _nodematch_counts(term::NodeMatch, val_i, val_j)
    (isnothing(val_i) || isnothing(val_j)) && return false
    val_i == val_j || return false
    return !term.diff || val_i == term.level
end

function compute(term::NodeMatch, net)
    attr_vals = get_vertex_attribute(net, term.attr)
    count = 0.0

    for e in edges(net)
        i, j = src(e), dst(e)
        val_i = get(attr_vals, i, nothing)
        val_j = get(attr_vals, j, nothing)
        _nodematch_counts(term, val_i, val_j) && (count += 1.0)
    end

    return count
end

function change_stat(term::NodeMatch, net, i::Int, j::Int)
    attr_vals = get_vertex_attribute(net, term.attr)

    val_i = get(attr_vals, i, nothing)
    val_j = get(attr_vals, j, nothing)

    return _nodematch_counts(term, val_i, val_j) ? 1.0 : 0.0
end

"""
    NodeMismatch <: NodalTerm
    NodeMismatch(attr)

Heterophily effect: the number of edges whose endpoints have *different*
(non-missing) values of the vertex attribute `attr`. Named
`"nodemismatch.<attr>"`.

This is the statistic that `NodeMatch(attr; diff=true)` computed before
0.2; it has no R ergm equivalent under the `nodematch` name (R's
`diff=TRUE` is per-level homophily — see [`NodeMatch`](@ref)).

# Fields
- `attr::Symbol`: Vertex attribute name
"""
struct NodeMismatch <: NodalTerm
    attr::Symbol
end

name(term::NodeMismatch) = "nodemismatch.$(term.attr)"

function compute(term::NodeMismatch, net)
    attr_vals = get_vertex_attribute(net, term.attr)
    count = 0.0

    for e in edges(net)
        i, j = src(e), dst(e)
        val_i = get(attr_vals, i, nothing)
        val_j = get(attr_vals, j, nothing)
        if !isnothing(val_i) && !isnothing(val_j) && val_i != val_j
            count += 1.0
        end
    end

    return count
end

function change_stat(term::NodeMismatch, net, i::Int, j::Int)
    attr_vals = get_vertex_attribute(net, term.attr)

    val_i = get(attr_vals, i, nothing)
    val_j = get(attr_vals, j, nothing)

    return (!isnothing(val_i) && !isnothing(val_j) && val_i != val_j) ? 1.0 : 0.0
end

"""
    AbsDiff <: NodalTerm

Absolute difference in a continuous attribute between edge endpoints.

# Fields
- `attr::Symbol`: Vertex attribute name
- `pow::Float64`: Power to raise the difference to (default 1)
"""
struct AbsDiff <: NodalTerm
    attr::Symbol
    pow::Float64

    AbsDiff(attr::Symbol; pow::Float64=1.0) = new(attr, pow)
end

name(term::AbsDiff) = "absdiff.$(term.attr)"

function compute(term::AbsDiff, net)
    attr_vals = get_vertex_attribute(net, term.attr)
    total = 0.0

    for e in edges(net)
        i, j = src(e), dst(e)
        val_i = Float64(get(attr_vals, i, 0.0))
        val_j = Float64(get(attr_vals, j, 0.0))

        total += abs(val_i - val_j)^term.pow
    end

    return total
end

function change_stat(term::AbsDiff, net, i::Int, j::Int)
    attr_vals = get_vertex_attribute(net, term.attr)

    val_i = Float64(get(attr_vals, i, 0.0))
    val_j = Float64(get(attr_vals, j, 0.0))

    return abs(val_i - val_j)^term.pow
end

"""
    NodeMix <: NodalTerm
    NodeMix(attr; levels=nothing, levels2=-1)
    NodeMix(attr, l1, l2)

Mixing-matrix cell counts for a categorical vertex attribute (R ergm's
`nodemix`): one statistic per mixing cell, counting the edges whose endpoint
levels fall in that cell, named `"mix.<attr>.<l1>.<l2>"`.

The cells follow statnet's level ordering. With sorted levels `u₁ < u₂ < …`:

- **undirected** networks use the unordered cells `(uᵢ, uⱼ)` with `i ≤ j`,
  in column-major order: `(u₁,u₁), (u₁,u₂), (u₂,u₂), (u₁,u₃), …`
- **directed** networks use all ordered (tail-level, head-level) cells in
  column-major order: `(u₁,u₁), (u₂,u₁), …, (u₁,u₂), …`

and statnet's reference handling: by default the **first cell is dropped**
(`levels2=-1`), acting as the reference category. `levels2` selects cells by
index into the ordered cell list — all-negative values drop those cells,
all-positive values keep exactly those cells (in the given order), and
`levels2=0` keeps every cell. `levels` restricts (and orders) the attribute
levels used to build the cells.

Like [`NodeFactor`](@ref), the cells are resolved from the network at model
construction, where the term expands into its per-cell statistics;
`NodeMix(attr, l1, l2)` constructs a single-cell term directly. Before
expansion, a multi-cell term evaluates to the sum of its selected cells.

# Fields
- `attr::Symbol`: Vertex attribute name
- `l1::Any`, `l2::Any`: The cell's levels (`nothing` for a multi-cell term);
  for directed networks `l1` is the tail (sender) level and `l2` the head
  (receiver) level
- `levels::Union{Nothing, Vector{Any}}`: Levels used to build the cells
- `levels2::Vector{Int}`: Cell selection (see above)
"""
struct NodeMix <: NodalTerm
    attr::Symbol
    l1::Any
    l2::Any
    levels::Union{Nothing, Vector{Any}}
    levels2::Vector{Int}

    NodeMix(attr::Symbol, l1, l2) = new(attr, l1, l2, nothing, Int[])

    function NodeMix(attr::Symbol; levels=nothing, levels2=-1)
        sel = levels2 isa Integer ? (levels2 == 0 ? Int[] : Int[levels2]) :
              Int[x for x in levels2]
        (isempty(sel) || all(>(0), sel) || all(<(0), sel)) ||
            throw(ArgumentError("levels2 must be all positive (cells to keep) " *
                                "or all negative (cells to drop)"))
        new(attr, nothing, nothing,
            levels === nothing ? nothing : collect(Any, levels), sel)
    end
end

# A term constructed as NodeMix(attr, l1, l2) counts a single cell; the
# keyword form is a multi-cell term resolved at model construction.
_nodemix_resolved(term::NodeMix) = term.l1 !== nothing || term.l2 !== nothing

name(term::NodeMix) = _nodemix_resolved(term) ?
    "mix.$(term.attr).$(term.l1).$(term.l2)" : "mix.$(term.attr)"

# The selected mixing cells of a multi-cell NodeMix, resolved against the
# network: statnet-ordered (level, level) pairs after levels2 selection.
function _nodemix_cells(term::NodeMix, net)
    if term.levels === nothing
        lv = _sorted_levels(net, term.attr)
    else
        avail = _sorted_levels(net, term.attr)
        unknown = [l for l in term.levels if !(l in avail)]
        isempty(unknown) ||
            throw(ArgumentError("mix.$(term.attr): levels $(unknown) do not " *
                                "occur on the network (available: $(avail))"))
        lv = collect(Any, term.levels)
    end
    L = length(lv)

    cells = Tuple{Any, Any}[]
    if is_directed(net)
        for cj in 1:L, ci in 1:L
            push!(cells, (lv[ci], lv[cj]))
        end
    else
        for cj in 1:L, ci in 1:cj
            push!(cells, (lv[ci], lv[cj]))
        end
    end

    sel = term.levels2
    if isempty(sel)
        selected = cells
    else
        # `ncells` (never rebound) is what the closure captures; rebinding
        # `cells` itself while captured would box it (Core.Box).
        ncells = length(cells)
        all(s -> 1 <= abs(s) <= ncells, sel) ||
            throw(ArgumentError("mix.$(term.attr): levels2 indices must lie in " *
                                "1:$(ncells) (got $(sel))"))
        if sel[1] > 0
            selected = cells[sel]
        else
            drop = Set(-s for s in sel)
            selected = [c for (k, c) in enumerate(cells) if !(k in drop)]
        end
    end
    isempty(selected) &&
        throw(ArgumentError("mix.$(term.attr): no mixing cells selected"))
    return selected
end

# Whether an edge (i-level, j-level) pair counts toward a single cell. For
# undirected networks the cell is an unordered pair of levels.
function _nodemix_counts(term::NodeMix, directed::Bool, val_i, val_j)
    (val_i === nothing || val_j === nothing) && return false
    val_i == term.l1 && val_j == term.l2 && return true
    return !directed && val_i == term.l2 && val_j == term.l1
end

function compute(term::NodeMix, net)
    if !_nodemix_resolved(term)
        return sum(compute(NodeMix(term.attr, l1, l2), net)
                   for (l1, l2) in _nodemix_cells(term, net))
    end

    attr_vals = get_vertex_attribute(net, term.attr)
    directed = is_directed(net)
    count = 0.0
    for e in edges(net)
        val_i = get(attr_vals, Int(src(e)), nothing)
        val_j = get(attr_vals, Int(dst(e)), nothing)
        _nodemix_counts(term, directed, val_i, val_j) && (count += 1.0)
    end
    return count
end

function change_stat(term::NodeMix, net, i::Int, j::Int)
    if !_nodemix_resolved(term)
        return sum(change_stat(NodeMix(term.attr, l1, l2), net, i, j)
                   for (l1, l2) in _nodemix_cells(term, net))
    end

    attr_vals = get_vertex_attribute(net, term.attr)
    val_i = get(attr_vals, i, nothing)
    val_j = get(attr_vals, j, nothing)
    return _nodemix_counts(term, is_directed(net), val_i, val_j) ? 1.0 : 0.0
end
