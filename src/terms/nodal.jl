"""
Nodal attribute ERGM terms.

Terms based on vertex-level attributes: nodefactor, nodecov, etc.

All `change_stat` methods return the state-independent add-direction change
statistic `g(y⁺ᵢⱼ) − g(y⁻ᵢⱼ)` (see `src/terms/structural.jl`).
"""

"""
    NodeFactor <: NodalTerm

Main effect of a categorical vertex attribute: the number of times a vertex
with the given attribute level appears as an endpoint of an edge. As in
statnet, an edge within the level counts twice.

# Fields
- `attr::Symbol`: Vertex attribute name
- `level::Any`: Specific level (if nothing, counts all vertices with the attribute)
"""
struct NodeFactor <: NodalTerm
    attr::Symbol
    level::Any

    NodeFactor(attr::Symbol; level=nothing) = new(attr, level)
end

name(term::NodeFactor) = isnothing(term.level) ? "nodefactor.$(term.attr)" : "nodefactor.$(term.attr).$(term.level)"

function compute(term::NodeFactor, net)
    attr_vals = get_vertex_attribute(net, term.attr)
    count = 0.0

    for v in vertices(net)
        val = get(attr_vals, v, nothing)
        if !isnothing(val) && (isnothing(term.level) || val == term.level)
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

    # Adding edge (i,j) adds one endpoint appearance for each of i and j
    val_i = get(attr_vals, i, nothing)
    val_j = get(attr_vals, j, nothing)

    matches_i = !isnothing(val_i) && (isnothing(term.level) || val_i == term.level)
    matches_j = !isnothing(val_j) && (isnothing(term.level) || val_j == term.level)

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

Homophily effect: tendency for edges between vertices with matching attribute values.

# Fields
- `attr::Symbol`: Vertex attribute name
- `diff::Bool`: If true, count non-matching pairs instead
"""
struct NodeMatch <: NodalTerm
    attr::Symbol
    diff::Bool

    NodeMatch(attr::Symbol; diff::Bool=false) = new(attr, diff)
end

name(term::NodeMatch) = term.diff ? "nodemismatch.$(term.attr)" : "nodematch.$(term.attr)"

function compute(term::NodeMatch, net)
    attr_vals = get_vertex_attribute(net, term.attr)
    count = 0.0

    for e in edges(net)
        i, j = src(e), dst(e)
        val_i = get(attr_vals, i, nothing)
        val_j = get(attr_vals, j, nothing)

        if !isnothing(val_i) && !isnothing(val_j)
            matches = (val_i == val_j)
            if term.diff
                count += matches ? 0.0 : 1.0
            else
                count += matches ? 1.0 : 0.0
            end
        end
    end

    return count
end

function change_stat(term::NodeMatch, net, i::Int, j::Int)
    attr_vals = get_vertex_attribute(net, term.attr)

    val_i = get(attr_vals, i, nothing)
    val_j = get(attr_vals, j, nothing)

    if isnothing(val_i) || isnothing(val_j)
        return 0.0
    end

    matches = (val_i == val_j)
    return term.diff ? (matches ? 0.0 : 1.0) : (matches ? 1.0 : 0.0)
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
