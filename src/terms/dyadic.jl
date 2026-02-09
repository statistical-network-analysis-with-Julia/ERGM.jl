"""
Dyadic ERGM terms.

Terms based on dyad-level covariates.
"""

"""
    EdgeCov <: DyadicTerm

Dyad-level covariate effect.

# Fields
- `covariate::Matrix{Float64}`: n×n matrix of dyad covariates
- `name_str::String`: Name for this term
"""
struct EdgeCov <: DyadicTerm
    covariate::Matrix{Float64}
    name_str::String

    function EdgeCov(covariate::Matrix{Float64}; name::String="edgecov")
        size(covariate, 1) == size(covariate, 2) ||
            throw(ArgumentError("Covariate matrix must be square"))
        new(covariate, name)
    end
end

name(term::EdgeCov) = term.name_str

function compute(term::EdgeCov, net)
    total = 0.0
    for e in edges(net)
        i, j = src(e), dst(e)
        total += term.covariate[i, j]
    end
    return total
end

function change_stat(term::EdgeCov, net, i::Int, j::Int)
    has_ij = has_edge(net, i, j)
    delta = term.covariate[i, j]
    return has_ij ? -delta : delta
end

"""
    EdgeCov(net::Network, attr::Symbol)

Create EdgeCov term from an edge attribute.
"""
function EdgeCov(net::Network, attr::Symbol)
    n = nv(net)
    cov = zeros(n, n)
    edge_vals = get_edge_attribute(net, attr)

    for ((i, j), val) in edge_vals
        cov[i, j] = Float64(val)
    end

    return EdgeCov(cov; name="edgecov.$(attr)")
end
