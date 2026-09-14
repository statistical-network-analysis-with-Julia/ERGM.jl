"""
Dyadic ERGM terms.

Terms based on dyad-level covariates.
"""

"""
    EdgeCov <: DyadicTerm

Dyad-level covariate effect (statnet `edgecov`): the sum of the covariate
over the edges present. Built from an `n×n` matrix (`EdgeCov(M;
name="edgecov")`; undirected networks read the `(min, max)` entry) or from
an edge attribute (`EdgeCov(net, :attr)`). Dyad-independent.

The matrix must be `n×n` for the network it is used on: `ERGMModel`/
`fit_ergm`, `compute` and `change_stat` all throw an `ArgumentError` naming
the sizes otherwise (a matrix built for another network is the commonest
`edgecov` slip, and a `BoundsError` would not say so).

# Fields
- `covariate::Matrix{Float64}`: n×n matrix of dyad covariates
- `name_str::String`: Name for this term

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
w = get_vertex_attribute(net, :wealth)
M = [Float64(abs(w[i] - w[j])) for i in 1:16, j in 1:16]      # wealth gap per dyad
term = EdgeCov(M; name="edgecov.wealthgap")
compute(term, net) == compute(AbsDiff(:wealth), net)          # true
name(term)                                                    # "edgecov.wealthgap"
```
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
    _check_covariate_size(term, net)
    total = 0.0
    for e in edges(net)
        i, j = src(e), dst(e)
        total += term.covariate[i, j]
    end
    return total
end

function change_stat(term::EdgeCov, net, i::Int, j::Int)
    # A size check is one integer comparison, so the hot loop keeps it: a
    # wrongly sized matrix must never become an out-of-bounds read
    size(term.covariate, 1) == Int(nv(net)) || _check_covariate_size(term, net)
    # Add-direction change: adding edge (i,j) contributes its covariate value.
    # Undirected edges are keyed by their canonical (min, max) orientation,
    # matching how compute() iterates edges.
    if is_directed(net)
        return term.covariate[i, j]
    else
        return term.covariate[min(i, j), max(i, j)]
    end
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
