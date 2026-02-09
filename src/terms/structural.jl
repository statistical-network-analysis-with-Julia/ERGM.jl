"""
Structural ERGM terms.

Terms based purely on network structure: edges, triangles, k-stars, etc.
"""

# ============================================================================
# Edges
# ============================================================================

"""
    Edges <: StructuralTerm

The number of edges in the network.
The most basic ERGM term, analogous to an intercept.
"""
struct Edges <: StructuralTerm end

name(::Edges) = "edges"

function compute(::Edges, net)
    return Float64(ne(net))
end

function change_stat(::Edges, net, i::Int, j::Int)
    # Adding an edge increases count by 1, removing decreases by 1
    return has_edge(net, i, j) ? -1.0 : 1.0
end

# ============================================================================
# Mutual
# ============================================================================

"""
    Mutual <: StructuralTerm

The number of mutual (reciprocated) dyads.
Only meaningful for directed networks.
"""
struct Mutual <: StructuralTerm end

name(::Mutual) = "mutual"

function compute(::Mutual, net)
    if !is_directed(net)
        return 0.0
    end

    count = 0
    for i in vertices(net)
        for j in outneighbors(net, i)
            if j > i && has_edge(net, j, i)
                count += 1
            end
        end
    end
    return Float64(count)
end

function change_stat(::Mutual, net, i::Int, j::Int)
    if !is_directed(net)
        return 0.0
    end

    # Toggling (i,j) affects mutual count only if (j,i) exists
    has_ij = has_edge(net, i, j)
    has_ji = has_edge(net, j, i)

    if has_ji
        # If j→i exists, toggling i→j changes mutual count
        return has_ij ? -1.0 : 1.0
    else
        return 0.0
    end
end

# ============================================================================
# Triangle
# ============================================================================

"""
    Triangle <: StructuralTerm

The number of triangles in the network.
"""
struct Triangle <: StructuralTerm end

name(::Triangle) = "triangle"

function compute(::Triangle, net)
    count = 0
    n = nv(net)

    for i in 1:n
        for j in outneighbors(net, i)
            j <= i && continue
            for k in outneighbors(net, i)
                k <= j && continue
                if has_edge(net, j, k) || has_edge(net, k, j)
                    count += 1
                end
            end
        end
    end

    return Float64(count)
end

function change_stat(::Triangle, net, i::Int, j::Int)
    # Count shared neighbors - each becomes part of a new/removed triangle
    shared = 0
    for k in vertices(net)
        k == i && continue
        k == j && continue

        # Check if k is connected to both i and j
        connected_to_i = has_edge(net, i, k) || has_edge(net, k, i)
        connected_to_j = has_edge(net, j, k) || has_edge(net, k, j)

        if connected_to_i && connected_to_j
            shared += 1
        end
    end

    return has_edge(net, i, j) ? -Float64(shared) : Float64(shared)
end

# ============================================================================
# K-Star
# ============================================================================

"""
    Kstar <: StructuralTerm

The number of k-stars in the network.
A k-star consists of a central node connected to k other nodes.

# Fields
- `k::Int`: Star size (number of spokes)
"""
struct Kstar <: StructuralTerm
    k::Int

    function Kstar(k::Int)
        k >= 2 || throw(ArgumentError("k must be at least 2"))
        new(k)
    end
end

name(term::Kstar) = "kstar$(term.k)"

function compute(term::Kstar, net)
    k = term.k
    count = 0.0

    for v in vertices(net)
        deg = length(neighbors(net, v))
        if deg >= k
            # Number of k-stars centered at v is C(deg, k)
            count += binomial(deg, k)
        end
    end

    return count
end

function change_stat(term::Kstar, net, i::Int, j::Int)
    k = term.k
    has_ij = has_edge(net, i, j)

    delta = 0.0

    # Effect on stars centered at i
    deg_i = length(neighbors(net, i))
    if has_ij
        # Removing edge: lose C(deg_i, k) - C(deg_i - 1, k) stars at i
        if deg_i >= k
            delta -= binomial(deg_i, k) - binomial(deg_i - 1, k)
        end
    else
        # Adding edge: gain C(deg_i + 1, k) - C(deg_i, k) stars at i
        if deg_i >= k - 1
            delta += binomial(deg_i + 1, k) - binomial(deg_i, k)
        end
    end

    # Effect on stars centered at j (if undirected)
    if !is_directed(net)
        deg_j = length(neighbors(net, j))
        if has_ij
            if deg_j >= k
                delta -= binomial(deg_j, k) - binomial(deg_j - 1, k)
            end
        else
            if deg_j >= k - 1
                delta += binomial(deg_j + 1, k) - binomial(deg_j, k)
            end
        end
    end

    return delta
end

# ============================================================================
# Two-Path
# ============================================================================

"""
    TwoPath <: StructuralTerm

The number of two-paths (i → j → k) in the network.
"""
struct TwoPath <: StructuralTerm end

name(::TwoPath) = "twopath"

function compute(::TwoPath, net)
    count = 0.0
    for j in vertices(net)
        in_deg = length(inneighbors(net, j))
        out_deg = length(outneighbors(net, j))
        count += in_deg * out_deg
    end
    return count
end

function change_stat(::TwoPath, net, i::Int, j::Int)
    has_ij = has_edge(net, i, j)

    # Toggling i→j affects two-paths through j (i→j→k) and through i (h→i→j)
    in_deg_i = length(inneighbors(net, i))
    out_deg_j = length(outneighbors(net, j))

    # Two-paths created/destroyed: (predecessors of i) + (successors of j)
    delta = Float64(in_deg_i + out_deg_j)

    return has_ij ? -delta : delta
end

# ============================================================================
# Geometrically Weighted Terms
# ============================================================================

"""
    GWESP <: StructuralTerm

Geometrically Weighted Edgewise Shared Partners.

# Fields
- `decay::Float64`: Decay parameter (higher = less downweighting)
"""
struct GWESP <: StructuralTerm
    decay::Float64

    function GWESP(decay::Float64=0.5)
        decay > 0 || throw(ArgumentError("decay must be positive"))
        new(decay)
    end
end

name(term::GWESP) = "gwesp.fixed.$(term.decay)"

function compute(term::GWESP, net)
    α = term.decay
    stat = 0.0

    for e in edges(net)
        i, j = src(e), dst(e)

        # Count shared partners
        esp = 0
        for k in vertices(net)
            k == i && continue
            k == j && continue

            if (has_edge(net, i, k) || has_edge(net, k, i)) &&
               (has_edge(net, j, k) || has_edge(net, k, j))
                esp += 1
            end
        end

        if esp > 0
            stat += exp(α) * (1 - (1 - exp(-α))^esp)
        end
    end

    return stat
end

function change_stat(term::GWESP, net, i::Int, j::Int)
    α = term.decay
    has_ij = has_edge(net, i, j)

    # Count current shared partners
    esp = 0
    for k in vertices(net)
        k == i && continue
        k == j && continue

        if (has_edge(net, i, k) || has_edge(net, k, i)) &&
           (has_edge(net, j, k) || has_edge(net, k, j))
            esp += 1
        end
    end

    direct_effect = esp > 0 ? exp(α) * (1 - (1 - exp(-α))^esp) : 0.0

    # Also affects other edges through shared partners
    # Simplified: only count direct effect
    return has_ij ? -direct_effect : direct_effect
end

"""
    GWDegree <: StructuralTerm

Geometrically Weighted Degree distribution.

# Fields
- `decay::Float64`: Decay parameter
"""
struct GWDegree <: StructuralTerm
    decay::Float64

    function GWDegree(decay::Float64=0.5)
        decay > 0 || throw(ArgumentError("decay must be positive"))
        new(decay)
    end
end

name(term::GWDegree) = "gwdegree.fixed.$(term.decay)"

function compute(term::GWDegree, net)
    α = term.decay
    stat = 0.0

    for v in vertices(net)
        deg = length(neighbors(net, v))
        if deg > 0
            stat += exp(α) * (1 - (1 - exp(-α))^deg)
        end
    end

    return stat
end

function change_stat(term::GWDegree, net, i::Int, j::Int)
    α = term.decay
    has_ij = has_edge(net, i, j)
    delta = 0.0

    # Effect on degree of i
    deg_i = length(neighbors(net, i))
    if has_ij
        # Removing: change from deg_i to deg_i - 1
        old_contrib = deg_i > 0 ? exp(α) * (1 - (1 - exp(-α))^deg_i) : 0.0
        new_contrib = deg_i > 1 ? exp(α) * (1 - (1 - exp(-α))^(deg_i - 1)) : 0.0
        delta += new_contrib - old_contrib
    else
        # Adding: change from deg_i to deg_i + 1
        old_contrib = deg_i > 0 ? exp(α) * (1 - (1 - exp(-α))^deg_i) : 0.0
        new_contrib = exp(α) * (1 - (1 - exp(-α))^(deg_i + 1))
        delta += new_contrib - old_contrib
    end

    # Effect on degree of j (if undirected)
    if !is_directed(net) && i != j
        deg_j = length(neighbors(net, j))
        if has_ij
            old_contrib = deg_j > 0 ? exp(α) * (1 - (1 - exp(-α))^deg_j) : 0.0
            new_contrib = deg_j > 1 ? exp(α) * (1 - (1 - exp(-α))^(deg_j - 1)) : 0.0
            delta += new_contrib - old_contrib
        else
            old_contrib = deg_j > 0 ? exp(α) * (1 - (1 - exp(-α))^deg_j) : 0.0
            new_contrib = exp(α) * (1 - (1 - exp(-α))^(deg_j + 1))
            delta += new_contrib - old_contrib
        end
    end

    return delta
end
