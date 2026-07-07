"""
Structural ERGM terms.

Terms based purely on network structure: edges, triangles, k-stars, etc.

All `change_stat` methods return the state-independent add-direction change
statistic `g(y⁺ᵢⱼ) − g(y⁻ᵢⱼ)`: the value of the statistic with edge (i,j)
present minus its value with the edge absent, holding the rest of the network
fixed. The returned value does not depend on whether (i,j) currently exists.
"""

# ============================================================================
# Helpers
# ============================================================================

# Float64 binomial coefficient C(n, k); 0.0 when n < k. Avoids Int overflow
# for large degrees.
function _binomial_f(n::Integer, k::Integer)
    (k < 0 || n < k) && return 0.0
    r = 1.0
    for t in 1:k
        r *= (n - k + t) / t
    end
    return r
end

# Dyad adjacency in either direction, with the directed edge (mi → mj)
# treated as absent. Used to evaluate statistics in the add-direction
# baseline state y⁻ᵢⱼ regardless of the dyad's current value.
function _adjacent_masked(net, a::Int, b::Int, mi::Int, mj::Int)
    if is_directed(net)
        fwd = has_edge(net, a, b) && !(a == mi && b == mj)
        bwd = has_edge(net, b, a) && !(b == mi && a == mj)
        return fwd || bwd
    else
        ((a == mi && b == mj) || (a == mj && b == mi)) && return false
        return has_edge(net, a, b)
    end
end

# Number of shared partners of the dyad (a,b) (either-direction adjacency),
# evaluated with edge (mi → mj) masked out.
function _shared_partners_masked(net, a::Int, b::Int, mi::Int, mj::Int)
    sp = 0
    for k in vertices(net)
        (k == a || k == b) && continue
        if _adjacent_masked(net, a, k, mi, mj) && _adjacent_masked(net, b, k, mi, mj)
            sp += 1
        end
    end
    return sp
end

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
    # Adding edge (i,j) always increases the edge count by 1
    return 1.0
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

    # Adding i→j creates a mutual dyad iff j→i exists
    return has_edge(net, j, i) ? 1.0 : 0.0
end

# ============================================================================
# Triangle
# ============================================================================

"""
    Triangle <: StructuralTerm

The number of triangles in the network.

For undirected networks this is the usual triangle count. For directed
networks it follows the statnet `triangle` definition: the number of
transitive triples plus the number of cyclic triples (ttriple + ctriple).
"""
struct Triangle <: StructuralTerm end

name(::Triangle) = "triangle"

function compute(::Triangle, net)
    if is_directed(net)
        # ttriple + ctriple:
        #   transitive triples: ordered (i,j,k) with i→j, j→k, i→k
        #   cyclic triples: cycles {i→j, j→k, k→i}, each counted once
        n = nv(net)
        ttriple = 0
        cyc3 = 0  # counts each cycle 3 times (once per rotation)
        for i in 1:n, j in 1:n
            (j == i || !has_edge(net, i, j)) && continue
            for k in 1:n
                (k == i || k == j || !has_edge(net, j, k)) && continue
                has_edge(net, i, k) && (ttriple += 1)
                has_edge(net, k, i) && (cyc3 += 1)
            end
        end
        return Float64(ttriple + cyc3 ÷ 3)
    end

    count = 0
    n = nv(net)
    for i in 1:n
        for j in outneighbors(net, i)
            j <= i && continue
            for k in outneighbors(net, i)
                k <= j && continue
                if has_edge(net, j, k)
                    count += 1
                end
            end
        end
    end

    return Float64(count)
end

function change_stat(::Triangle, net, i::Int, j::Int)
    if is_directed(net)
        # Change in ttriple + ctriple from adding i→j: for each third vertex k,
        # count the transitive triples in which i→j takes each of its three
        # roles, plus the cyclic triples it closes.
        delta = 0
        for k in vertices(net)
            (k == i || k == j) && continue
            y_ik = has_edge(net, i, k)
            y_ki = has_edge(net, k, i)
            y_jk = has_edge(net, j, k)
            y_kj = has_edge(net, k, j)
            delta += (y_ik & y_jk) + (y_ki & y_kj) + (y_ik & y_kj) + (y_jk & y_ki)
        end
        return Float64(delta)
    end

    # Undirected: adding (i,j) closes one triangle per shared neighbor
    shared = 0
    for k in vertices(net)
        (k == i || k == j) && continue
        if has_edge(net, i, k) && has_edge(net, j, k)
            shared += 1
        end
    end

    return Float64(shared)
end

# ============================================================================
# K-Star
# ============================================================================

"""
    Kstar <: StructuralTerm

The number of k-stars in the network.
A k-star consists of a central node connected to k other nodes.

For directed networks this counts out-stars (statnet's `ostar`), since
`neighbors` returns out-neighbors for directed networks.

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
        # Number of k-stars centered at v is C(deg, k)
        count += _binomial_f(deg, k)
    end

    return count
end

function change_stat(term::Kstar, net, i::Int, j::Int)
    k = term.k
    has_ij = has_edge(net, i, j)

    # Degrees in the baseline state without edge (i,j)
    deg_i = length(neighbors(net, i)) - (has_ij ? 1 : 0)
    delta = _binomial_f(deg_i + 1, k) - _binomial_f(deg_i, k)

    # Stars centered at j gain a spoke too when the network is undirected
    if !is_directed(net)
        deg_j = length(neighbors(net, j)) - (has_ij ? 1 : 0)
        delta += _binomial_f(deg_j + 1, k) - _binomial_f(deg_j, k)
    end

    return delta
end

# ============================================================================
# Two-Path
# ============================================================================

"""
    TwoPath <: StructuralTerm

The number of two-paths in the network.

For directed networks: pairs of edges (h → v, v → k) with h ≠ k
(statnet's `twopath`/`m2star`). For undirected networks: the number of
2-stars, `Σᵥ C(deg(v), 2)`.
"""
struct TwoPath <: StructuralTerm end

name(::TwoPath) = "twopath"

function compute(::TwoPath, net)
    count = 0.0
    if is_directed(net)
        for v in vertices(net)
            in_deg = length(inneighbors(net, v))
            out_deg = length(outneighbors(net, v))
            # Exclude h→v→h returns through mutual dyads
            mutual = 0
            for k in outneighbors(net, v)
                has_edge(net, k, v) && (mutual += 1)
            end
            count += in_deg * out_deg - mutual
        end
    else
        for v in vertices(net)
            count += _binomial_f(length(neighbors(net, v)), 2)
        end
    end
    return count
end

function change_stat(::TwoPath, net, i::Int, j::Int)
    if is_directed(net)
        # New two-paths: (k→i, i→j) for k ≠ j, and (i→j, j→k) for k ≠ i.
        # Neither in-degree of i nor out-degree of j involves the edge i→j.
        in_deg_i = length(inneighbors(net, i))
        out_deg_j = length(outneighbors(net, j))
        y_ji = has_edge(net, j, i) ? 1 : 0
        return Float64(in_deg_i + out_deg_j - 2 * y_ji)
    end

    # Undirected 2-stars: each endpoint gains deg⁻ new 2-stars, where deg⁻
    # is its degree without edge (i,j)
    has_ij = has_edge(net, i, j)
    deg_i = length(neighbors(net, i)) - (has_ij ? 1 : 0)
    deg_j = length(neighbors(net, j)) - (has_ij ? 1 : 0)
    return Float64(deg_i + deg_j)
end

# ============================================================================
# Geometrically Weighted Terms
# ============================================================================

"""
    GWESP <: StructuralTerm

Geometrically Weighted Edgewise Shared Partners with fixed decay
(statnet's `gwesp(decay, fixed=TRUE)`).

Shared partners are counted using either-direction adjacency. For directed
networks this differs from statnet's default outgoing-two-path (`OTP`)
definition.

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

# Weight of an edge with s shared partners: eᵅ(1 − (1 − e⁻ᵅ)ˢ)
_gwesp_weight(α::Float64, s::Integer) = exp(α) * (1 - (1 - exp(-α))^s)

function compute(term::GWESP, net)
    α = term.decay
    stat = 0.0

    for e in edges(net)
        i, j = src(e), dst(e)

        # Count shared partners
        esp = 0
        for k in vertices(net)
            (k == i || k == j) && continue
            if (has_edge(net, i, k) || has_edge(net, k, i)) &&
               (has_edge(net, j, k) || has_edge(net, k, j))
                esp += 1
            end
        end

        stat += _gwesp_weight(α, esp)
    end

    return stat
end

function change_stat(term::GWESP, net, i::Int, j::Int)
    α = term.decay
    w = 1 - exp(-α)

    # Direct effect: the added edge (i,j) enters the sum with its own
    # shared-partner count (which never involves the dyad's own edges)
    esp_ij = _shared_partners_masked(net, i, j, i, j)
    delta = _gwesp_weight(α, esp_ij)

    # Indirect effect: adding (i,j) makes i and j adjacent, so every edge
    # (i,k) or (j,k) whose other endpoint is a shared partner of the dyad
    # gains one shared partner. An edge moving from s to s+1 shared partners
    # changes the statistic by wˢ. For directed networks this only happens
    # when the dyad was not already adjacent via the reverse edge j→i.
    if !(is_directed(net) && has_edge(net, j, i))
        for k in vertices(net)
            (k == i || k == j) && continue
            if _adjacent_masked(net, i, k, i, j) && _adjacent_masked(net, j, k, i, j)
                # Multiplicity: number of (directed) edges on each dyad
                if is_directed(net)
                    m_ik = (has_edge(net, i, k) ? 1 : 0) + (has_edge(net, k, i) ? 1 : 0)
                    m_jk = (has_edge(net, j, k) ? 1 : 0) + (has_edge(net, k, j) ? 1 : 0)
                else
                    m_ik = 1
                    m_jk = 1
                end
                esp_ik = _shared_partners_masked(net, i, k, i, j)
                esp_jk = _shared_partners_masked(net, j, k, i, j)
                delta += m_ik * w^esp_ik + m_jk * w^esp_jk
            end
        end
    end

    return delta
end

"""
    GWDegree <: StructuralTerm

Geometrically Weighted Degree distribution with fixed decay
(statnet's `gwdegree(decay, fixed=TRUE)`).

For directed networks this weights out-degrees (statnet's `gwodegree`),
since `neighbors` returns out-neighbors for directed networks.

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

# Contribution of a node with degree d: eᵅ(1 − (1 − e⁻ᵅ)ᵈ)
_gwdeg_weight(α::Float64, d::Integer) = d > 0 ? exp(α) * (1 - (1 - exp(-α))^d) : 0.0

function compute(term::GWDegree, net)
    α = term.decay
    stat = 0.0

    for v in vertices(net)
        stat += _gwdeg_weight(α, length(neighbors(net, v)))
    end

    return stat
end

function change_stat(term::GWDegree, net, i::Int, j::Int)
    α = term.decay
    has_ij = has_edge(net, i, j)

    # Degree of i in the baseline state without edge (i,j)
    deg_i = length(neighbors(net, i)) - (has_ij ? 1 : 0)
    delta = _gwdeg_weight(α, deg_i + 1) - _gwdeg_weight(α, deg_i)

    # For undirected networks j's degree changes too
    if !is_directed(net) && i != j
        deg_j = length(neighbors(net, j)) - (has_ij ? 1 : 0)
        delta += _gwdeg_weight(α, deg_j + 1) - _gwdeg_weight(α, deg_j)
    end

    return delta
end
