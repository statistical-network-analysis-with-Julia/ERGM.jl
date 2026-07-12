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

# Directed arc a → b with the arc (mi → mj) treated as absent.
_arc_masked(net, a::Int, b::Int, mi::Int, mj::Int) =
    !(a == mi && b == mj) && has_edge(net, a, b)

# ----------------------------------------------------------------------------
# Sorted neighbor-list intersection
#
# `outneighbors`/`inneighbors`/`neighbors` on a `Network` return the sorted
# adjacency vectors of the backing `Graphs.SimpleDiGraph`, so shared-partner
# style counts can be computed by merging two sorted lists in
# O(deg(a) + deg(b)) instead of scanning all n vertices per toggle.
# ----------------------------------------------------------------------------

# Count elements common to the sorted vectors a and b, skipping the values
# s1..s4 (pass 0 for unused slots; vertex ids are >= 1).
@inline function _isect_count(a, b, s1::Int, s2::Int, s3::Int=0, s4::Int=0)
    ia, ib = 1, 1
    la, lb = length(a), length(b)
    cnt = 0
    @inbounds while ia <= la && ib <= lb
        x = a[ia]
        y = b[ib]
        if x < y
            ia += 1
        elseif y < x
            ib += 1
        else
            (x == s1 || x == s2 || x == s3 || x == s4) || (cnt += 1)
            ia += 1
            ib += 1
        end
    end
    return cnt
end

# Count elements common to the sorted vectors a and b that are > lo.
@inline function _isect_count_above(a, b, lo::Int)
    ia = searchsortedfirst(a, lo + 1)
    ib = searchsortedfirst(b, lo + 1)
    la, lb = length(a), length(b)
    cnt = 0
    @inbounds while ia <= la && ib <= lb
        x = a[ia]
        y = b[ib]
        if x < y
            ia += 1
        elseif y < x
            ib += 1
        else
            cnt += 1
            ia += 1
            ib += 1
        end
    end
    return cnt
end

# Sum f(k)::Float64 over every k common to the sorted vectors a and b,
# skipping the values s1 and s2. (An accumulating helper rather than a
# foreach so the caller's closure never reassigns a captured variable,
# which would box it.)
@inline function _sum_common(f::F, a, b, s1::Int, s2::Int) where {F}
    ia, ib = 1, 1
    la, lb = length(a), length(b)
    s = 0.0
    @inbounds while ia <= la && ib <= lb
        x = a[ia]
        y = b[ib]
        if x < y
            ia += 1
        elseif y < x
            ib += 1
        else
            (x == s1 || x == s2) || (s += f(Int(x)))
            ia += 1
            ib += 1
        end
    end
    return s
end

# Sum f(k)::Float64 over every k in the sorted union of the sorted vectors
# a and b, skipping the values s1 and s2.
@inline function _sum_union(f::F, a, b, s1::Int, s2::Int) where {F}
    ia, ib = 1, 1
    s = 0.0
    while true
        (k, ia, ib) = _union_next(a, b, ia, ib)
        k == 0 && break
        (k == s1 || k == s2) || (s += f(k))
    end
    return s
end

# Next value of the sorted union of sorted vectors a and b, starting at
# cursors (ia, ib). Returns (value, ia′, ib′); value == 0 signals exhaustion
# (vertex ids are >= 1).
@inline function _union_next(a, b, ia::Int, ib::Int)
    la, lb = length(a), length(b)
    if ia > la
        ib > lb && return (0, ia, ib)
        return (Int(@inbounds b[ib]), ia, ib + 1)
    elseif ib > lb
        return (Int(@inbounds a[ia]), ia + 1, ib)
    end
    x = Int(@inbounds a[ia])
    y = Int(@inbounds b[ib])
    x < y && return (x, ia + 1, ib)
    y < x && return (y, ia, ib + 1)
    return (x, ia + 1, ib + 1)
end

# Count elements common to the sorted unions (a1 ∪ a2) and (b1 ∪ b2),
# skipping the values s1..s4.
function _union_isect_count(a1, a2, b1, b2, s1::Int, s2::Int, s3::Int=0, s4::Int=0)
    (x, ca1, ca2) = _union_next(a1, a2, 1, 1)
    (y, cb1, cb2) = _union_next(b1, b2, 1, 1)
    cnt = 0
    while x != 0 && y != 0
        if x < y
            (x, ca1, ca2) = _union_next(a1, a2, ca1, ca2)
        elseif y < x
            (y, cb1, cb2) = _union_next(b1, b2, cb1, cb2)
        else
            (x == s1 || x == s2 || x == s3 || x == s4) || (cnt += 1)
            (x, ca1, ca2) = _union_next(a1, a2, ca1, ca2)
            (y, cb1, cb2) = _union_next(b1, b2, cb1, cb2)
        end
    end
    return cnt
end

# Number of shared partners of the dyad (a,b) under either-direction
# adjacency, evaluated with the arc (mi → mj) masked out. O(deg) via
# neighbor-list intersection; the masked endpoints are excluded from the
# merge and re-checked pointwise with the mask applied.
function _shared_partners_masked(net, a::Int, b::Int, mi::Int, mj::Int)
    if is_directed(net)
        cnt = _union_isect_count(outneighbors(net, a), inneighbors(net, a),
                                 outneighbors(net, b), inneighbors(net, b),
                                 a, b, mi, mj)
    else
        cnt = _isect_count(neighbors(net, a), neighbors(net, b), a, b, mi, mj)
    end
    for v in (mi, mj)
        (v == 0 || v == a || v == b) && continue
        if _adjacent_masked(net, a, v, mi, mj) && _adjacent_masked(net, b, v, mi, mj)
            cnt += 1
        end
    end
    return cnt
end

# Type-specific shared partners of the ordered pair (a,b) in a directed
# network, with the arc (mi → mj) masked out (statnet's dgwesp semantics):
#   :OTP  k with a→k→b   (outgoing two-path)
#   :ITP  k with b→k→a   (incoming two-path)
#   :OSP  k with a→k, b→k (outgoing shared partner)
#   :ISP  k with k→a, k→b (incoming shared partner)
function _sp_typed_masked(net, a::Int, b::Int, t::Symbol, mi::Int, mj::Int)
    if t === :OTP
        cnt = _isect_count(outneighbors(net, a), inneighbors(net, b), a, b, mi, mj)
    elseif t === :ITP
        cnt = _isect_count(outneighbors(net, b), inneighbors(net, a), a, b, mi, mj)
    elseif t === :OSP
        cnt = _isect_count(outneighbors(net, a), outneighbors(net, b), a, b, mi, mj)
    else  # :ISP
        cnt = _isect_count(inneighbors(net, a), inneighbors(net, b), a, b, mi, mj)
    end
    # The masked endpoints were excluded from the merge; re-check them
    # pointwise with the arc mask applied
    for v in (mi, mj)
        (v == 0 || v == a || v == b) && continue
        ok = if t === :OTP
            _arc_masked(net, a, v, mi, mj) && _arc_masked(net, v, b, mi, mj)
        elseif t === :ITP
            _arc_masked(net, b, v, mi, mj) && _arc_masked(net, v, a, mi, mj)
        elseif t === :OSP
            _arc_masked(net, a, v, mi, mj) && _arc_masked(net, b, v, mi, mj)
        else
            _arc_masked(net, v, a, mi, mj) && _arc_masked(net, v, b, mi, mj)
        end
        ok && (cnt += 1)
    end
    return cnt
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
        # ttriple + ctriple by iterating edges and intersecting sorted
        # adjacency lists (O(Σ_edges deg) instead of O(n³)):
        #   transitive triples: for each arc i→j, k with j→k and i→k
        #   cyclic triples: for each arc i→j, k with j→k and k→i; every
        #   3-cycle is found once per rotation, hence the ÷ 3
        ttriple = 0
        cyc3 = 0
        for e in edges(net)
            i, j = Int(src(e)), Int(dst(e))
            i == j && continue
            ttriple += _isect_count(outneighbors(net, i), outneighbors(net, j), i, j)
            cyc3 += _isect_count(outneighbors(net, j), inneighbors(net, i), i, j)
        end
        return Float64(ttriple + cyc3 ÷ 3)
    end

    # Undirected: for each edge (i,j) with i < j, count common neighbors
    # k > j, so each triangle {i < j < k} is counted exactly once (from its
    # lowest edge)
    count = 0
    for e in edges(net)
        i, j = Int(src(e)), Int(dst(e))  # canonical i <= j
        i == j && continue
        count += _isect_count_above(neighbors(net, i), neighbors(net, j), j)
    end
    return Float64(count)
end

function change_stat(::Triangle, net, i::Int, j::Int)
    if is_directed(net)
        # Change in ttriple + ctriple from adding i→j: for each third vertex
        # k, count the transitive triples in which i→j takes each of its
        # three roles, plus the cyclic triples it closes:
        #   (i→k & j→k) + (k→i & k→j) + (i→k & k→j) + (j→k & k→i)
        # Each conjunction is a sorted-list intersection over the relevant
        # adjacency lists; skipping the values i and j makes the count
        # independent of the dyad's own arcs.
        out_i = outneighbors(net, i)
        out_j = outneighbors(net, j)
        in_i = inneighbors(net, i)
        in_j = inneighbors(net, j)
        delta = _isect_count(out_i, out_j, i, j) +
                _isect_count(in_i, in_j, i, j) +
                _isect_count(out_i, in_j, i, j) +
                _isect_count(out_j, in_i, i, j)
        return Float64(delta)
    end

    # Undirected: adding (i,j) closes one triangle per shared neighbor
    return Float64(_isect_count(neighbors(net, i), neighbors(net, j), i, j))
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
    GWESP(decay=0.5; type=:OTP)

Geometrically Weighted Edgewise Shared Partners with fixed decay
(statnet's `gwesp(decay, fixed=TRUE)`).

For **undirected** networks, shared partners of an edge (i,j) are the common
neighbors of i and j; `type` is ignored.

For **directed** networks, `type` selects the shared-partner definition for
each directed edge i→j, matching statnet's `dgwesp` types (the shared
partner k is always distinct from i and j):

- `:OTP` — outgoing two-path, k with i→k→j (statnet's default for directed
  `gwesp`; also the default here)
- `:ITP` — incoming two-path, k with j→k→i
- `:OSP` — outgoing shared partner, k with i→k and j→k
- `:ISP` — incoming shared partner, k with k→i and k→j
- `:union` — k adjacent to both i and j in *either* direction. This is not a
  statnet type; it preserves this package's historical (pre-0.2) directed
  GWESP behavior and is named `"gwesp.union.fixed.<decay>"` so it can never
  be confused with statnet's OTP-based `"gwesp.fixed.<decay>"`.

# Fields
- `decay::Float64`: Decay parameter (higher = less downweighting)
- `type::Symbol`: Directed shared-partner type (see above)
"""
struct GWESP <: StructuralTerm
    decay::Float64
    type::Symbol

    function GWESP(decay::Float64=0.5; type::Symbol=:OTP)
        decay > 0 || throw(ArgumentError("decay must be positive"))
        type in (:OTP, :ITP, :OSP, :ISP, :union) ||
            throw(ArgumentError("type must be :OTP, :ITP, :OSP, :ISP, or :union"))
        new(decay, type)
    end
end

function name(term::GWESP)
    term.type === :union && return "gwesp.union.fixed.$(term.decay)"
    # statnet names its directed OTP default identically to the undirected
    # term; the other dgwesp types carry the type in the name
    term.type === :OTP && return "gwesp.fixed.$(term.decay)"
    return "gwesp.$(term.type).fixed.$(term.decay)"
end

# Weight of an edge with s shared partners: eᵅ(1 − (1 − e⁻ᵅ)ˢ)
_gwesp_weight(α::Float64, s::Integer) = exp(α) * (1 - (1 - exp(-α))^s)

function compute(term::GWESP, net)
    α = term.decay
    stat = 0.0
    typed = is_directed(net) && term.type !== :union

    for e in edges(net)
        i, j = Int(src(e)), Int(dst(e))
        i == j && continue
        esp = typed ? _sp_typed_masked(net, i, j, term.type, 0, 0) :
                      _shared_partners_masked(net, i, j, 0, 0)
        stat += _gwesp_weight(α, esp)
    end

    return stat
end

function change_stat(term::GWESP, net, i::Int, j::Int)
    α = term.decay
    w = 1 - exp(-α)

    if is_directed(net) && term.type !== :union
        return _gwesp_change_typed(term.type, net, i, j, α, w)
    end

    # Direct effect: the added edge (i,j) enters the sum with its own
    # shared-partner count (which never involves the dyad's own edges)
    delta = _gwesp_weight(α, _shared_partners_masked(net, i, j, i, j))

    # Indirect effect: adding (i,j) makes i and j adjacent, so every edge
    # (i,k) or (j,k) whose other endpoint is a shared partner of the dyad
    # gains one shared partner. An edge moving from s to s+1 shared partners
    # changes the statistic by wˢ. For directed (union-type) networks this
    # only happens when the dyad was not already adjacent via the reverse
    # edge j→i.
    if is_directed(net)
        if !has_edge(net, j, i)
            # Walk the common either-direction neighbors of i and j
            oi, ii_ = outneighbors(net, i), inneighbors(net, i)
            oj, ij_ = outneighbors(net, j), inneighbors(net, j)
            (x, ci1, ci2) = _union_next(oi, ii_, 1, 1)
            (y, cj1, cj2) = _union_next(oj, ij_, 1, 1)
            while x != 0 && y != 0
                if x < y
                    (x, ci1, ci2) = _union_next(oi, ii_, ci1, ci2)
                elseif y < x
                    (y, cj1, cj2) = _union_next(oj, ij_, cj1, cj2)
                else
                    k = x
                    if k != i && k != j
                        # Multiplicity: number of directed edges on each dyad
                        m_ik = (has_edge(net, i, k) ? 1 : 0) + (has_edge(net, k, i) ? 1 : 0)
                        m_jk = (has_edge(net, j, k) ? 1 : 0) + (has_edge(net, k, j) ? 1 : 0)
                        delta += m_ik * w^_shared_partners_masked(net, i, k, i, j)
                        delta += m_jk * w^_shared_partners_masked(net, j, k, i, j)
                    end
                    (x, ci1, ci2) = _union_next(oi, ii_, ci1, ci2)
                    (y, cj1, cj2) = _union_next(oj, ij_, cj1, cj2)
                end
            end
        end
    else
        delta += _sum_common(neighbors(net, i), neighbors(net, j), i, j) do k
            w^_shared_partners_masked(net, i, k, i, j) +
                w^_shared_partners_masked(net, j, k, i, j)
        end
    end

    return delta
end

# Add-direction GWESP change statistic for the typed directed variants.
# Adding the arc i→j has (a) a direct effect — the new edge enters the sum
# with its own type-t shared-partner count — and (b) indirect effects on the
# existing edges for which i→j completes a new type-t two-path. An edge
# moving from s to s+1 shared partners changes the statistic by wˢ. All
# shared-partner counts are evaluated with the arc i→j masked out, so the
# result is independent of the dyad's current state.
function _gwesp_change_typed(t::Symbol, net, i::Int, j::Int, α::Float64, w::Float64)
    delta = _gwesp_weight(α, _sp_typed_masked(net, i, j, t, i, j))

    if t === :OTP
        # sp(a,b) = #{k: a→k→b}. The arc i→j is the first leg of i→j→b for
        # edges (i,b) with j→b, and the second leg of a→i→j for edges (a,j)
        # with a→i.
        delta += _sum_common(outneighbors(net, i), outneighbors(net, j), i, j) do b
            w^_sp_typed_masked(net, i, b, :OTP, i, j)
        end
        delta += _sum_common(inneighbors(net, i), inneighbors(net, j), i, j) do a
            w^_sp_typed_masked(net, a, j, :OTP, i, j)
        end
    elseif t === :ITP
        # sp(a,b) = #{k: b→k→a}. The arc i→j is the first leg of i→j→a for
        # edges (a,i) with j→a, and the second leg of b→i→j for edges (j,b)
        # with b→i.
        delta += _sum_common(inneighbors(net, i), outneighbors(net, j), i, j) do a
            w^_sp_typed_masked(net, a, i, :ITP, i, j)
        end
        delta += _sum_common(outneighbors(net, j), inneighbors(net, i), i, j) do b
            w^_sp_typed_masked(net, j, b, :ITP, i, j)
        end
    elseif t === :OSP
        # sp(a,b) = #{k: a→k, b→k}. The arc i→j gives edges (i,b) with b→j a
        # new shared out-partner j, and edges (a,i) with a→j a new shared
        # out-partner j.
        delta += _sum_common(outneighbors(net, i), inneighbors(net, j), i, j) do b
            w^_sp_typed_masked(net, i, b, :OSP, i, j)
        end
        delta += _sum_common(inneighbors(net, i), inneighbors(net, j), i, j) do a
            w^_sp_typed_masked(net, a, i, :OSP, i, j)
        end
    else  # :ISP
        # sp(a,b) = #{k: k→a, k→b}. The arc i→j gives edges (j,b) with i→b a
        # new shared in-partner i, and edges (a,j) with i→a a new shared
        # in-partner i.
        delta += _sum_common(outneighbors(net, j), outneighbors(net, i), i, j) do b
            w^_sp_typed_masked(net, j, b, :ISP, i, j)
        end
        delta += _sum_common(inneighbors(net, j), outneighbors(net, i), i, j) do a
            w^_sp_typed_masked(net, a, j, :ISP, i, j)
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

"""
    GWIDegree <: StructuralTerm
    GWIDegree(decay=0.5)

Geometrically Weighted In-Degree distribution with fixed decay
(statnet's `gwidegree(decay, fixed=TRUE)`). Only defined for directed
networks; the coefficient is named `"gwideg.fixed.<decay>"` as in statnet.

# Fields
- `decay::Float64`: Decay parameter
"""
struct GWIDegree <: StructuralTerm
    decay::Float64

    function GWIDegree(decay::Float64=0.5)
        decay > 0 || throw(ArgumentError("decay must be positive"))
        new(decay)
    end
end

name(term::GWIDegree) = "gwideg.fixed.$(term.decay)"

function compute(term::GWIDegree, net)
    α = term.decay
    stat = 0.0
    for v in vertices(net)
        stat += _gwdeg_weight(α, length(inneighbors(net, v)))
    end
    return stat
end

function change_stat(term::GWIDegree, net, i::Int, j::Int)
    # Adding the arc i→j only changes j's in-degree
    α = term.decay
    deg_j = length(inneighbors(net, j)) - (has_edge(net, i, j) ? 1 : 0)
    return _gwdeg_weight(α, deg_j + 1) - _gwdeg_weight(α, deg_j)
end

"""
    GWODegree <: StructuralTerm
    GWODegree(decay=0.5)

Geometrically Weighted Out-Degree distribution with fixed decay
(statnet's `gwodegree(decay, fixed=TRUE)`). Only defined for directed
networks; the coefficient is named `"gwodeg.fixed.<decay>"` as in statnet.

# Fields
- `decay::Float64`: Decay parameter
"""
struct GWODegree <: StructuralTerm
    decay::Float64

    function GWODegree(decay::Float64=0.5)
        decay > 0 || throw(ArgumentError("decay must be positive"))
        new(decay)
    end
end

name(term::GWODegree) = "gwodeg.fixed.$(term.decay)"

function compute(term::GWODegree, net)
    α = term.decay
    stat = 0.0
    for v in vertices(net)
        stat += _gwdeg_weight(α, length(outneighbors(net, v)))
    end
    return stat
end

function change_stat(term::GWODegree, net, i::Int, j::Int)
    # Adding the arc i→j only changes i's out-degree
    α = term.decay
    deg_i = length(outneighbors(net, i)) - (has_edge(net, i, j) ? 1 : 0)
    return _gwdeg_weight(α, deg_i + 1) - _gwdeg_weight(α, deg_i)
end

# ============================================================================
# Degree count terms
# ============================================================================

"""
    Degree <: StructuralTerm
    Degree(d)

The number of vertices with degree exactly `d` (statnet's `degree(d)`,
coefficient name `"degree<d>"`). Only defined for undirected networks, as in
R ergm — use [`IDegree`](@ref) / [`ODegree`](@ref) for directed networks.

As in statnet, a vector (or range) of degrees produces one term per degree:

```julia
fit_ergm(net, [Edges(); Degree(0:2)])
```

# Fields
- `d::Int`: The degree counted
"""
struct Degree <: StructuralTerm
    d::Int

    function Degree(d::Integer)
        d >= 0 || throw(ArgumentError("d must be non-negative"))
        new(Int(d))
    end
end

Degree(ds::AbstractVector{<:Integer}) = [Degree(d) for d in ds]

name(term::Degree) = "degree$(term.d)"

function compute(term::Degree, net)
    count = 0
    for v in vertices(net)
        length(neighbors(net, v)) == term.d && (count += 1)
    end
    return Float64(count)
end

function change_stat(term::Degree, net, i::Int, j::Int)
    d = term.d
    has_ij = has_edge(net, i, j)

    # Degrees of the endpoints in the baseline state without edge (i,j);
    # adding the edge moves each endpoint from degree k to k+1
    deg_i = length(neighbors(net, i)) - (has_ij ? 1 : 0)
    deg_j = length(neighbors(net, j)) - (has_ij ? 1 : 0)
    delta = (deg_i + 1 == d) - (deg_i == d) + (deg_j + 1 == d) - (deg_j == d)
    return Float64(delta)
end

"""
    IDegree <: StructuralTerm
    IDegree(d)

The number of vertices with in-degree exactly `d` (statnet's `idegree(d)`,
coefficient name `"idegree<d>"`). Only defined for directed networks.

As in statnet, a vector (or range) of degrees produces one term per degree:
`IDegree(0:2) == [IDegree(0), IDegree(1), IDegree(2)]`.

# Fields
- `d::Int`: The in-degree counted
"""
struct IDegree <: StructuralTerm
    d::Int

    function IDegree(d::Integer)
        d >= 0 || throw(ArgumentError("d must be non-negative"))
        new(Int(d))
    end
end

IDegree(ds::AbstractVector{<:Integer}) = [IDegree(d) for d in ds]

name(term::IDegree) = "idegree$(term.d)"

function compute(term::IDegree, net)
    count = 0
    for v in vertices(net)
        length(inneighbors(net, v)) == term.d && (count += 1)
    end
    return Float64(count)
end

function change_stat(term::IDegree, net, i::Int, j::Int)
    # Adding the arc i→j only changes j's in-degree
    d = term.d
    deg_j = length(inneighbors(net, j)) - (has_edge(net, i, j) ? 1 : 0)
    return Float64((deg_j + 1 == d) - (deg_j == d))
end

"""
    ODegree <: StructuralTerm
    ODegree(d)

The number of vertices with out-degree exactly `d` (statnet's `odegree(d)`,
coefficient name `"odegree<d>"`). Only defined for directed networks.

As in statnet, a vector (or range) of degrees produces one term per degree:
`ODegree(0:2) == [ODegree(0), ODegree(1), ODegree(2)]`.

# Fields
- `d::Int`: The out-degree counted
"""
struct ODegree <: StructuralTerm
    d::Int

    function ODegree(d::Integer)
        d >= 0 || throw(ArgumentError("d must be non-negative"))
        new(Int(d))
    end
end

ODegree(ds::AbstractVector{<:Integer}) = [ODegree(d) for d in ds]

name(term::ODegree) = "odegree$(term.d)"

function compute(term::ODegree, net)
    count = 0
    for v in vertices(net)
        length(outneighbors(net, v)) == term.d && (count += 1)
    end
    return Float64(count)
end

function change_stat(term::ODegree, net, i::Int, j::Int)
    # Adding the arc i→j only changes i's out-degree
    d = term.d
    deg_i = length(outneighbors(net, i)) - (has_edge(net, i, j) ? 1 : 0)
    return Float64((deg_i + 1 == d) - (deg_i == d))
end

# ============================================================================
# Geometrically Weighted Dyadwise Shared Partners
# ============================================================================

"""
    GWDSP <: StructuralTerm
    GWDSP(decay=0.5; type=:OTP)

Geometrically Weighted Dyadwise Shared Partners with fixed decay
(statnet's `gwdsp(decay, fixed=TRUE)`): the shared-partner analogue of
[`GWESP`](@ref) summed over *all dyads* — tied or not — instead of over
edges only.

For **undirected** networks the statistic sums `eᵅ(1 − (1 − e⁻ᵅ)^dsp(i,j))`
over all unordered dyads `{i,j}`, where `dsp(i,j)` is the number of common
neighbors of i and j; `type` is ignored.

For **directed** networks, `type` selects the shared-partner definition,
matching statnet's `dgwdsp` types (the shared partner k is always distinct
from i and j). Following statnet's C implementation, the sum is over
*ordered* dyads for the two-path types and *unordered* dyads for the
symmetric shared-partner types:

- `:OTP` — outgoing two-path, k with i→k→j, summed over ordered dyads
  (statnet's default for directed networks; also the default here). For
  dyadwise shared partners `:OTP` and `:ITP` yield the same statistic.
- `:ITP` — incoming two-path, k with j→k→i, summed over ordered dyads
- `:OSP` — outgoing shared partner, k with i→k and j→k, summed over
  unordered dyads (the count is symmetric in i and j)
- `:ISP` — incoming shared partner, k with k→i and k→j, summed over
  unordered dyads
- `:union` — k adjacent to both i and j in *either* direction, summed over
  unordered dyads. This is not a statnet type; it mirrors `GWESP`'s
  `:union` and is named `"gwdsp.union.fixed.<decay>"`.

# Fields
- `decay::Float64`: Decay parameter (higher = less downweighting)
- `type::Symbol`: Directed shared-partner type (see above)
"""
struct GWDSP <: StructuralTerm
    decay::Float64
    type::Symbol

    function GWDSP(decay::Float64=0.5; type::Symbol=:OTP)
        decay > 0 || throw(ArgumentError("decay must be positive"))
        type in (:OTP, :ITP, :OSP, :ISP, :union) ||
            throw(ArgumentError("type must be :OTP, :ITP, :OSP, :ISP, or :union"))
        new(decay, type)
    end
end

function name(term::GWDSP)
    term.type === :union && return "gwdsp.union.fixed.$(term.decay)"
    # Same convention as GWESP: the directed OTP default shares the
    # undirected term's name; the other dgwdsp types carry the type
    term.type === :OTP && return "gwdsp.fixed.$(term.decay)"
    return "gwdsp.$(term.type).fixed.$(term.decay)"
end

function compute(term::GWDSP, net)
    α = term.decay
    n = Int(nv(net))
    stat = 0.0

    if is_directed(net) && term.type !== :union
        t = term.type
        if t === :OTP || t === :ITP
            # Two-path types: ordered dyads (statnet's ddsp convention)
            for i in 1:n, j in 1:n
                i == j && continue
                stat += _gwesp_weight(α, _sp_typed_masked(net, i, j, t, 0, 0))
            end
        else
            # OSP/ISP counts are symmetric in (i,j): unordered dyads
            for i in 1:n, j in (i+1):n
                stat += _gwesp_weight(α, _sp_typed_masked(net, i, j, t, 0, 0))
            end
        end
        return stat
    end

    # Undirected, or directed :union: unordered dyads with (either-direction)
    # common-neighbor counts
    for i in 1:n, j in (i+1):n
        stat += _gwesp_weight(α, _shared_partners_masked(net, i, j, 0, 0))
    end
    return stat
end

function change_stat(term::GWDSP, net, i::Int, j::Int)
    α = term.decay
    w = 1 - exp(-α)

    if is_directed(net) && term.type !== :union
        return _gwdsp_change_typed(term.type, net, i, j, w)
    end

    # Adding (i,j) makes i and j adjacent, so every dyad (i,k) with k
    # adjacent to j gains the shared partner j, and every dyad (j,k) with k
    # adjacent to i gains the shared partner i. A dyad moving from s to s+1
    # shared partners changes the statistic by wˢ. The dyad {i,j} itself is
    # unaffected — its own edge is never a shared partner. All shared-partner
    # counts are evaluated with the dyad's edge masked out, so the result is
    # independent of the dyad's current state.
    if is_directed(net)
        # :union — no change when i and j stay adjacent via the reverse arc
        has_edge(net, j, i) && return 0.0
        delta = _sum_union(outneighbors(net, j), inneighbors(net, j), i, j) do k
            w^_shared_partners_masked(net, i, k, i, j)
        end
        delta += _sum_union(outneighbors(net, i), inneighbors(net, i), i, j) do k
            w^_shared_partners_masked(net, j, k, i, j)
        end
        return delta
    end

    delta = 0.0
    for k in neighbors(net, j)
        (k == i || k == j) && continue
        delta += w^_shared_partners_masked(net, i, k, i, j)
    end
    for k in neighbors(net, i)
        (k == i || k == j) && continue
        delta += w^_shared_partners_masked(net, j, k, i, j)
    end
    return delta
end

# Add-direction GWDSP change statistic for the typed directed variants.
# Adding the arc i→j completes new type-t two-paths/shared partners for the
# dyads listed below; each affected dyad gains exactly one shared partner,
# changing the statistic by wˢ (s = its masked shared-partner count). Unlike
# GWESP there is no direct effect: the dyad (i,j)'s own count never involves
# its own arcs, and dyads contribute whether or not they are tied.
function _gwdsp_change_typed(t::Symbol, net, i::Int, j::Int, w::Float64)
    delta = 0.0
    if t === :OTP
        # Ordered dyads. i→j is the first leg of i→j→b for dyads (i,b) with
        # j→b, and the second leg of a→i→j for dyads (a,j) with a→i.
        for b in outneighbors(net, j)
            (b == i || b == j) && continue
            delta += w^_sp_typed_masked(net, i, b, :OTP, i, j)
        end
        for a in inneighbors(net, i)
            (a == i || a == j) && continue
            delta += w^_sp_typed_masked(net, a, j, :OTP, i, j)
        end
    elseif t === :ITP
        # Ordered dyads; sp(a,b) = #{k: b→k→a}. i→j is the middle-out leg of
        # x→i→j for dyads (j,x) with x→i, and the middle-in leg of i→j→y for
        # dyads (y,i) with j→y.
        for x in inneighbors(net, i)
            (x == i || x == j) && continue
            delta += w^_sp_typed_masked(net, j, x, :ITP, i, j)
        end
        for y in outneighbors(net, j)
            (y == i || y == j) && continue
            delta += w^_sp_typed_masked(net, y, i, :ITP, i, j)
        end
    elseif t === :OSP
        # Unordered dyads; sp(a,b) = #{k: a→k, b→k}. j becomes a new shared
        # out-partner of the dyad {i,u} for every u with u→j.
        for u in inneighbors(net, j)
            (u == i || u == j) && continue
            delta += w^_sp_typed_masked(net, i, u, :OSP, i, j)
        end
    else  # :ISP
        # Unordered dyads; sp(a,b) = #{k: k→a, k→b}. i becomes a new shared
        # in-partner of the dyad {j,u} for every u with i→u.
        for u in outneighbors(net, i)
            (u == i || u == j) && continue
            delta += w^_sp_typed_masked(net, j, u, :ISP, i, j)
        end
    end
    return delta
end
