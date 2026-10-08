"""
Further statnet terms that published models use: `concurrent`, `gwnsp`,
`esp`, `degrange`/`idegrange`/`odegrange`, `meandeg`, `density`,
`triadcensus`, `sender` and `receiver`.

Every term follows R ergm 4's definition and label (pinned against
`summary()` in the provenanced `ergm_terms.toml` fixture) and the package's
add-direction change-statistic convention (pinned against brute force).
Terms that span several statistics in R — `esp(0:2)`, `triadcensus`,
`sender`, `receiver` — are, like `Degree(0:2)` and a multi-level `NodeFactor`, one
*specification* that expands into one statistic per level when the model is
built (`materialize`, so `summary_stats` and every fit see the expansion).
"""

# ============================================================================
# concurrent
# ============================================================================

"""
    Concurrent <: StructuralTerm
    Concurrent()

The number of vertices with degree two or more — the number of actors with
concurrent partnerships (statnet's `concurrent`, coefficient name
`"concurrent"`; Morris & Kretzschmar's concurrency term of the tergm/msm
models). Undirected networks only, as in R: the model is refused on a
directed network, and so are `compute`/`change_stat`.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
compute(Concurrent(), net)     # 11.0 — R: summary(flomarriage ~ concurrent)
name(Concurrent())             # "concurrent"
```
"""
struct Concurrent <: StructuralTerm end

name(::Concurrent) = "concurrent"

function compute(term::Concurrent, net)
    _refuse_directed(term, net)
    c = 0
    for v in vertices(net)
        length(neighbors(net, v)) >= 2 && (c += 1)
    end
    return Float64(c)
end

function change_stat(term::Concurrent, net, i::Int, j::Int)
    _refuse_directed(term, net)
    has_ij = has_edge(net, i, j)
    # Each endpoint moves from degree k (without the tie) to k + 1; it
    # becomes concurrent exactly when k == 1
    di = length(neighbors(net, i)) - has_ij
    dj = length(neighbors(net, j)) - has_ij
    return Float64((di == 1) + (dj == 1))
end

requires_undirected(::Concurrent) = true
_directed_variant_hint(::Concurrent) =
    "a directed degree term such as `ODegree` / `IDegree` (R ergm has no directed concurrent)"

# ============================================================================
# gwnsp
# ============================================================================

"""
    GWNSP <: StructuralTerm
    GWNSP(decay=0.5; type=:OTP)

Geometrically weighted **non-edgewise** shared partners with fixed `decay`
(statnet's `gwnsp(decay, fixed=TRUE)` / `dgwnsp(decay, fixed=TRUE, type=)`):
the geometrically weighted count of shared partners over the dyads that are
NOT ties, i.e. `GWDSP(decay) − GWESP(decay)` for the same shared-partner
`type` (`:OTP`, `:ITP`, `:OSP`, `:ISP` on a directed network; ignored on an
undirected one). Coefficient names follow R: `"gwnsp.fixed.0.5"` on an
undirected network, `"gwnsp.OTP.fixed.0.5"` on a directed one
(`name(term, net)`).

# Fields
- `decay::Float64`: Decay parameter, ≥ 0
- `type::Symbol`: Directed shared-partner type

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
compute(GWNSP(0.5), net)   # 36.1804... — R: summary(flomarriage ~ gwnsp(0.5, fixed=TRUE))
compute(GWNSP(0.0), net)   # 35.0 — non-tied dyads with ≥ 1 shared partner
compute(GWNSP(0.5), net) ≈ compute(GWDSP(0.5), net) - compute(GWESP(0.5), net)   # true
```
"""
struct GWNSP <: StructuralTerm
    decay::Float64
    type::Symbol

    function GWNSP(decay::Real=0.5; type::Symbol=:OTP)
        type in (:OTP, :ITP, :OSP, :ISP) ||
            throw(ArgumentError("type must be :OTP, :ITP, :OSP, or :ISP"))
        new(_check_decay(Float64(decay)), type)
    end
end

function name(term::GWNSP)
    d = _decay_label(term.decay)
    term.type === :OTP && return "gwnsp.fixed.$d"
    return "gwnsp.$(term.type).fixed.$d"
end

function name(term::GWNSP, net)
    d = _decay_label(term.decay)
    is_directed(net) || return "gwnsp.fixed.$d"
    return "gwnsp.$(term.type).fixed.$d"
end

# (the ESP part goes through the weight kernel directly: `GWESP(...)` is not a
# type-stable constructor since `fixed=false` returns the curved term)
compute(t::GWNSP, net) =
    compute(GWDSP(t.decay; type=t.type), net) - _gwesp_compute(_GWKernel(t.decay), t.type, net)

change_stat(t::GWNSP, net, i::Int, j::Int) =
    change_stat(GWDSP(t.decay; type=t.type), net, i, j) -
    _gwesp_change(_GWKernel(t.decay), t.type, net, i, j)

# ============================================================================
# esp
# ============================================================================

# Implemented from the definition in R ergm's documentation of `esp`/`desp`
# (the number of edges with exactly k shared partners, for the shared-partner
# types listed there) and validated against R's `summary()` output
# (`ergm_terms.toml`, section (i)) and by brute force; no ergm source was
# consulted. The statistic is GWESP's edgewise shared-partner sum with the
# weight w(s) = 1[s = k] in place of the geometric weight, so it runs on the
# same shared-partner code (`_gwesp_compute`, `_gwesp_change`) through a
# weight kernel: the change statistic of a dyad is its own count's weight
# plus, for every tie whose shared-partner count the dyad raises from s to
# s + 1, the increment w(s + 1) − w(s).
struct _ESPKernel
    k::Int
end
@inline _spw(K::_ESPKernel, s::Integer) = s == K.k ? 1.0 : 0.0
@inline _spinc(K::_ESPKernel, s::Integer) = (s + 1 == K.k ? 1.0 : 0.0) - (s == K.k ? 1.0 : 0.0)

"""
    ESP <: StructuralTerm
    ESP(k; type=:OTP)
    ESP(ks; type=:OTP)

Edgewise shared partners (statnet's `esp(d)` / `desp(d, type=)`): the number
of ties with exactly `k` shared partners. On an undirected network the
shared partners of a tie {i,j} are the common neighbours of i and j; on a
directed network `type` selects R's shared-partner configuration of the arc
i→j, as for [`GWESP`](@ref): `:OTP` (i→v→j, the default), `:ITP` (j→v→i),
`:OSP` (i→v and j→v) or `:ISP` (v→i and v→j). R's fifth type, `"RTP"`, is
not implemented and is refused.

As in statnet, a vector (or range) of counts is **one term that expands into
one statistic per count** when the model is built, like [`Degree`](@ref):
`ESP(0:2)` becomes `esp0`, `esp1`, `esp2`. The unexpanded term is a
specification, not a statistic — `compute(ESP(0:2), net)` throws; use
[`summary_stats`](@ref) or build the model. Coefficient names are R's:
`"esp<k>"` on an undirected network and `"esp.<type><k>"` on a directed one
(`name(term, net)`; the one-argument `name` gives the undirected label for
`:OTP`). The counts must be distinct and non-negative.

`ESP(k)` is the `k`-th term of the shared-partner distribution that
[`GWESP`](@ref) weights geometrically: `GWESP(α) = Σₖ eᵅ(1 − (1 − e⁻ᵅ)ᵏ)·ESP(k)`.
Its attainable range is R's, `(0, Inf)`: an observed count of 0 fixes the
coefficient at `-Inf` (R's `drop=TRUE`).

# Fields
- `ks::Vector{Int}`: The shared-partner count(s)
- `type::Symbol`: Directed shared-partner type (ignored on an undirected network)

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
compute(ESP(0), net)                       # 12.0 — R: summary(flomarriage ~ esp(0))
summary_stats(net, [ESP(0:2)])             # (esp0 = 12.0, esp1 = 7.0, esp2 = 1.0)
name(ESP(1))                               # "esp1"
dnet = network(3; directed=true); add_edge!(dnet, 1, 2); add_edge!(dnet, 2, 3); add_edge!(dnet, 1, 3)
name(ESP(1), dnet)                         # "esp.OTP1" — R's directed label
compute(ESP(1), dnet)                      # 1.0 — the arc 1→3 has the two-path 1→2→3
```
"""
struct ESP <: StructuralTerm
    ks::Vector{Int}
    type::Symbol

    function ESP(ks::AbstractVector{<:Integer}; type::Symbol=:OTP)
        type === :RTP && throw(ArgumentError(
            "ESP: the reciprocated two-path type \"RTP\" of R's desp is not " *
            "implemented (as for GWESP); use :OTP, :ITP, :OSP or :ISP"))
        type in (:OTP, :ITP, :OSP, :ISP) ||
            throw(ArgumentError("ESP: type must be :OTP, :ITP, :OSP or :ISP (got $(repr(type)))"))
        isempty(ks) && throw(ArgumentError("ESP needs at least one shared-partner count"))
        all(>=(0), ks) || throw(ArgumentError(
            "ESP: shared-partner counts must be non-negative (got $(collect(ks)))"))
        allunique(ks) || throw(ArgumentError(
            "ESP: shared-partner counts must be distinct (got $(collect(ks))), as in R's esp(d)"))
        new(Int[k for k in ks], type)
    end
end

ESP(k::Integer; type::Symbol=:OTP) = ESP(Int[k]; type=type)

function name(t::ESP)
    pre = t.type === :OTP ? "esp" : "esp.$(t.type)"
    return length(t.ks) == 1 ? "$pre$(t.ks[1])" : "$pre($(join(t.ks, ",")))"
end

# R ergm 4.12: `esp` on a directed network is labelled with its type
# (`esp.OTP1`), on an undirected one without (`esp1`)
function name(t::ESP, net)
    pre = is_directed(net) ? "esp.$(t.type)" : "esp"
    return length(t.ks) == 1 ? "$pre$(t.ks[1])" : "$pre($(join(t.ks, ",")))"
end

@inline function _single_esp(t::ESP)
    length(t.ks) == 1 && return @inbounds t.ks[1]
    _multi_esp_error(t)
end

@noinline _multi_esp_error(t::ESP) = throw(ArgumentError(
    "term '$(name(t))' spans $(length(t.ks)) shared-partner counts and is not a " *
    "single statistic; it expands to one ESP(k) per count when the model is built. " *
    "Expand via ERGMModel (fit_ergm / summary_stats do this for you), or call " *
    "compute on each ESP(k) separately."))

compute(t::ESP, net) = _gwesp_compute(_ESPKernel(_single_esp(t)), t.type, net)
change_stat(t::ESP, net, i::Int, j::Int) =
    _gwesp_change(_ESPKernel(_single_esp(t)), t.type, net, i, j)

# A multi-count term is a specification: one statistic per count
materialize(t::ESP, net) =
    length(t.ks) == 1 ? t : AbstractERGMTerm[ESP(k; type=t.type) for k in t.ks]

# ============================================================================
# degrange / idegrange / odegrange
# ============================================================================

function _check_degrange(kind::String, from, to)
    from >= 0 || throw(ArgumentError("$kind: from must be non-negative (got $from)"))
    to > from || throw(ArgumentError("$kind: to must exceed from (got from=$from, to=$to)"))
    return Int(from), (isinf(to) ? typemax(Int) : Int(to))
end

_degrange_label(prefix::String, from::Int, to::Int) =
    to == typemax(Int) ? "$prefix$(from)+" : "$(prefix)$(from)to$(to)"

@inline _in_range(d::Int, from::Int, to::Int) = from <= d < to

"""
    DegRange <: StructuralTerm
    DegRange(from, to=Inf)

The number of vertices whose degree `d` satisfies `from ≤ d < to`
(statnet's `degrange(from, to)`; coefficient name `"deg<from>+"` when `to`
is infinite, `"deg<from>to<to>"` otherwise). Undirected networks only — use
[`IDegRange`](@ref) / [`ODegRange`](@ref) on a directed one.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
compute(DegRange(2), net)       # 11.0 — degree ≥ 2
compute(DegRange(1, 3), net)    # 6.0  — degree 1 or 2
name(DegRange(1, 3))            # "deg1to3"
```
"""
struct DegRange <: StructuralTerm
    from::Int
    to::Int
    DegRange(from::Real, to::Real=Inf) = new(_check_degrange("DegRange", from, to)...)
end

"""
    IDegRange <: StructuralTerm
    IDegRange(from, to=Inf)

The number of vertices whose in-degree `d` satisfies `from ≤ d < to`
(statnet's `idegrange`; `"ideg<from>+"` / `"ideg<from>to<to>"`). Directed
networks only.

# Example
```julia
using ERGM
net = network(4; directed=true); add_edges!(net, [(1, 3), (2, 3), (4, 3), (3, 1)])
compute(IDegRange(2), net)      # 1.0 — vertex 3 receives three arcs
name(IDegRange(1, 2))           # "ideg1to2"
```
"""
struct IDegRange <: StructuralTerm
    from::Int
    to::Int
    IDegRange(from::Real, to::Real=Inf) = new(_check_degrange("IDegRange", from, to)...)
end

"""
    ODegRange <: StructuralTerm
    ODegRange(from, to=Inf)

The number of vertices whose out-degree `d` satisfies `from ≤ d < to`
(statnet's `odegrange`; `"odeg<from>+"` / `"odeg<from>to<to>"`). Directed
networks only.

# Example
```julia
using ERGM
net = network(4; directed=true); add_edges!(net, [(1, 2), (1, 3), (1, 4), (2, 1)])
compute(ODegRange(2), net)      # 1.0 — vertex 1 sends three arcs
compute(ODegRange(1, 2), net)   # 1.0 — vertex 2 sends one
```
"""
struct ODegRange <: StructuralTerm
    from::Int
    to::Int
    ODegRange(from::Real, to::Real=Inf) = new(_check_degrange("ODegRange", from, to)...)
end

name(t::DegRange) = _degrange_label("deg", t.from, t.to)
name(t::IDegRange) = _degrange_label("ideg", t.from, t.to)
name(t::ODegRange) = _degrange_label("odeg", t.from, t.to)

function compute(t::DegRange, net)
    _refuse_directed(t, net)
    return Float64(count(v -> _in_range(length(neighbors(net, v)), t.from, t.to),
                         vertices(net)))
end
compute(t::IDegRange, net) =
    Float64(count(v -> _in_range(length(inneighbors(net, v)), t.from, t.to), vertices(net)))
compute(t::ODegRange, net) =
    Float64(count(v -> _in_range(length(outneighbors(net, v)), t.from, t.to), vertices(net)))

# A vertex whose degree moves from k (tie absent) to k + 1 enters the range
# when k + 1 == from and leaves it when k + 1 == to
@inline _range_step(k::Int, from::Int, to::Int) =
    Float64(_in_range(k + 1, from, to) - _in_range(k, from, to))

function change_stat(t::DegRange, net, i::Int, j::Int)
    _refuse_directed(t, net)
    has_ij = has_edge(net, i, j)
    di = length(neighbors(net, i)) - has_ij
    dj = length(neighbors(net, j)) - has_ij
    return _range_step(di, t.from, t.to) + _range_step(dj, t.from, t.to)
end

function change_stat(t::IDegRange, net, i::Int, j::Int)
    k = length(inneighbors(net, j)) - has_edge(net, i, j)
    return _range_step(k, t.from, t.to)
end

function change_stat(t::ODegRange, net, i::Int, j::Int)
    k = length(outneighbors(net, i)) - has_edge(net, i, j)
    return _range_step(k, t.from, t.to)
end

requires_undirected(::DegRange) = true
requires_directed(::IDegRange) = true
requires_directed(::ODegRange) = true
_directed_variant_hint(t::DegRange) =
    "`ODegRange` / `IDegRange` (statnet odegrange/idegrange)"

# ============================================================================
# meandeg / density
# ============================================================================

"""
    MeanDeg <: StructuralTerm
    MeanDeg()

The mean vertex degree (statnet's `meandeg`, coefficient name
`"meandeg"`): `2·edges/n` on an undirected network and `edges/n` on a
directed one (mean out-degree), so its coefficient is the `edges`
coefficient on a size-invariant scale. Dyad-independent.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
compute(MeanDeg(), net)        # 2.5 — 2·20/16
```
"""
struct MeanDeg <: StructuralTerm end

"""
    Density <: StructuralTerm
    Density()

The network density, `edges / n_dyads` (statnet's `density`, coefficient
name `"density"`), with `n_dyads = n(n−1)/2` undirected and `n(n−1)`
directed. Dyad-independent.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
compute(Density(), net)        # 0.1667 — 20/120
```
"""
struct Density <: StructuralTerm end

name(::MeanDeg) = "meandeg"
name(::Density) = "density"

_degree_scale(net) = is_directed(net) ? 1.0 / nv(net) : 2.0 / nv(net)
_all_dyads(net) = is_directed(net) ? nv(net) * (nv(net) - 1) : nv(net) * (nv(net) - 1) / 2

compute(::MeanDeg, net) = ne(net) * _degree_scale(net)
change_stat(::MeanDeg, net, i::Int, j::Int) = _degree_scale(net)
compute(::Density, net) = ne(net) / _all_dyads(net)
change_stat(::Density, net, i::Int, j::Int) = 1.0 / _all_dyads(net)

is_dyad_dependent(::MeanDeg) = false
is_dyad_dependent(::Density) = false

# ============================================================================
# triadcensus
# ============================================================================

# R's triad-type order (sna's triad.census / ergm's triadcensus): the 16
# MAN types of a directed triad, and the 4 edge counts of an undirected one
const _DIRECTED_TRIAD_TYPES = ("003", "012", "102", "021D", "021U", "021C", "111D",
                               "111U", "030T", "030C", "201", "120D", "120U", "120C",
                               "210", "300")
const _UNDIRECTED_TRIAD_TYPES = ("0", "1", "2", "3")

# The MAN type index (0-based, R's order) of the directed triad on (a, b, c)
# whose six arcs are given; built once into a 64-entry lookup table.
function _classify_directed_triad(ab, ba, ac, ca, bc, cb)
    arcs = ((1, 2, ab), (2, 1, ba), (1, 3, ac), (3, 1, ca), (2, 3, bc), (3, 2, cb))
    has(x, y) = any(t -> t[1] == x && t[2] == y && t[3], arcs)
    pairs = ((1, 2), (1, 3), (2, 3))
    m = count(p -> has(p...) && has(reverse(p)...), pairs)
    a = count(p -> has(p...) ⊻ has(reverse(p)...), pairs)
    outd(v) = count(w -> w != v && has(v, w), 1:3)
    ind(v) = count(w -> w != v && has(w, v), 1:3)
    code = if (m, a) == (0, 0)
        "003"
    elseif (m, a) == (0, 1)
        "012"
    elseif (m, a) == (1, 0)
        "102"
    elseif (m, a) == (0, 2)
        # out-star "021D" (A<-B->C), in-star "021U" (A->B<-C), chain "021C"
        any(v -> outd(v) == 2, 1:3) ? "021D" : any(v -> ind(v) == 2, 1:3) ? "021U" : "021C"
    elseif (m, a) == (1, 1)
        # the asymmetric arc points INTO the mutual pair: "111D" (A<->B<-C);
        # out of it: "111U" (A<->B->C)
        mp = first(p for p in pairs if has(p...) && has(reverse(p)...))
        third = only(setdiff(1:3, mp))
        any(v -> has(third, v), mp) ? "111D" : "111U"
    elseif (m, a) == (0, 3)
        any(v -> outd(v) == 2, 1:3) ? "030T" : "030C"
    elseif (m, a) == (2, 0)
        "201"
    elseif (m, a) == (1, 2)
        # the third vertex sends to both members of the mutual pair: "120D";
        # receives from both: "120U"; otherwise a chain through it: "120C"
        mp = first(p for p in pairs if has(p...) && has(reverse(p)...))
        third = only(setdiff(1:3, mp))
        all(v -> has(third, v), mp) ? "120D" : all(v -> has(v, third), mp) ? "120U" : "120C"
    elseif (m, a) == (2, 1)
        "210"
    else
        "300"
    end
    return findfirst(==(code), _DIRECTED_TRIAD_TYPES) - 1
end

const _TRIAD_TABLE = Tuple(_classify_directed_triad(((b >> k) & 1 == 1 for k in 0:5)...)
                           for b in 0:63)

# Type index of the directed triad (a, b, c) with the arc (mi → mj) forced
# to `state` (true = present)
@inline function _triad_index(net, a::Int, b::Int, c::Int, mi::Int, mj::Int, state::Bool)
    arc(x, y) = (x == mi && y == mj) ? state : has_edge(net, x, y)
    bits = arc(a, b) + 2arc(b, a) + 4arc(a, c) + 8arc(c, a) + 16arc(b, c) + 32arc(c, b)
    return @inbounds _TRIAD_TABLE[bits + 1]
end

"""
    TriadCensus <: StructuralTerm
    TriadCensus()
    TriadCensus(levels)

The triad census (statnet's `triadcensus(levels)`): the number of triads of
each type. On a directed network the 16 Holland–Leinhardt MAN types in R's
order (`003, 012, 102, 021D, 021U, 021C, 111D, 111U, 030T, 030C, 201, 120D,
120U, 120C, 210, 300`); on an undirected one the 4 types `0`–`3` (number of
ties in the triad). `levels` are R's 0-based type indices; the default
drops type 0 (`003` / `0`), whose count is determined by the others, as R
does. Coefficient names are `"triadcensus.<type>"`.

Like `Degree(0:2)`, a multi-level `TriadCensus` is one term that expands
into one statistic per level when the model is built; a single level
(`TriadCensus(3)`) is itself a statistic. The change statistic visits the
`n − 2` triads containing the toggled dyad.

# Example
```julia
using ERGM
using ERGM: TriadCensus      # public, not exported (Siena.jl exports a GOF TriadCensus)
net = load_dataset(:florentine_marriage)
summary_stats(net, [TriadCensus()])   # (triadcensus.1 = 195.0, triadcensus.2 = 38.0, triadcensus.3 = 3.0)
compute(TriadCensus(3), net) == compute(Triangle(), net)   # true
```
"""
struct TriadCensus <: StructuralTerm
    levels::Union{Nothing,Vector{Int}}

    function TriadCensus(levels::Union{Nothing,AbstractVector{<:Integer}}=nothing)
        if levels !== nothing
            isempty(levels) && throw(ArgumentError("TriadCensus needs at least one level"))
            all(l -> 0 <= l <= 15, levels) || throw(ArgumentError(
                "TriadCensus levels are R's 0-based type indices: 0-15 on a directed " *
                "network, 0-3 on an undirected one"))
            levels = Int[l for l in levels]
        end
        new(levels)
    end
end

TriadCensus(level::Integer) = TriadCensus([level])

_triad_types(net) = is_directed(net) ? _DIRECTED_TRIAD_TYPES : _UNDIRECTED_TRIAD_TYPES

function _triad_levels(t::TriadCensus, net)
    types = _triad_types(net)
    t.levels === nothing && return collect(1:(length(types) - 1))
    bad = [l for l in t.levels if l >= length(types)]
    isempty(bad) || throw(ArgumentError(
        "TriadCensus level(s) $(join(bad, ", ")) do not exist on a" *
        (is_directed(net) ? " directed network (0-15)" : "n undirected network (0-3)")))
    return t.levels
end

name(t::TriadCensus) = t.levels === nothing ? "triadcensus" :
    length(t.levels) == 1 ? "triadcensus.$(t.levels[1])" :
    "triadcensus($(join(t.levels, ",")))"

function name(t::TriadCensus, net)
    lv = _triad_levels(t, net)
    types = _triad_types(net)
    length(lv) == 1 && return "triadcensus.$(types[lv[1] + 1])"
    return "triadcensus($(join((types[l + 1] for l in lv), ",")))"
end

@inline function _single_triad_level(t::TriadCensus)
    lv = t.levels
    (lv !== nothing && length(lv) == 1) && return @inbounds lv[1]
    throw(ArgumentError(
        "term '$(name(t))' spans several triad types and is not a single statistic; " *
        "it expands to one statistic per type when the model is built (fit_ergm / " *
        "summary_stats do this), or call compute on TriadCensus(level) for one type"))
end

function compute(t::TriadCensus, net)
    level = _single_triad_level(t)
    n = Int(nv(net))
    c = 0
    if is_directed(net)
        for a in 1:n, b in (a + 1):n, cc in (b + 1):n
            _triad_index(net, a, b, cc, 0, 0, false) == level && (c += 1)
        end
    else
        for a in 1:n, b in (a + 1):n, cc in (b + 1):n
            (has_edge(net, a, b) + has_edge(net, a, cc) + has_edge(net, b, cc)) == level &&
                (c += 1)
        end
    end
    return Float64(c)
end

function change_stat(t::TriadCensus, net, i::Int, j::Int)
    level = _single_triad_level(t)
    n = Int(nv(net))
    delta = 0
    if is_directed(net)
        for k in 1:n
            (k == i || k == j) && continue
            delta += (_triad_index(net, i, j, k, i, j, true) == level) -
                     (_triad_index(net, i, j, k, i, j, false) == level)
        end
    else
        for k in 1:n
            (k == i || k == j) && continue
            # ties of the triad besides (i, j): adding (i, j) moves its type
            # from s to s + 1
            s = has_edge(net, i, k) + has_edge(net, j, k)
            delta += (s + 1 == level) - (s == level)
        end
    end
    return Float64(delta)
end

# ============================================================================
# sender / receiver
# ============================================================================

"""
    Sender <: StructuralTerm
    Sender(; nodes=nothing)
    Sender(node::Integer)

Sender (out-degree) effects (statnet's `sender`): one statistic per vertex
`k`, the out-degree of `k` (`"sender<k>"`), whose coefficient is that
vertex's propensity to send ties — the p1/p2 sender effect. By default
every vertex but the first (R's `nodes=-1`, the base) gets one, so the
`edges` term stays identifiable; `nodes=` names the vertices explicitly.
Directed networks only. Dyad-independent.

Like a multi-level `NodeFactor`, the vertex set is resolved and expanded
into one statistic per vertex when the model is built; `Sender(k)` is a
single statistic.

# Example
```julia
using ERGM
net = network(3; directed=true); add_edges!(net, [(1, 2), (1, 3), (3, 2)])
summary_stats(net, [Sender()])        # (sender2 = 0.0, sender3 = 1.0)
compute(Sender(1), net)               # 2.0
```
"""
struct Sender <: StructuralTerm
    nodes::Union{Nothing,Vector{Int}}
    Sender(; nodes::Union{Nothing,AbstractVector{<:Integer}}=nothing) =
        new(_check_nodes("Sender", nodes))
end
Sender(node::Integer) = Sender(; nodes=[node])

"""
    Receiver <: StructuralTerm
    Receiver(; nodes=nothing)
    Receiver(node::Integer)

Receiver (in-degree) effects (statnet's `receiver`): one statistic per
vertex `k`, the in-degree of `k` (`"receiver<k>"`) — the p1/p2 receiver
effect. By default every vertex but the first; `nodes=` names them.
Directed networks only. Dyad-independent. Expands like [`Sender`](@ref).

# Example
```julia
using ERGM
net = network(3; directed=true); add_edges!(net, [(1, 2), (1, 3), (3, 2)])
summary_stats(net, [Receiver()])      # (receiver2 = 2.0, receiver3 = 1.0)
```
"""
struct Receiver <: StructuralTerm
    nodes::Union{Nothing,Vector{Int}}
    Receiver(; nodes::Union{Nothing,AbstractVector{<:Integer}}=nothing) =
        new(_check_nodes("Receiver", nodes))
end
Receiver(node::Integer) = Receiver(; nodes=[node])

function _check_nodes(kind::String, nodes)
    nodes === nothing && return nothing
    isempty(nodes) && throw(ArgumentError("$kind: nodes must name at least one vertex"))
    all(>=(1), nodes) || throw(ArgumentError("$kind: vertex ids are positive integers"))
    return Int[v for v in nodes]
end

function _nodes_of(t::Union{Sender,Receiver}, net)
    n = Int(nv(net))
    t.nodes === nothing && return collect(2:n)
    bad = [v for v in t.nodes if v > n]
    isempty(bad) || throw(ArgumentError(
        "$(nameof(typeof(t))): vertex $(join(bad, ", ")) does not exist (the network " *
        "has $n vertices)"))
    return t.nodes
end

_actor_prefix(::Sender) = "sender"
_actor_prefix(::Receiver) = "receiver"

function name(t::Union{Sender,Receiver})
    p = _actor_prefix(t)
    t.nodes === nothing && return p
    length(t.nodes) == 1 && return "$p$(t.nodes[1])"
    return "$p($(join(t.nodes, ",")))"
end

@inline function _single_node(t::Union{Sender,Receiver})
    ns = t.nodes
    (ns !== nothing && length(ns) == 1) && return @inbounds ns[1]
    throw(ArgumentError(
        "term '$(name(t))' spans several vertices and is not a single statistic; it " *
        "expands to one statistic per vertex when the model is built (fit_ergm / " *
        "summary_stats do this), or call compute on $(nameof(typeof(t)))(k) for one vertex"))
end

compute(t::Sender, net) = Float64(length(outneighbors(net, _single_node(t))))
compute(t::Receiver, net) = Float64(length(inneighbors(net, _single_node(t))))
change_stat(t::Sender, net, i::Int, j::Int) = Float64(i == _single_node(t))
change_stat(t::Receiver, net, i::Int, j::Int) = Float64(j == _single_node(t))

requires_directed(::Sender) = true
requires_directed(::Receiver) = true
is_dyad_dependent(::Sender) = false
is_dyad_dependent(::Receiver) = false

# ============================================================================
# Expansion at model construction (the `materialize` step)
# ============================================================================

function materialize(t::TriadCensus, net)
    lv = _triad_levels(t, net)
    return length(lv) == 1 && t.levels !== nothing ? t : [TriadCensus(l) for l in lv]
end

function materialize(t::Union{Sender,Receiver}, net)
    ns = _nodes_of(t, net)
    T = typeof(t)
    return length(ns) == 1 && t.nodes !== nothing ? t : [T(v) for v in ns]
end

# ============================================================================
# transitiveties / cyclicalties
# ============================================================================

"""
    TransitiveTies <: StructuralTerm
    TransitiveTies()

The number of ties `i→j` that close at least one transitive two-path
`i→k→j` (statnet's `transitiveties`, coefficient name `"transitiveties"`) —
the term of Goodreau's tergm tutorial and of many directed friendship
models. It is `GWESP(0; type=:OTP)` under R's own name: ties with one or
more outgoing-two-path shared partners. On an undirected network it is the
number of ties with at least one shared partner.

# Example
```julia
using ERGM
net = load_dataset(:sampson)
compute(TransitiveTies(), net)                          # 69.0 — R: summary(samplike ~ transitiveties)
compute(TransitiveTies(), net) == compute(GWESP(0.0), net)   # true
```
"""
struct TransitiveTies <: StructuralTerm end

"""
    CyclicalTies <: StructuralTerm
    CyclicalTies()

The number of ties `i→j` that close at least one cyclical two-path
`j→k→i` (statnet's `cyclicalties`, coefficient name `"cyclicalties"`): ties
with one or more incoming-two-path shared partners, `GWESP(0; type=:ITP)`
under R's name. On an undirected network it coincides with
[`TransitiveTies`](@ref), as in R.

# Example
```julia
using ERGM
net = load_dataset(:sampson)
compute(CyclicalTies(), net)                            # 62.0 — R: summary(samplike ~ cyclicalties)
name(CyclicalTies())                                    # "cyclicalties"
```
"""
struct CyclicalTies <: StructuralTerm end

name(::TransitiveTies) = "transitiveties"
name(::CyclicalTies) = "cyclicalties"
const _TRANSITIVE_TIES = GWESP(0.0; type=:OTP)
const _CYCLICAL_TIES = GWESP(0.0; type=:ITP)
compute(::TransitiveTies, net) = compute(_TRANSITIVE_TIES, net)
compute(::CyclicalTies, net) = compute(_CYCLICAL_TIES, net)
change_stat(::TransitiveTies, net, i::Int, j::Int) = change_stat(_TRANSITIVE_TIES, net, i, j)
change_stat(::CyclicalTies, net, i::Int, j::Int) = change_stat(_CYCLICAL_TIES, net, i, j)

# ============================================================================
# offset()
# ============================================================================

"""
    Offset <: AbstractERGMTerm
    Offset(term, coef)

A term whose coefficient is **fixed at `coef`, not estimated** — statnet's
`offset(term)` with `offset.coef = coef`. The statistic is `term`'s; the
coefficient name is `"offset(<name>)"`. A multi-statistic term (a
multi-level `NodeFactor`, `NodeMatch(attr; diff=true)` levels,
`Degree(0:2)`, …) takes one `coef` per statistic, or one `coef` for all.

Two kinds of fixed coefficient:

- **A finite offset** — e.g. `Offset(Edges(), -log(n))`, the network-size
  adjustment of egocentric and size-invariant models: the coefficient enters
  every likelihood, simulation and log-likelihood at its fixed value.
- **An infinite offset is a constraint on the sample space.** `coef = -Inf`
  forbids every tie that would raise the statistic — a *structural zero*,
  e.g. `Offset(NodeMix(:Grade, 7, 8), -Inf)`, or a same-sex
  `Offset(NodeMatch(:sex; diff=true, level="F"), -Inf)` in a heterosexual
  partnership network — and `coef = +Inf` forces every tie that would raise
  it (on a statistic a tie can lower, the roles swap). The pseudo-likelihood
  drops the dyads so fixed, the samplers never move against the constraint,
  and the log-likelihood is the conditional one given the constraint (R's
  convention). An observed network that violates the constraint has
  probability 0 under the model and is refused with an `ArgumentError`
  (R fits it regardless). The statistic may be dyad-dependent
  (`Offset(Triangle(), -Inf)`: no tie may close a triangle); an `mcmle` fit
  then reports no log-likelihood, AIC or BIC (`NaN`, said in `show` and
  `approximations`): such a constraint has no dyad-independent reference to
  bridge from.

Fits report an offset as R does: the fixed value, standard error 0, and a
note that it is not estimated; `dof` and AIC/BIC count only the estimated
coefficients (and, with an infinite offset, BIC's sample size is the dyads
the offset leaves free — R's `logLik` nobs). Offsets work in `mple`
(including `se=:bootstrap`), `mcmle` (including `missing=:mle`),
`simulate_ergm` and `gof`.

# Example
```julia
using ERGM
net = load_dataset(:faux_mesa_high)
fit = fit_ergm(net, [Offset(Edges(), -5.0), NodeMatch(:Grade)])
coef(fit)[1]                     # -5.0: fixed
name(Offset(Edges(), -5.0))      # "offset(edges)"
# Structural zeros: grade 7 has no tie to grades 8 and 9, and cannot have one
zero = fit_ergm(net, [Edges(), NodeMatch(:Grade), Offset(NodeMix(:Grade, 7, 8), -Inf),
                      Offset(NodeMix(:Grade, 7, 9), -Inf)])
round.(coef(zero)[1:2]; digits=3)   # [-5.671, 2.468] — R ergm's fit
```
"""
struct Offset{T<:AbstractERGMTerm} <: AbstractERGMTerm
    term::T
    coef::Vector{Float64}

    function Offset(term::T, coef::AbstractVector{<:Real}) where {T<:AbstractERGMTerm}
        term isa Offset && throw(ArgumentError("Offset: the term is already an offset"))
        isempty(coef) && throw(ArgumentError("Offset: give at least one coefficient"))
        any(isnan, coef) && throw(ArgumentError(
            "Offset: an offset coefficient must be a number (finite, -Inf or +Inf); " *
            "got $(coef)"))
        new{T}(term, Float64[c for c in coef])
    end
end
Offset(term::AbstractERGMTerm, coef::Real) = Offset(term, [coef])

name(t::Offset) = "offset($(name(t.term)))"
name(t::Offset, net) = "offset($(name(t.term, net)))"
compute(t::Offset, net) = compute(t.term, net)
change_stat(t::Offset, net, i::Int, j::Int) = change_stat(t.term, net, i, j)

required_vertex_attributes(t::Offset) = required_vertex_attributes(t.term)
required_edge_attributes(t::Offset) = required_edge_attributes(t.term)
requires_directed(t::Offset) = requires_directed(t.term)
requires_undirected(t::Offset) = requires_undirected(t.term)
is_dyad_dependent(t::Offset) = is_dyad_dependent(t.term)
supports_missing(t::Offset) = supports_missing(t.term)
_directed_variant_hint(t::Offset) = _directed_variant_hint(t.term)
_probe_term(t::Offset) = _probe_term(t.term)       # an offset of a user term is probed too

function materialize(t::Offset, net)
    m = materialize(t.term, net)
    if m isa AbstractVector
        k = length(m)
        length(t.coef) in (1, k) || throw(ArgumentError(
            "offset($(name(t.term, net))): $(length(t.coef)) coefficients for a term " *
            "that expands to $k statistics ($(join((name(x, net) for x in m), ", "))); " *
            "give one coefficient per statistic, or one for all"))
        return [Offset(m[i], [t.coef[length(t.coef) == 1 ? 1 : i]]) for i in 1:k]
    end
    length(t.coef) == 1 || throw(ArgumentError(
        "offset($(name(t.term, net))): $(length(t.coef)) coefficients for a single statistic"))
    return Offset(m, t.coef)
end

_specification(t::Offset) = Offset(_specification(t.term), t.coef)

# The offset coordinates of a model's (materialized) term set: a mask and the
# fixed values (NaN where the coefficient is estimated)
function _offset_info(ts::TermSet)
    p = length(ts)
    mask = falses(p)
    vals = fill(NaN, p)
    for (k, t) in enumerate(ts.terms)
        if t isa Offset
            mask[k] = true
            vals[k] = t.coef[1]
        end
    end
    return mask, vals
end
_offset_info(model::ERGMModel) = _offset_info(model.formula.terms)
_has_offsets(model::ERGMModel) = any(t -> t isa Offset, model.formula.terms.terms)

# θ'x with the convention 0·(±Inf) = 0: a fixed -Inf offset whose statistic
# (or change statistic) is zero contributes nothing
function _offset_dot(θ::AbstractVector, x::AbstractVector)
    s = 0.0
    @inbounds for k in eachindex(θ, x)
        x[k] == 0 && continue
        s += θ[k] * x[k]
    end
    return s
end

# θ'x over the FINITE coordinates of θ: an infinite offset is a constraint on
# the sample space, not a term of the (conditional) log-likelihood
function _finite_dot(θ::AbstractVector, x::AbstractVector)
    s = 0.0
    @inbounds for k in eachindex(θ, x)
        isfinite(θ[k]) || continue
        s += θ[k] * x[k]
    end
    return s
end

# log of a dyad's contribution to a dyad-independent normalizer, given its
# change statistics δ: log(1 + exp(θ'δ)) when the dyad is free; 0 when an
# infinite offset forbids the tie; θ_finite'δ when one forces it
function _dyad_logZ(θ::AbstractVector, δ::AbstractVector)
    side = 0
    @inbounds for k in eachindex(θ, δ)
        (isinf(θ[k]) && δ[k] != 0) || continue
        side = (θ[k] > 0) == (δ[k] > 0) ? 1 : -1
    end
    side == -1 && return 0.0
    η = _finite_dot(θ, δ)
    return side == 1 ? η : _log1pexp(η)
end

# The coefficient vector a sampler runs on: an infinite offset becomes a huge
# finite number of its sign, so that a move against the constraint is never
# accepted (exp(-1e300·Δ) underflows to exactly 0) while a move that leaves
# the statistic unchanged contributes exactly 0 instead of Inf·0 = NaN
_sampler_theta(θ::Vector{Float64}) =
    any(isinf, θ) ? Float64[isinf(x) ? copysign(1e300, x) : x for x in θ] : θ

# ============================================================================
# Attainable range of a statistic (R ergm's `minval` / `maxval`)
# ============================================================================

"""
    attainable_range(term, net) -> (lo::Float64, hi::Float64)

The smallest and largest values the statistic of `term` can take on networks
like `net` (the same vertex count and directedness): the `minval`/`maxval`
that R ergm's terms declare. R's `ergm.checkextreme.model` compares the
observed statistics with them. An observed value equal to `lo` fixes the
coefficient at `-Inf` and one equal to `hi` at `+Inf` (R's default
`drop=TRUE`); [`extreme_statistics`](@ref ERGM.Extension.extreme_statistics)
makes that comparison. A side with no bound is `-Inf` or `Inf`, and
`(-Inf, Inf)` declares nothing.

ERGM.jl's values are R's (read off `ergm_model(...)\$minval/maxval` for ergm
4.12). Count statistics start at 0, degree-type counts end at the number of
vertices, and `edges` and `mutual` end at the number of dyads. `nodecov`,
`absdiff`, `edgecov` and any term without a method have no bound. One
deliberate difference: R declares no lower bound for `gwesp`, `gwdsp` and
`gwnsp`, but with a decay ≥ 0 (the only decays ERGM.jl accepts) they are
weighted counts and cannot go below 0, so they get `(0, Inf)`. On a network
with no shared partner R fits a finite, unidentified coefficient (−0.31 on a
30-node perfect matching, after warning that the statistic is "not
varying"); here it is fixed at `-Inf`, as a boundary found in the design
already is.

The range matters where the pseudo-likelihood design cannot show the bound:
a statistic whose change statistics are all zero on the observed network
(a `NodeMix` cell of a singleton level, a `Triangle` on a network with no
two-path) gives the design a flat direction, not a one-signed gradient.

Part of [`ERGM.Extension`](@ref), and an extension point: a package adds a
method for its own term type, or for its own network type, instead of
keeping a separate range table. ERGMMulti declares its layer terms' ranges
on a `MultilayerNetwork`, and ERGMRank its rank terms' ranges on a
`RankNetwork`. ERGM.jl's own methods take an `AbstractNetwork`, so they never
claim a bound on another package's network type.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)             # 16 vertices, undirected
ERGM.Extension.attainable_range(Edges(), net)        # (0.0, 120.0)
ERGM.Extension.attainable_range(Degree(2), net)      # (0.0, 16.0)
ERGM.Extension.attainable_range(NodeCov(:wealth), net)   # (-Inf, Inf)
# A third-party term declares its own range
struct Isolates <: AbstractERGMTerm end
ERGM.Extension.attainable_range(::Isolates, net::AbstractNetwork) = (0.0, Float64(nv(net)))
ERGM.Extension.attainable_range(Isolates(), net)     # (0.0, 16.0)
```
"""
attainable_range(::AbstractERGMTerm, net) = (-Inf, Inf)

_n_pairs(net) = (n = Int(nv(net)); is_directed(net) ? n * (n - 1) : n * (n - 1) ÷ 2)

attainable_range(::Edges, net::AbstractNetwork) = (0.0, Float64(_n_pairs(net)))
attainable_range(::Mutual, net::AbstractNetwork) = (0.0, Float64(Int(nv(net)) * (Int(nv(net)) - 1) ÷ 2))
attainable_range(::Density, net::AbstractNetwork) = (0.0, 1.0)
attainable_range(::MeanDeg, net::AbstractNetwork) = (0.0, Float64(Int(nv(net)) - 1))
for T in (:Triangle, :Kstar, :OStar, :IStar, :TwoPath, :TransitiveTies, :CyclicalTies,
          :GWESP, :GWDSP, :GWNSP, :ESP,
          :TriadCensus, :NodeMatch, :NodeMismatch, :NodeMix, :NodeFactor,
          :MaterializedNodeMatch, :MaterializedNodeMismatch, :MaterializedNodeMix,
          :MaterializedNodeFactor)
    @eval attainable_range(::$T, net::AbstractNetwork) = (0.0, Inf)
end
for T in (:Degree, :IDegree, :ODegree, :GWDegree, :GWIDegree, :GWODegree, :Concurrent,
          :DegRange, :IDegRange, :ODegRange)
    @eval attainable_range(::$T, net::AbstractNetwork) = (0.0, Float64(nv(net)))
end
attainable_range(::Union{Sender,Receiver}, net::AbstractNetwork) = (0.0, Float64(Int(nv(net)) - 1))
attainable_range(t::Offset, net) = attainable_range(t.term, net)

"""
    extreme_statistics(terms, net) -> Vector{Tuple{Int,Symbol}}

The statistics whose observed value on `net` sits at an end of their
[`attainable_range`](@ref ERGM.Extension.attainable_range), as
`(position, :min)` or `(position, :max)`: R ergm's
`ergm.checkextreme.model`. `terms` is a `TermSet` or a vector of terms,
normally as materialized for `net`; the observed values are `compute(term,
net)` at face value. `Offset` terms are fixed already and never listed.

This is the `extreme` seed that
[`boundary_columns`](@ref ERGM.Extension.boundary_columns) and
[`mple_fit_design`](@ref ERGM.Extension.mple_fit_design) take. It dispatches
through `attainable_range` and `compute` alone, so it works for any term and
network type that has methods for both.

Part of [`ERGM.Extension`](@ref).

# Example
```julia
using ERGM
# A perfect matching on 10 vertices: five ties and no two-path, so the
# triangle count is at its minimum although no single dyad shows it
net = network(10; directed=false)
for i in 1:2:9
    add_edge!(net, i, i + 1)
end
ERGM.Extension.extreme_statistics([Edges(), Triangle()], net)   # [(2, :min)]
```
"""
extreme_statistics(ts::TermSet, net) = extreme_statistics(ts.terms, net)
function extreme_statistics(terms::Union{Tuple,AbstractVector}, net)
    out = Tuple{Int,Symbol}[]
    for (k, t) in enumerate(terms)
        t isa Offset && continue
        lo, hi = attainable_range(t, net)
        (isfinite(lo) || isfinite(hi)) || continue
        g = Float64(compute(t, net))
        if g == lo
            push!(out, (k, :min))
        elseif g == hi
            push!(out, (k, :max))
        end
    end
    return out
end
