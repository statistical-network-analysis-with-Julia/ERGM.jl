using ERGM
using Graphs: Graphs
using LinearAlgebra
using NetworkCore
using StatsAPI: StatsAPI
using Random
using Statistics
using StatsBase
using Test
using Aqua

# Run a docstring example statement by statement in `mod` and compare every
# statement whose line ends in a `# value` comment with that value (the
# approach of Revel.jl's docs gate). A comment is a claim when its text —
# after an optional leading `≈`, cut at ` — `, ` – `, `; `, two spaces or
# ` (`, and at the last ` = ` — parses as a literal: numbers, strings,
# symbols, `true`/`false`, `nothing`, `NaN`/`Inf`, vectors and tuples of
# those, or arithmetic/`log`/`sqrt`/`exp` of numbers; a comment that starts
# with a number followed by prose ("# 23.0 endpoint appearances") claims that
# number. Anything else is prose. A number written with d decimals matches to
# half a unit of its last digit; `≈` allows 1 %. Returns (claims checked,
# mismatch descriptions); throws when a statement throws.
const _LITERAL_CALLS = (:+, :-, :*, :/, :^, :log, :sqrt, :exp, :log1p, :log2)
_is_literal(x) = x isa Union{Number, String, QuoteNode, Bool, Nothing} ||
    x in (:NaN, :Inf, :nothing, :true, :false) ||
    (x isa Expr && (x.head in (:vect, :tuple) && all(_is_literal, x.args) ||
                    x.head === :call && x.args[1] in _LITERAL_CALLS &&
                        all(a -> _is_literal(a) && !(a isa String), x.args[2:end])))
function _claim(comment::AbstractString)
    text = strip(comment)
    approx = startswith(text, "≈")
    approx && (text = strip(text[nextind(text, 1):end]))
    for sep in (" — ", " – ", "; ", "  ", " (")
        text = first(split(text, sep))
    end
    # a NamedTuple written out, "(burnin = 2400, interval = 100)", is prose
    startswith(text, "(") && occursin("=", text) && return nothing
    occursin(" = ", text) && (text = strip(last(split(text, " = "))))
    isempty(text) && return nothing
    ex = try Meta.parse(text; raise=true) catch; nothing end
    lead = false
    if ex === nothing || (ex isa Expr && ex.head === :incomplete) || !_is_literal(ex)
        # "23.0 endpoint appearances": a leading number, then prose
        m = match(r"^(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s+[A-Za-z]", text)
        m === nothing && return nothing
        text = m.captures[1]
        ex = Meta.parse(text)
        lead = true
    end
    digits = maximum((length(m.captures[1]) for m in eachmatch(r"\d\.(\d+)", text)); init=0)
    return (value=Core.eval(Main, ex), approx=approx, digits=digits, text=text, lead=lead)
end
_matches(v::Number, c::Number, approx, digits) =
    (isnan(c) && isnan(v)) || v == c ||
    (approx ? isapprox(v, c; rtol=0.01, atol=1e-12) :
     digits > 0 && abs(v - c) <= 0.5 * 10.0^(-digits) * (1 + 1e-9))
_matches(v::AbstractString, c::AbstractString, _, _) = v == c
_matches(v::Symbol, c::QuoteNode, _, _) = v == c.value
_matches(v::Symbol, c::Symbol, _, _) = v == c
_matches(v::Bool, c::Bool, _, _) = v == c
_matches(v::Nothing, c::Nothing, _, _) = true
_matches(v::Union{AbstractVector,Tuple}, c::Union{AbstractVector,Tuple}, a, d) =
    length(v) == length(c) && all(_matches(x, y, a, d) for (x, y) in zip(v, c))
_matches(v, c, a, d) = false
function check_block(mod::Module, block::AbstractString; where="")
    checked = 0; bad = String[]
    pos = 1
    while pos <= lastindex(block)
        ex, next = Meta.parse(block, pos; raise=true)
        src = block[pos:prevind(block, next)]
        pos = next
        ex === nothing && continue
        val = Core.eval(mod, ex)
        line = rstrip(last(split(rstrip(src), "\n")))
        # the comment of the statement's last line (not a `#` inside a string)
        m = match(r"^(?:[^\"#]|\"(?:[^\"\\]|\\.)*\")*#(.*)$", line)
        m === nothing && continue
        startswith(strip(line), "#") && continue
        c = _claim(m.captures[1])
        c === nothing && continue
        # a leading number before prose claims a NUMBER ("# 2 sd off target"
        # after a NamedTuple is prose)
        c.lead && !(val isa Number) && continue
        checked += 1
        _matches(val, c.value, c.approx, c.digits) ||
            push!(bad, "$where: `$(strip(split(line, "#")[1]))` is $(repr(val)), the comment says $(c.text)")
    end
    return checked, bad
end

"""
Brute-force add-direction change statistic: g(y⁺ij) − g(y⁻ij) computed by
actually toggling the dyad. Restores the network to its original state.
"""
function brute_change_stat(term, net, i, j)
    had = has_edge(net, i, j)
    had && rem_edge!(net, i, j)
    s0 = compute(term, net)
    add_edge!(net, i, j)
    s1 = compute(term, net)
    had || rem_edge!(net, i, j)
    return s1 - s0
end

"Check change_stat against brute force for every dyad of the network."
function check_change_stats(term, net; atol=1e-10)
    n = nv(net)
    for i in 1:n
        j_range = is_directed(net) ? (1:n) : ((i+1):n)
        for j in j_range
            i == j && continue
            expected = brute_change_stat(term, net, i, j)
            actual = change_stat(term, net, i, j)
            if !isapprox(actual, expected; atol=atol)
                return (i, j, actual, expected)
            end
        end
    end
    return nothing
end

# Florentine families marriage network (Padgett), the standard R ergm
# example dataset. 16 families, 20 marriage ties, Pucci (12) isolated.
function florentine_marriage()
    net = network(16; directed=false)
    ties = [(1, 9), (2, 6), (2, 7), (2, 9), (3, 5), (3, 9), (4, 7), (4, 11),
            (4, 15), (5, 11), (5, 15), (7, 8), (7, 16), (9, 13), (9, 14),
            (9, 16), (10, 14), (11, 15), (13, 15), (13, 16)]
    for (i, j) in ties
        add_edge!(net, i, j)
    end
    wealth = Dict(1 => 10, 2 => 36, 3 => 55, 4 => 44, 5 => 20, 6 => 32,
                  7 => 8, 8 => 42, 9 => 103, 10 => 48, 11 => 49, 12 => 3,
                  13 => 27, 14 => 10, 15 => 146, 16 => 48)
    set_vertex_attribute!(net, :wealth, wealth)
    return net
end

# Sampson's monastery "like" network (statnet's `samplike`, bundled with R
# ergm: 18 monks, 88 directed ties), rebuilt from the edge list the
# provenanced term fixture exports, so the directed golden rows are
# reproducible from the TOML alone.
function samplike()
    g = load_golden(joinpath(@__DIR__, "fixtures", "ergm_terms.toml"))
    net = network(Int(g.values["samplike_n"]); directed=true)
    for (t, h) in zip(g.values["samplike_tails"], g.values["samplike_heads"])
        add_edge!(net, Int(t), Int(h))
    end
    return net
end

# Small test fixtures with a mix of triangles, stars, and isolates
function fixture_undirected()
    net = network(7; directed=false)
    for (i, j) in [(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (2, 5), (5, 6), (1, 5)]
        add_edge!(net, i, j)
    end
    return net
end

function fixture_directed()
    net = network(7; directed=true)
    for (i, j) in [(1, 2), (2, 1), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3),
                   (1, 5), (5, 6), (6, 2)]
        add_edge!(net, i, j)
    end
    return net
end

function set_test_attrs!(net)
    set_vertex_attribute!(net, :group,
        Dict(1 => "A", 2 => "A", 3 => "B", 4 => "B", 5 => "A", 6 => "B", 7 => "A"))
    set_vertex_attribute!(net, :age,
        Dict(1 => 20.0, 2 => 35.0, 3 => 28.0, 4 => 51.0, 5 => 42.0, 6 => 33.0, 7 => 60.0))
    return net
end

# A "third-party" term: it lives outside ERGM.jl's own term hierarchy and
# declares its requirements purely through the public term-trait protocol
# (src/terms/traits.jl). Everything ERGM does with it at model construction —
# attribute validation, direction validation — must follow from the
# declarations alone, exactly as for a built-in term.
struct ForeignTerm <: AbstractERGMTerm
    vattr::Symbol
    eattr::Symbol
end
ERGM.name(t::ForeignTerm) = "foreign.$(t.vattr)"
ERGM.compute(t::ForeignTerm, net) = Float64(ne(net))
ERGM.change_stat(t::ForeignTerm, net, i::Int, j::Int) = 1.0
ERGM.required_vertex_attributes(t::ForeignTerm) = (t.vattr,)
ERGM.required_edge_attributes(t::ForeignTerm) = (t.eattr,)
ERGM.requires_directed(::ForeignTerm) = true
ERGM.is_dyad_dependent(::ForeignTerm) = false

# A constraint term: the reserved abstract type exists, but constraints are
# refused at ERGMFormula construction (see the "Refused, not mis-fit" testset)
struct FakeConstraint <: ERGM.ConstraintTerm end

# A user term written the way the ERGMUserterms README writes one: NO type
# annotations on the dyad arguments of `change_stat` (the number of ties
# whose endpoint ids sum to an even number; dyad-independent)
struct ExampleTerm <: AbstractERGMTerm end
ERGM.name(::ExampleTerm) = "example"
ERGM.compute(::ExampleTerm, net) =
    Float64(count(e -> iseven(src(e) + dst(e)), edges(net)))
ERGM.change_stat(::ExampleTerm, net, i, j) = iseven(i + j) ? 1.0 : 0.0
ERGM.is_dyad_dependent(::ExampleTerm) = false

# A user term whose value on the EMPTY network is not zero: the number of
# non-ties of an undirected network (dyad-independent)
struct GEmptyTerm <: AbstractERGMTerm end
ERGM.name(::GEmptyTerm) = "nonedges"
ERGM.compute(::GEmptyTerm, net) = nv(net) * (nv(net) - 1) / 2 - ne(net)
ERGM.change_stat(::GEmptyTerm, net, i, j) = -1.0
ERGM.is_dyad_dependent(::GEmptyTerm) = false

# A user term that DECLARES dyad independence but whose change statistic
# reads the reverse tie (a mutual count): the declaration is false
struct MisdeclaredMutual <: AbstractERGMTerm end
ERGM.name(::MisdeclaredMutual) = "badmutual"
ERGM.compute(::MisdeclaredMutual, net) =
    count(e -> has_edge(net, dst(e), src(e)), edges(net)) / 2
ERGM.change_stat(::MisdeclaredMutual, net, i, j) = has_edge(net, j, i) ? 1.0 : 0.0
ERGM.is_dyad_dependent(::MisdeclaredMutual) = false

# A user term that reads an edge attribute LIVE (the sum of the ties'
# :weight, default 1): the samplers' rem_edge! deletes the weight, so under
# MCMC it would decay to an edge count. And the same term snapshotting the
# weights when the model is built (ERGMUserterms' WeightedEdges pattern).
struct LiveWeightTerm <: AbstractERGMTerm end
_lw(net, i, j) = (w = get_edge_attribute(net, :weight, i, j); w === nothing ? 1.0 : Float64(w))
ERGM.name(::LiveWeightTerm) = "liveweight"
ERGM.compute(::LiveWeightTerm, net) = sum((_lw(net, src(e), dst(e)) for e in edges(net)); init=0.0)
ERGM.change_stat(::LiveWeightTerm, net, i, j) = _lw(net, i, j)
struct SnapWeightTerm <: AbstractERGMTerm
    W::Matrix{Float64}
end
ERGM.name(::SnapWeightTerm) = "liveweight"
ERGM.compute(t::SnapWeightTerm, net) = sum((t.W[src(e), dst(e)] for e in edges(net)); init=0.0)
ERGM.change_stat(t::SnapWeightTerm, net, i, j) = t.W[i, j]
ERGM.is_dyad_dependent(::SnapWeightTerm) = false
struct SnappedWeightTerm <: AbstractERGMTerm end        # reads live, but a model snapshots it
ERGM.name(::SnappedWeightTerm) = "liveweight"
ERGM.compute(::SnappedWeightTerm, net) = compute(LiveWeightTerm(), net)
ERGM.change_stat(::SnappedWeightTerm, net, i, j) = _lw(net, i, j)
ERGM.is_dyad_dependent(::SnappedWeightTerm) = false
function ERGM.Extension.materialize(::SnappedWeightTerm, net)
    n = nv(net)
    return SnapWeightTerm([i == j ? 0.0 : _lw(net, i, j) for i in 1:n, j in 1:n])
end

# A term with no `change_stat` (the fallback's message is tested), and one
# whose change statistic throws inside a sampler chain
struct NoChangeStat <: AbstractERGMTerm end
struct ThrowingTerm <: AbstractERGMTerm end
ERGM.name(::ThrowingTerm) = "throwing"
ERGM.compute(::ThrowingTerm, net) = 0.0
ERGM.change_stat(::ThrowingTerm, net, i, j) = throw(ArgumentError("boom from a chain"))

# A third-party term that declares its attainable range through the
# extension API, and one that does not (extension-API testset)
struct ExtIsolates <: AbstractERGMTerm end
struct ExtIsolatesUnranged <: AbstractERGMTerm end
ERGM.compute(::Union{ExtIsolates,ExtIsolatesUnranged}, net) =
    Float64(count(v -> degree(net, v) == 0, vertices(net)))
ERGM.Extension.attainable_range(::ExtIsolates, net::AbstractNetwork) = (0.0, Float64(nv(net)))
struct ExtForeignNetwork end        # a network type that is not an AbstractNetwork

@testset "ERGM.jl" begin
    @testset "Change statistic convention" begin
        net = network(5)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        add_edge!(net, 3, 4)

        edges_term = Edges()
        @test compute(edges_term, net) == 3.0

        # The add-direction change statistic is state-independent: it is +1
        # for the edges term whether or not the dyad currently has an edge
        @test change_stat(edges_term, net, 1, 3) == 1.0
        @test change_stat(edges_term, net, 1, 2) == 1.0
    end

    @testset "Structural terms match brute force (undirected)" begin
        net = fixture_undirected()
        for term in [Edges(), Triangle(), Kstar(2), Kstar(3), TwoPath(),
                     GWESP(0.0), GWESP(0.5), GWESP(1.2),
                     GWDegree(0.0), GWDegree(0.5), GWDegree(1.2),
                     Degree(0), Degree(1), Degree(2), Degree(3),
                     GWDSP(0.0), GWDSP(0.5), GWDSP(1.2)]
            @test check_change_stats(term, net) === nothing
        end
    end

    @testset "Structural terms match brute force (directed)" begin
        net = fixture_directed()
        for term in [Edges(), Mutual(), Triangle(), TwoPath(),
                     OStar(2), OStar(3), IStar(2), IStar(3),
                     GWESP(0.0), GWESP(0.5), GWESP(0.5; type=:ITP), GWESP(0.5; type=:OSP),
                     GWESP(0.5; type=:ISP), GWESP(0.5; type=:union),
                     GWESP(0.0; type=:OSP), GWESP(1.2; type=:ITP),
                     IDegree(0), IDegree(1), IDegree(2), ODegree(1), ODegree(2),
                     GWIDegree(0.0), GWIDegree(0.5), GWIDegree(1.2),
                     GWODegree(0.0), GWODegree(0.5),
                     GWDSP(0.0), GWDSP(0.5), GWDSP(0.5; type=:ITP), GWDSP(0.5; type=:OSP),
                     GWDSP(0.5; type=:ISP), GWDSP(0.5; type=:union),
                     GWDSP(0.0; type=:ISP), GWDSP(1.2; type=:OSP)]
            @test check_change_stats(term, net) === nothing
        end
    end

    @testset "Directed GWESP shared-partner types (hand-computed)" begin
        # weight(s) = e^α (1 − (1 − e^{-α})^s), so weight(1) = 1 exactly and
        # weight(2) = 1 + (1 − e^{-α})
        w2(α) = 1 + (1 - exp(-α))

        # Fixture A: arcs 1→3, 3→2, 1→4, 4→2, 1→2. The edge 1→2 has two
        # outgoing two-paths (via 3 and 4); by hand:
        #   OTP counts {2,0,0,0,0}; ITP all 0; OSP counts {1,1,0,0,0}
        #   (edges 1→3, 1→4 share out-partner 2); ISP counts {1,1,0,0,0}
        #   (edges 3→2, 4→2 share in-partner 1); union counts {1,1,1,1,2}
        netA = network(5; directed=true)
        for (i, j) in [(1, 3), (3, 2), (1, 4), (4, 2), (1, 2)]
            add_edge!(netA, i, j)
        end
        for α in (0.5, 1.2)
            @test compute(GWESP(α), netA) ≈ w2(α)                      # OTP default
            @test compute(GWESP(α; type=:OTP), netA) ≈ w2(α)
            @test compute(GWESP(α; type=:ITP), netA) ≈ 0.0
            @test compute(GWESP(α; type=:OSP), netA) ≈ 2.0
            @test compute(GWESP(α; type=:ISP), netA) ≈ 2.0
            @test compute(GWESP(α; type=:union), netA) ≈ 4.0 + w2(α)
        end

        # Fixture B: arcs 1→2, 1→3, 2→3, 1→4, 2→4. The edge 1→2 has two
        # shared out-partners (3 and 4); by hand:
        #   OTP counts {1,1} (edges 1→3, 1→4 via 2); ITP all 0;
        #   OSP counts {2}; ISP counts {1,1} (edges 2→3, 2→4 via 1);
        #   union counts {2,1,1,1,1}
        netB = network(5; directed=true)
        for (i, j) in [(1, 2), (1, 3), (2, 3), (1, 4), (2, 4)]
            add_edge!(netB, i, j)
        end
        for α in (0.5, 1.2)
            @test compute(GWESP(α; type=:OTP), netB) ≈ 2.0
            @test compute(GWESP(α; type=:ITP), netB) ≈ 0.0
            @test compute(GWESP(α; type=:OSP), netB) ≈ w2(α)
            @test compute(GWESP(α; type=:ISP), netB) ≈ 2.0
            @test compute(GWESP(α; type=:union), netB) ≈ 4.0 + w2(α)
        end

        # Cyclic triad 1→2→3→1: every edge has exactly one incoming
        # two-path and no other kind of shared partner
        netC = network(3; directed=true)
        for (i, j) in [(1, 2), (2, 3), (3, 1)]
            add_edge!(netC, i, j)
        end
        @test compute(GWESP(0.5; type=:OTP), netC) ≈ 0.0
        @test compute(GWESP(0.5; type=:ITP), netC) ≈ 3.0
        @test compute(GWESP(0.5; type=:OSP), netC) ≈ 0.0
        @test compute(GWESP(0.5; type=:ISP), netC) ≈ 0.0
        @test compute(GWESP(0.5; type=:union), netC) ≈ 3.0

        # Names. Without a network the default type gets the undirected
        # label; the union variant is named so it cannot be confused with
        # statnet's
        @test ERGM.name(GWESP(0.5)) == "gwesp.fixed.0.5"
        @test ERGM.name(GWESP(0.5; type=:OTP)) == "gwesp.fixed.0.5"
        @test ERGM.name(GWESP(0.5; type=:ITP)) == "gwesp.ITP.fixed.0.5"
        @test ERGM.name(GWESP(0.5; type=:union)) == "gwesp.union.fixed.0.5"
        @test_throws ArgumentError GWESP(0.5; type=:XYZ)
        # With the network known, the label is R's: on a DIRECTED network
        # ergm 4.12.0 names the default `gwesp.OTP.fixed.0.5` (pinned by the
        # provenanced ergm_terms fixture), on an undirected one the type is
        # ignored in the label as it is in the statistic
        @test name(GWESP(0.5), netA) == "gwesp.OTP.fixed.0.5"
        @test name(GWESP(0.5; type=:ISP), netA) == "gwesp.ISP.fixed.0.5"
        @test name(GWESP(0.5; type=:union), netA) == "gwesp.union.fixed.0.5"
        m = ERGMModel(ERGMFormula([Edges(), GWESP(0.5), GWDSP(0.5)]), netA)
        @test m.formula.terms.names == ["edges", "gwesp.OTP.fixed.0.5", "gwdsp.OTP.fixed.0.5"]
        @test keys(summary_stats(netA, [GWESP(0.5)])) == (Symbol("gwesp.OTP.fixed.0.5"),)
        @test sprint(show, m.formula) == "ERGMFormula: edges + gwesp.OTP.fixed.0.5 + gwdsp.OTP.fixed.0.5"
        @test sprint(show, m) ==
              "ERGMModel{Int64,true}: 5 vertices, 5 edges (directed); terms: edges + " *
              "gwesp.OTP.fixed.0.5 + gwdsp.OTP.fixed.0.5"

        # Undirected networks ignore the type: all variants agree, and so do
        # their labels
        unet = fixture_undirected()
        base = compute(GWESP(0.5), unet)
        for t in (:OTP, :ITP, :OSP, :ISP, :union)
            @test compute(GWESP(0.5; type=t), unet) ≈ base
            @test name(GWESP(0.5; type=t), unet) ==
                  (t === :union ? "gwesp.union.fixed.0.5" : "gwesp.fixed.0.5")
        end
        @test name(Edges(), unet) == "edges"
        um = ERGMModel(ERGMFormula([Edges(), GWESP(0.5; type=:ISP)]), unet)
        @test um.formula.terms.names == ["edges", "gwesp.fixed.0.5"]
        @test sprint(show, um) ==
              "ERGMModel{Int64,false}: $(nv(unet)) vertices, $(ne(unet)) edges " *
              "(undirected); terms: edges + gwesp.fixed.0.5"
    end

    @testset "Degree count terms (hand-computed)" begin
        # Path 1-2-3-4 plus isolate 5: degrees 1, 2, 2, 1, 0
        net = network(5; directed=false)
        for (i, j) in [(1, 2), (2, 3), (3, 4)]
            add_edge!(net, i, j)
        end
        @test compute(Degree(0), net) == 1.0
        @test compute(Degree(1), net) == 2.0
        @test compute(Degree(2), net) == 2.0
        @test compute(Degree(3), net) == 0.0
        @test ERGM.name(Degree(2)) == "degree2"

        # fixture_directed in-degrees: [2,2,2,1,2,1,0]; out: [2,2,2,1,2,1,0]
        dnet = fixture_directed()
        @test compute(IDegree(0), dnet) == 1.0
        @test compute(IDegree(1), dnet) == 2.0
        @test compute(IDegree(2), dnet) == 4.0
        @test compute(ODegree(0), dnet) == 1.0
        @test compute(ODegree(1), dnet) == 2.0
        @test compute(ODegree(2), dnet) == 4.0
        @test ERGM.name(IDegree(1)) == "idegree1"
        @test ERGM.name(ODegree(3)) == "odegree3"

        # A vector of degrees is ONE term that expands into one statistic per
        # degree at model construction, as statnet's degree(0:2) does
        @test Degree(0:2) == Degree([0, 1, 2])
        @test Degree(0:2) isa Degree
        @test IDegree([1, 3]).degrees == [1, 3]
        @test ODegree(1:2) == ODegree([1, 2])
        model = ERGMModel(ERGMFormula([Edges(); Degree(0:2)]), net)
        @test model.formula.terms.names == ["edges", "degree0", "degree1", "degree2"]
        @test compute_all(model.formula.terms, net) == [3.0, 1.0, 2.0, 2.0]
        model = ERGMModel(ERGMFormula([Edges(), Degree(0:2)]), net)
        @test model.formula.terms.names == ["edges", "degree0", "degree1", "degree2"]

        @test_throws ArgumentError Degree(-1)
        @test_throws ArgumentError Degree(Int[])

        # Direction requirements match R ergm: degree is undirected-only,
        # idegree/odegree (and the GW variants) directed-only
        @test_throws ArgumentError ERGMModel(ERGMFormula([Edges(), Degree(1)]), dnet)
        for t in (IDegree(1), ODegree(1), GWIDegree(0.5), GWODegree(0.5))
            @test_throws ArgumentError ERGMModel(ERGMFormula([Edges(), t]), net)
        end
    end

    @testset "GWIDegree / GWODegree (hand-computed)" begin
        # weight(d) = eᵅ(1 − (1 − e⁻ᵅ)ᵈ): weight(0) = 0, weight(1) = 1,
        # weight(2) = 1 + (1 − e⁻ᵅ)
        w2(α) = 1 + (1 - exp(-α))

        # Arcs 1→2, 1→3, 2→3: in-degrees [0, 1, 2], out-degrees [2, 1, 0]
        net = network(3; directed=true)
        for (i, j) in [(1, 2), (1, 3), (2, 3)]
            add_edge!(net, i, j)
        end
        for α in (0.5, 1.2)
            @test compute(GWIDegree(α), net) ≈ 1.0 + w2(α)
            @test compute(GWODegree(α), net) ≈ 1.0 + w2(α)
        end
        @test ERGM.name(GWIDegree(0.5)) == "gwideg.fixed.0.5"
        @test ERGM.name(GWODegree(0.5)) == "gwodeg.fixed.0.5"
        @test_throws ArgumentError GWIDegree(-1.0)
        @test_throws ArgumentError GWODegree(-0.1)
        # decay 0 is legal (statnet's gwidegree(0, fixed=TRUE)): the number
        # of vertices with in-/out-degree ≥ 1
        @test compute(GWIDegree(0.0), net) == 2.0
        @test compute(GWODegree(0.0), net) == 2.0
    end

    @testset "GWDSP shared-partner types (hand-computed)" begin
        w2(α) = 1 + (1 - exp(-α))

        # Undirected fixtures. Path 1-2-3: only the (untied) dyad {1,3} has
        # a shared partner. Triangle: every dyad has one. 4-cycle: the two
        # diagonals have two each, the tied dyads none.
        path3 = network(3; directed=false)
        add_edge!(path3, 1, 2); add_edge!(path3, 2, 3)
        tri = network(3; directed=false)
        for (i, j) in [(1, 2), (2, 3), (1, 3)]
            add_edge!(tri, i, j)
        end
        cyc4 = network(4; directed=false)
        for (i, j) in [(1, 2), (2, 3), (3, 4), (1, 4)]
            add_edge!(cyc4, i, j)
        end
        for α in (0.5, 1.2)
            @test compute(GWDSP(α), path3) ≈ 1.0
            @test compute(GWDSP(α), tri) ≈ 3.0
            @test compute(GWDSP(α), cyc4) ≈ 2.0 * w2(α)
        end

        # Directed fixture (netA of the GWESP tests): arcs 1→3, 3→2, 1→4,
        # 4→2, 1→2. Every statnet type sums over ORDERED dyads (R's `ddsp`;
        # OSP/ISP used to be summed over unordered dyads, at
        # exactly half of R's dgwdsp — 316.085/229.136 on samplike, pinned by
        # the provenanced fixture). By hand:
        #   OTP: only (1,2) has shared partners, two of them
        #   ITP: only (2,1) — the same two-paths read backwards, so the
        #     statistic equals OTP's (statnet: OTP and ITP are equivalent
        #     for DSP)
        #   OSP: the pairs {1,3}, {1,4}, {3,4} share one out-partner each,
        #     counted for both orderings: 6 ordered dyads with one partner
        #   ISP: {2,3}, {2,4}, {3,4} share one in-partner: 6 ordered dyads
        #   union (unordered): {1,2} and {3,4} have two, the four tied dyads
        #     one each
        netA = network(5; directed=true)
        for (i, j) in [(1, 3), (3, 2), (1, 4), (4, 2), (1, 2)]
            add_edge!(netA, i, j)
        end
        for α in (0.5, 1.2)
            @test compute(GWDSP(α), netA) ≈ w2(α)                    # OTP default
            @test compute(GWDSP(α; type=:OTP), netA) ≈ w2(α)
            @test compute(GWDSP(α; type=:ITP), netA) ≈ w2(α)
            @test compute(GWDSP(α; type=:OSP), netA) ≈ 6.0
            @test compute(GWDSP(α; type=:ISP), netA) ≈ 6.0
            @test compute(GWDSP(α; type=:union), netA) ≈ 4.0 + 2.0 * w2(α)
        end
        # A 3-node check of the ordered-dyad convention: arcs 1→3, 2→3 give
        # the pair {1,2} one shared out-partner; R's dgwdsp(type="OSP") is 2
        net3 = network(3; directed=true)
        add_edge!(net3, 1, 3); add_edge!(net3, 2, 3)
        @test compute(GWDSP(0.5; type=:OSP), net3) == 2.0
        @test compute(GWDSP(0.5; type=:ISP), net3) == 0.0
        # ... and the change statistic is the ordered-dyad one too
        @test change_stat(GWDSP(0.5; type=:OSP), net3, 1, 3) ≈ 2.0
        @test change_stat(GWDSP(0.5; type=:ISP), net3, 3, 1) ≈ 0.0
        add_edge!(net3, 3, 1); add_edge!(net3, 3, 2)
        @test change_stat(GWDSP(0.5; type=:ISP), net3, 3, 1) ≈ 2.0

        # Names follow the GWESP convention: the undirected label without a
        # network, R's directed label (`gwdsp.OTP.fixed.0.5`) with one
        @test ERGM.name(GWDSP(0.5)) == "gwdsp.fixed.0.5"
        @test ERGM.name(GWDSP(0.5; type=:OTP)) == "gwdsp.fixed.0.5"
        @test ERGM.name(GWDSP(0.5; type=:OSP)) == "gwdsp.OSP.fixed.0.5"
        @test ERGM.name(GWDSP(0.5; type=:union)) == "gwdsp.union.fixed.0.5"
        @test name(GWDSP(0.5), netA) == "gwdsp.OTP.fixed.0.5"
        @test name(GWDSP(0.5; type=:OSP), netA) == "gwdsp.OSP.fixed.0.5"
        @test name(GWDSP(0.5; type=:OSP), path3) == "gwdsp.fixed.0.5"
        @test_throws ArgumentError GWDSP(0.5; type=:XYZ)
        @test_throws ArgumentError GWDSP(-0.5)

        # Undirected networks ignore the type: all variants agree
        unet = fixture_undirected()
        base = compute(GWDSP(0.5), unet)
        for t in (:OTP, :ITP, :OSP, :ISP, :union)
            @test compute(GWDSP(0.5; type=t), unet) ≈ base
        end
    end

    @testset "NodeFactor drops the first level by default (statnet parity)" begin
        net = set_test_attrs!(fixture_undirected())
        # Endpoint appearances: level A (vertices 1,2,5,7) = 3+3+4+0 = 10,
        # level B (vertices 3,4,6) = 3+2+1 = 6

        # Default: levels are sorted {A, B}, the first is the reference
        model = ERGMModel(ERGMFormula([Edges(), NodeFactor(:group)]), net)
        @test model.formula.terms.names == ["edges", "nodefactor.group.B"]
        @test compute_all(model.formula.terms, net) == [8.0, 6.0]

        # base=0 keeps every level; base=2 drops B instead
        m0 = ERGMModel(ERGMFormula([NodeFactor(:group; base=0)]), net)
        @test m0.formula.terms.names == ["nodefactor.group.A", "nodefactor.group.B"]
        @test compute_all(m0.formula.terms, net) == [10.0, 6.0]
        m2 = ERGMModel(ERGMFormula([NodeFactor(:group; base=2)]), net)
        @test m2.formula.terms.names == ["nodefactor.group.A"]

        # Explicit levels select and order the statistics
        ml = ERGMModel(ERGMFormula([NodeFactor(:group; levels=["B", "A"])]), net)
        @test ml.formula.terms.names == ["nodefactor.group.B", "nodefactor.group.A"]

        # An unexpanded multi-level term evaluates to the sum of its levels
        @test compute(NodeFactor(:group), net) == 6.0
        @test compute(NodeFactor(:group; base=0), net) == 16.0

        # Fitting via the front door matches the explicit single-level term
        f1 = fit_ergm(net, [Edges(), NodeFactor(:group)])
        f2 = fit_ergm(net, [Edges(), NodeFactor(:group; level="B")])
        @test coef(f1) ≈ coef(f2) atol = 1e-8
        @test length(coef(f1)) == 2

        # Fail loudly: conflicting keywords, unknown levels, nothing left
        @test_throws ArgumentError NodeFactor(:group; level="A", levels=["A"])
        @test_throws ArgumentError NodeFactor(:group; base=-1)
        @test_throws ArgumentError ERGMModel(
            ERGMFormula([NodeFactor(:group; levels=["Z"])]), net)
        single = network(3; directed=false)
        set_vertex_attribute!(single, :g, Dict(1 => "X", 2 => "X", 3 => "X"))
        @test_throws ArgumentError ERGMModel(ERGMFormula([NodeFactor(:g)]), single)
    end

    @testset "NodeMix mixing-matrix cells (statnet ordering and reference)" begin
        # Undirected path 1-2-3-4 with groups A, A, B, B:
        # edge 1-2 is (A,A), 2-3 is (A,B), 3-4 is (B,B)
        net = network(4; directed=false)
        for (i, j) in [(1, 2), (2, 3), (3, 4)]
            add_edge!(net, i, j)
        end
        set_vertex_attribute!(net, :g, Dict(1 => "A", 2 => "A", 3 => "B", 4 => "B"))

        # Single-cell terms
        @test compute(NodeMix(:g, "A", "A"), net) == 1.0
        @test compute(NodeMix(:g, "A", "B"), net) == 1.0
        @test compute(NodeMix(:g, "B", "B"), net) == 1.0
        @test ERGM.name(NodeMix(:g, "A", "B")) == "mix.g.A.B"

        # statnet cell order for undirected: (A,A), (A,B), (B,B); the first
        # cell is dropped by default (levels2 = -1)
        mm = ERGMModel(ERGMFormula([Edges(), NodeMix(:g)]), net)
        @test mm.formula.terms.names == ["edges", "mix.g.A.B", "mix.g.B.B"]
        @test compute_all(mm.formula.terms, net) == [3.0, 1.0, 1.0]

        # levels2 = 0 keeps every cell; positive indices select cells
        mall = ERGMModel(ERGMFormula([NodeMix(:g; levels2=0)]), net)
        @test mall.formula.terms.names == ["mix.g.A.A", "mix.g.A.B", "mix.g.B.B"]
        @test compute_all(mall.formula.terms, net) == [1.0, 1.0, 1.0]
        msel = ERGMModel(ERGMFormula([NodeMix(:g; levels2=[3, 1])]), net)
        @test msel.formula.terms.names == ["mix.g.B.B", "mix.g.A.A"]

        # An unexpanded multi-cell term evaluates to the sum of its cells
        @test compute(NodeMix(:g), net) == 2.0
        @test compute(NodeMix(:g; levels2=0), net) == 3.0

        # Directed 4-cycle 1→2→3→4→1 with the same groups: cells in
        # column-major (tail level, head level) order (A,A),(B,A),(A,B),(B,B)
        dnet = network(4; directed=true)
        for (i, j) in [(1, 2), (2, 3), (3, 4), (4, 1)]
            add_edge!(dnet, i, j)
        end
        set_vertex_attribute!(dnet, :g, Dict(1 => "A", 2 => "A", 3 => "B", 4 => "B"))
        md = ERGMModel(ERGMFormula([NodeMix(:g; levels2=0)]), dnet)
        @test md.formula.terms.names ==
              ["mix.g.A.A", "mix.g.B.A", "mix.g.A.B", "mix.g.B.B"]
        @test compute_all(md.formula.terms, dnet) == [1.0, 1.0, 1.0, 1.0]
        # Direction matters for the off-diagonal cells
        @test compute(NodeMix(:g, "A", "B"), dnet) == 1.0   # 2→3
        @test compute(NodeMix(:g, "B", "A"), dnet) == 1.0   # 4→1
        # ... and the default drops the first cell (A,A)
        mdef = ERGMModel(ERGMFormula([NodeMix(:g)]), dnet)
        @test mdef.formula.terms.names ==
              ["mix.g.B.A", "mix.g.A.B", "mix.g.B.B"]

        # Fitting through the front door works on the expanded statistics
        fit = fit_ergm(net, [Edges(), NodeMix(:g)])
        @test length(coef(fit)) == 3

        # Fail loudly: mixed-sign or out-of-range levels2, empty selection,
        # unknown levels, missing attribute
        @test_throws ArgumentError NodeMix(:g; levels2=[1, -2])
        @test_throws ArgumentError ERGMModel(
            ERGMFormula([NodeMix(:g; levels2=[7])]), net)
        @test_throws ArgumentError ERGMModel(
            ERGMFormula([NodeMix(:g; levels2=[-1, -2, -3])]), net)
        @test_throws ArgumentError ERGMModel(
            ERGMFormula([NodeMix(:g; levels=["A", "Z"])]), net)
        @test_throws ArgumentError ERGMModel(ERGMFormula([NodeMix(:missing_attr)]), net)
    end

    @testset "Randomized cross-validation: change_stat == g(y⁺) − g(y⁻)" begin
        rng = Random.Xoshiro(20260712)
        n = 12

        # Every geometrically weighted term appears at decay 0.0 as well as at
        # a positive decay: the α = 0 branch of the weight formulas is exact
        # (0^0 == 1) and must agree with brute force like any other decay.
        undirected_terms() = [Triangle(), GWESP(0.0), GWESP(0.5), GWESP(1.2),
                              GWDegree(0.0), GWDegree(0.7), Kstar(2), Kstar(3),
                              TwoPath(), Degree(0), Degree(2), Degree(3),
                              GWDSP(0.0), GWDSP(0.5), GWDSP(1.2)]
        directed_terms() = [Triangle(), GWESP(0.0), GWESP(0.5), GWESP(0.5; type=:ITP),
                            GWESP(0.5; type=:OSP), GWESP(0.5; type=:ISP),
                            GWESP(0.5; type=:union), GWESP(1.2; type=:OSP),
                            GWESP(0.0; type=:ITP), GWESP(0.0; type=:union),
                            OStar(2), OStar(3), IStar(2), IStar(3), TwoPath(),
                            IDegree(0), IDegree(2), ODegree(2),
                            GWIDegree(0.0), GWIDegree(0.7),
                            GWODegree(0.0), GWODegree(0.7),
                            GWDSP(0.0), GWDSP(0.5), GWDSP(0.5; type=:ITP),
                            GWDSP(0.5; type=:OSP), GWDSP(0.5; type=:ISP),
                            GWDSP(0.5; type=:union), GWDSP(1.2; type=:OSP),
                            GWDSP(0.0; type=:OSP), GWDSP(0.0; type=:union)]

        for rep in 1:25, directed in (false, true)
            net = network(n; directed=directed)
            p = 0.10 + 0.15 * rand(rng)
            for i in 1:n, j in (directed ? (1:n) : ((i+1):n))
                i == j && continue
                rand(rng) < p && add_edge!(net, i, j)
            end

            terms = directed ? directed_terms() : undirected_terms()
            for _ in 1:10
                i, j = rand(rng, 1:n), rand(rng, 1:n)
                i == j && continue
                for term in terms
                    expected = brute_change_stat(term, net, i, j)
                    actual = change_stat(term, net, i, j)
                    if !isapprox(actual, expected; atol=1e-9)
                        @test (term, rep, directed, i, j, actual, expected) === nothing
                    else
                        @test true
                    end
                end
            end
        end
    end

    @testset "Nodal and dyadic terms match brute force" begin
        for make_net in (fixture_undirected, fixture_directed)
            net = set_test_attrs!(make_net())
            for term in [NodeFactor(:group; level="A"), NodeFactor(:group),
                         NodeCov(:age), NodeMatch(:group),
                         NodeMatch(:group; diff=true, level="A"),
                         NodeMatch(:group; diff=true, level="B"),
                         NodeMismatch(:group), AbsDiff(:age),
                         NodeMix(:group, "A", "A"), NodeMix(:group, "A", "B"),
                         NodeMix(:group, "B", "A"), NodeMix(:group, "B", "B"),
                         NodeMix(:group), NodeMix(:group; levels2=0)]
                @test check_change_stats(term, net) === nothing
            end

            n = nv(net)
            cov_matrix = [Float64(i * j % 7) for i in 1:n, j in 1:n]
            if !is_directed(net)
                cov_matrix = (cov_matrix + cov_matrix') / 2
            end
            @test check_change_stats(EdgeCov(cov_matrix), net) === nothing
        end
    end

    @testset "Mutual term" begin
        net = network(3)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 1)  # Mutual
        add_edge!(net, 2, 3)  # Asymmetric

        mutual_term = Mutual()
        @test compute(mutual_term, net) == 1.0

        # Adding a reciprocating edge creates a mutual dyad
        @test change_stat(mutual_term, net, 3, 2) == 1.0

        # Adding an edge with no reverse tie does not
        @test change_stat(mutual_term, net, 1, 3) == 0.0
    end

    @testset "Triangle term" begin
        net = network(4; directed=false)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        add_edge!(net, 1, 3)

        tri_term = Triangle()
        @test compute(tri_term, net) == 1.0

        add_edge!(net, 3, 4)
        add_edge!(net, 1, 4)
        @test compute(tri_term, net) == 2.0

        # Directed: ttriple + ctriple (statnet definition)
        dnet = network(3; directed=true)
        add_edge!(dnet, 1, 2)
        add_edge!(dnet, 2, 3)
        add_edge!(dnet, 1, 3)  # transitive triple
        @test compute(tri_term, dnet) == 1.0
        rem_edge!(dnet, 1, 3)
        add_edge!(dnet, 3, 1)  # cyclic triple
        @test compute(tri_term, dnet) == 1.0
    end

    @testset "NodeFactor and NodeMatch counts" begin
        net = network(4; directed=false)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        add_edge!(net, 3, 4)
        set_vertex_attribute!(net, :group, Dict(1 => "A", 2 => "A", 3 => "B", 4 => "B"))

        # Endpoint appearances of "A" vertices: vertex 1 (deg 1) + vertex 2
        # (deg 2) = 3; the within-group edge 1–2 counts twice, as in statnet
        @test compute(NodeFactor(:group; level="A"), net) == 3.0

        # 1–2 matches (A,A), 3–4 matches (B,B), 2–3 does not
        @test compute(NodeMatch(:group), net) == 2.0

        # Differential homophily (R's nodematch(diff=TRUE)): one statistic
        # per level, counting only that level's matched edges
        @test compute(NodeMatch(:group; diff=true, level="A"), net) == 1.0
        @test compute(NodeMatch(:group; diff=true, level="B"), net) == 1.0
        @test ERGM.name(NodeMatch(:group; diff=true, level="A")) == "nodematch.group.A"
        @test ERGM.name(NodeMatch(:group)) == "nodematch.group"

        # The old mismatch count lives under an honest name now
        @test compute(NodeMismatch(:group), net) == 1.0
        @test ERGM.name(NodeMismatch(:group)) == "nodemismatch.group"

        # diff=true without a level is R's nodematch(diff=TRUE): a
        # specification that expands into one statistic per level (every
        # sorted level, or `levels=`), under R's labels; before expansion it
        # evaluates to the sum over its levels
        nm = summary_stats(net, [NodeMatch(:group; diff=true)])
        @test keys(nm) == (Symbol("nodematch.group.A"), Symbol("nodematch.group.B"))
        @test collect(values(nm)) == [1.0, 1.0]
        @test compute(NodeMatch(:group; diff=true), net) == compute(NodeMatch(:group), net)
        @test keys(summary_stats(net, [NodeMatch(:group; diff=true, levels=["B"])])) ==
              (Symbol("nodematch.group.B"),)
        @test compute(NodeMatch(:group; diff=true, levels=["B"]), net) == 1.0
        m = ERGMModel(ERGMFormula([Edges(), NodeMatch(:group; diff=true)]), net)
        @test m.formula.terms.names == ["edges", "nodematch.group.A", "nodematch.group.B"]
        @test ERGM.Extension.expand_terms([NodeMatch(:group; diff=true)], net) ==
              [NodeMatch(:group; diff=true, level="A"), NodeMatch(:group; diff=true, level="B")]
        e = try summary_stats(net, [NodeMatch(:group; diff=true, levels=["Z"])]); nothing catch err; err end
        @test e isa ArgumentError && occursin("do not occur", e.msg)
        # a level without diff=true, or both level and levels, is an error
        @test_throws ArgumentError NodeMatch(:group; level="A")
        @test_throws ArgumentError NodeMatch(:group; levels=["A"])
        @test_throws ArgumentError NodeMatch(:group; diff=true, level="A", levels=["A"])
    end

    @testset "Network copies preserve every term statistic" begin
        # Regression for the attribute-dropping copy bug: on a copied network
        # every term (attribute-based or structural) must compute the same
        # statistic as on the original
        for make_net in (fixture_undirected, fixture_directed)
            net = set_test_attrs!(make_net())
            n = nv(net)
            cov_matrix = [Float64(i * j % 7) for i in 1:n, j in 1:n]
            if !is_directed(net)
                cov_matrix = (cov_matrix + cov_matrix') / 2
            end

            terms = AbstractERGMTerm[Edges(), Triangle(),
                                     TwoPath(), GWESP(0.5),
                                     NodeFactor(:group),
                                     NodeFactor(:group; level="A"),
                                     NodeCov(:age), NodeMatch(:group),
                                     NodeMatch(:group; diff=true, level="A"),
                                     NodeMismatch(:group),
                                     AbsDiff(:age), EdgeCov(cov_matrix)]
            # Star/degree terms are direction-specific, as in R ergm
            if is_directed(net)
                append!(terms, [Mutual(), OStar(2), IStar(3), GWODegree(0.5), GWIDegree(0.5)])
            else
                append!(terms, [Kstar(2), Kstar(3), GWDegree(0.5)])
            end

            for term in terms
                @test compute(term, copy(net)) == compute(term, net)
                @test compute(term, ERGM._copy_network(net)) == compute(term, net)
            end
        end

        # The default MCMC start network inherits the observed network's
        # attributes and settings
        net = set_test_attrs!(fixture_undirected())
        start = ERGM._random_network(net)
        @test is_directed(start) == is_directed(net)
        @test get_vertex_attribute(start, :group) == get_vertex_attribute(net, :group)
        @test get_vertex_attribute(start, :age) == get_vertex_attribute(net, :age)

        # ... and so do the networks returned by sample_networks
        model = ERGMModel(ERGMFormula([Edges(), NodeMatch(:group)]), net)
        sims = sample_networks(model, [-1.0, 0.5]; n_sim=3, burnin=200, interval=20)
        @test all(get_vertex_attribute(s, :group) == get_vertex_attribute(net, :group)
                  for s in sims)
    end

    @testset "MPLE analytic check (edges-only = logit density)" begin
        # Undirected ring: 10 edges over 45 dyads
        net = network(10; directed=false)
        for i in 1:9
            add_edge!(net, i, i + 1)
        end
        add_edge!(net, 1, 10)

        result = fit_ergm(net, [Edges()])
        d = 10 / 45
        @test result.method == :mple
        @test result.converged
        @test result.coefficients[1] ≈ log(d / (1 - d)) atol = 1e-4

        # Directed version: 10 edges over 90 dyads
        dnet = network(10; directed=true)
        for i in 1:9
            add_edge!(dnet, i, i + 1)
        end
        add_edge!(dnet, 10, 1)
        dresult = fit_ergm(dnet, [Edges()])
        dd = 10 / 90
        @test dresult.coefficients[1] ≈ log(dd / (1 - dd)) atol = 1e-4
    end

    @testset "Sampler targets exp(θ'g): edges-only stationary mean" begin
        Random.seed!(20260706)

        # Under an edges-only ERGM each dyad is independent Bernoulli(σ(θ)),
        # so the expected edge count is n_dyads · σ(θ)
        n = 8
        n_dyads = n * (n - 1) ÷ 2
        θ = [-1.0]

        net = network(n; directed=false)
        add_edge!(net, 1, 2)  # arbitrary observed network to size the model
        model = ERGMModel(ERGMFormula([Edges()]), net)

        sims = sample_networks(model, θ; n_sim=400, burnin=2000, interval=25)
        mean_edges = mean(Float64(ne(s)) for s in sims)
        expected = n_dyads / (1 + exp(1.0))

        @test length(sims) == 400
        @test all(nv(s) == n for s in sims)
        @test mean_edges ≈ expected atol = 0.6

        # Non-trivial θ sign check: positive θ must yield denser networks
        sims_pos = sample_networks(model, [1.0]; n_sim=200, burnin=2000, interval=25)
        @test mean(Float64(ne(s)) for s in sims_pos) > mean_edges
    end

    @testset "MCMLE agrees with MPLE for dyad-independent model" begin
        Random.seed!(99)
        flo = florentine_marriage()
        # Binary wealth split for a homophily term. Edges + NodeMatch is
        # dyad-independent, so MPLE is the exact MLE and MCMLE must agree
        # with it up to Monte Carlo error
        wealth = get_vertex_attribute(flo, :wealth)
        rich = Dict(v => (w > 40 ? "rich" : "poor") for (v, w) in wealth)
        set_vertex_attribute!(flo, :rich, rich)

        terms = [Edges(), NodeMatch(:rich)]
        mple_fit = fit_ergm(flo, terms; method=:mple)
        mcmle_fit = fit_ergm(flo, terms; method=:mcmle, n_samples=800)

        @test mple_fit.converged
        @test mcmle_fit.converged
        @test mcmle_fit.coefficients ≈ mple_fit.coefficients atol = 0.3
    end

    @testset "Removed development-era keywords are refused, not ignored" begin
        net = network(6; directed=false)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        add_edge!(net, 3, 4)
        model = ERGMModel(ERGMFormula([Edges()]), net)
        # `mcmle(...; tol=)` was a no-op and `max_iter` a spelling of `maxiter`;
        # both are gone (0.2.0 is the first release) and are MethodErrors of
        # the estimator, and `fit_ergm` names the estimator that takes `tol`
        @test_throws MethodError mcmle(model; tol=1e-4, n_samples=100, maxiter=2)
        @test_throws MethodError mcmle(model; max_iter=2, n_samples=100)
        e = try fit_ergm(net, [Edges()]; method=:mcmle, tol=1e-4); nothing catch err; err end
        @test e isa ArgumentError && occursin("`tol`", e.msg) && occursin("method=:mple", e.msg)
        e = try fit_ergm(net, [Edges()]; method=:mcmle, max_iter=2); nothing catch err; err end
        @test e isa ArgumentError && occursin("`max_iter`", e.msg) &&
              occursin("keywords of each estimator", e.msg)
    end

    @testset "MCMLE regression: attribute terms survive network copies" begin
        Random.seed!(20260711)

        # Homophilous undirected network with a binary vertex attribute :g
        net = network(12; directed=false)
        set_vertex_attribute!(net, :g,
            Dict(v => (v % 2 == 0 ? "b" : "a") for v in 1:12))
        ties = [(1, 3), (1, 5), (3, 5), (5, 7), (7, 9), (9, 11), (1, 11),
                (2, 4), (4, 6), (6, 8), (8, 10), (10, 12),
                (1, 2), (5, 6), (9, 10)]
        for (i, j) in ties
            add_edge!(net, i, j)
        end

        obs_nodematch = compute(NodeMatch(:g), net)
        @test obs_nodematch == 12.0

        result = fit_ergm(net, [Edges(), NodeMatch(:g)]; method=:mcmle,
                          n_samples=800)
        @test result.converged

        # The sampled mean of the nodematch statistic at the fitted θ must
        # match the observed statistic. This is exactly what the old
        # attribute-dropping _copy_network broke: sampled nodematch was
        # identically zero while the observed count was 12
        sampled_nodematch = mean(result.mcmc_samples[:, 2])
        @test sampled_nodematch > 0.0
        @test sampled_nodematch ≈ obs_nodematch atol = 2.0

        # simulate_ergm must produce networks with nonzero nodematch
        sims = simulate_ergm(result; n_sim=20, burnin=2000, interval=200)
        sim_nodematch = [compute(NodeMatch(:g), s) for s in sims]
        @test mean(sim_nodematch) > 0.0

        # gof consumes those simulations without error
        gof_result = gof(result; n_sim=10, stats=[:degree])
        @test gof_result isa GOFResult
        @test [s.name for s in gof_result.statistics] == ["degree"]
    end

    @testset "Simulation smoke test" begin
        Random.seed!(7)
        net = network(5)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)

        result = fit_ergm(net, [Edges()])
        sims = simulate_ergm(result; n_sim=2, burnin=100, interval=10)

        @test length(sims) == 2
        @test all(nv(s) == 5 for s in sims)
        @test all(is_directed(s) for s in sims)
    end

    @testset "GOF" begin
        Random.seed!(11)
        net = network(5; directed=false)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        add_edge!(net, 3, 4)
        add_edge!(net, 4, 5)

        result = fit_ergm(net, [Edges()])
        gof_result = gof(result; n_sim=20, stats=[:degree, :esp, :distance])

        # gof extends NetworkCore.jl's shared generic and returns the shared
        # GOFResult container
        @test gof_result isa GOFResult
        @test n_simulations(gof_result) == 20
        @test [s.name for s in gof_result.statistics] ==
              ["degree", "esp", "distance"]
        for stat_gof in gof_result.statistics
            @test length(stat_gof.observed) > 0
            # Two-sided Monte Carlo p-values live in [0, 1] (and, with the
            # (1 + k)/(N + 1) estimator, are never exactly zero)
            @test all(0.0 .< stat_gof.p_values .<= 1.0)
        end

        # ... and renders through the shared formatted display
        out = sprint(show, gof_result)
        @test occursin("Goodness-of-fit assessment: ERGM", out)
        @test occursin("MC p-value", out)
        @test occursin("Based on 20 simulated networks", out)
    end

    @testset "GOF: directed degree split and typed ESP" begin
        Random.seed!(13)
        dnet = fixture_directed()
        fit = fit_ergm(dnet, [Edges()])
        g = gof(fit; n_sim=10, stats=[:degree, :esp], burnin=500, interval=50)
        panel(gr, pname) = only(s for s in gr.statistics if s.name == pname)
        panel_names(gr) = [s.name for s in gr.statistics]

        # For directed networks :degree splits into in- and out-degree panels
        @test "idegree" in panel_names(g)
        @test "odegree" in panel_names(g)
        @test !("degree" in panel_names(g))

        # Observed distributions match hand-counted in-/out-degrees
        n = nv(dnet)
        idegs = [length(inneighbors(dnet, v)) for v in vertices(dnet)]
        odegs = [length(outneighbors(dnet, v)) for v in vertices(dnet)]
        @test panel(g, "idegree").observed == [count(==(d), idegs) for d in 0:(n-1)]
        @test panel(g, "odegree").observed == [count(==(d), odegs) for d in 0:(n-1)]
        # ... which differ from each other in general, and here at least
        # come from different vectors (fixture has equal marginals, so also
        # check a panel each can be requested on its own)
        gi = gof(fit; n_sim=5, stats=[:idegree], burnin=300, interval=30)
        @test panel_names(gi) == ["idegree"]

        # The directed ESP distribution uses OTP shared partners by default
        # (netA: edge 1→2 has two outgoing two-paths, the other four none)
        netA = network(5; directed=true)
        for (i, j) in [(1, 3), (3, 2), (1, 4), (4, 2), (1, 2)]
            add_edge!(netA, i, j)
        end
        espA = ERGM._gof_esp(netA, [netA])
        @test espA.observed == [4, 0, 1]
        # ... and the union (pre-0.2) definition remains available
        espU = ERGM._gof_esp(netA, [netA]; type=:union)
        @test espU.observed == [0, 4, 1]
        @test_throws ArgumentError ERGM._gof_esp(netA, [netA]; type=:XYZ)

        # esp_type flows through the public gof
        gu = gof(fit; n_sim=5, stats=[:esp], burnin=300, interval=30,
                 rng=Random.Xoshiro(9), esp_type=:union)
        @test panel_names(gu) == ["esp"]

        # Undirected networks keep a single :degree panel
        unet = fixture_undirected()
        ufit = fit_ergm(unet, [Edges()])
        ug = gof(ufit; n_sim=5, stats=[:degree], burnin=300, interval=30)
        @test panel_names(ug) == ["degree"]
    end

    @testset "StatsAPI interface" begin
        flo = florentine_marriage()
        fit = fit_ergm(flo, [Edges(), NodeCov(:wealth)])

        # ERGM extends the StatsAPI generics, so with `using ERGM, StatsBase`
        # both packages export the very same functions
        @test coef === StatsBase.coef
        @test stderror === StatsBase.stderror
        @test vcov === StatsBase.vcov

        @test coef(fit) == fit.coefficients
        @test stderror(fit) == fit.std_errors
        @test vcov(fit) == fit.vcov
        @test loglikelihood(fit) == fit.loglik
        @test aic(fit) == fit.aic
        @test bic(fit) == fit.bic
        @test nobs(fit) == 120  # 16 · 15 / 2 undirected dyads
        @test dof(fit) == 2
    end

    @testset "Fail loudly: missing attributes" begin
        flo = florentine_marriage()  # has :wealth

        # Typo'd attribute in an attribute-based term errors at model
        # construction, naming the missing attribute and listing available
        for bad_term in (NodeCov(:welth), NodeFactor(:welth), NodeMatch(:welth),
                         NodeMismatch(:welth), AbsDiff(:welth))
            err = try
                ERGMModel(ERGMFormula([Edges(), bad_term]), flo)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("welth", err.msg)      # names the missing attribute
            @test occursin(":wealth", err.msg)    # lists the available ones
        end

        # ... and through the fit_ergm front door too
        @test_throws ArgumentError fit_ergm(flo, [Edges(), NodeCov(:welth)])

        # Correct attribute passes
        @test ERGMModel(ERGMFormula([Edges(), NodeCov(:wealth)]), flo) isa ERGMModel

        # An attribute that exists but is not set on EVERY vertex is refused
        # too (statnet: "Attribute has missing data, which is not currently
        # supported by ergm"): naming the term, the attribute, how many
        # vertices lack a value and which. Before 0.2 the missing vertices
        # were silently zero-filled into the design.
        part = copy(flo)
        set_vertex_attribute!(part, :x, Dict(v => Float64(v) for v in 1:16 if !(v in (3, 7, 12))))
        set_vertex_attribute!(part, :g, Dict(v => (isodd(v) ? "A" : "B") for v in 1:16 if !(v in (3, 7, 12))))
        for (bad_term, attr) in ((NodeCov(:x), :x), (AbsDiff(:x), :x),
                                 (NodeMatch(:g), :g), (NodeMismatch(:g), :g),
                                 (NodeFactor(:g), :g), (NodeMix(:g), :g),
                                 (NodeMatch(:g; diff=true, level="A"), :g),
                                 (NodeMix(:g, "A", "B"), :g))
            err = try
                ERGMModel(ERGMFormula([Edges(), bad_term]), part)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("'$(ERGM.name(bad_term))'", err.msg)   # names the term
            @test occursin(":$attr", err.msg)                     # and the attribute
            @test occursin("3 of 16 vertices", err.msg)           # the count
            @test occursin("vertices 3, 7, 12", err.msg)          # and the ids
            @test occursin("statnet refuses NA", err.msg)
        end
        @test_throws ArgumentError fit_ergm(part, [Edges(), NodeCov(:x)])

        # A value of `missing` (an NA on its way in from R / DataFrames) is
        # no value either
        na = copy(flo)
        set_vertex_attribute!(na, :x, Dict(v => (v == 5 ? missing : Float64(v)) for v in 1:16))
        err = try; ERGMModel(ERGMFormula([Edges(), NodeCov(:x)]), na); nothing; catch e; e; end
        @test err isa ArgumentError && occursin("1 of 16 vertices have no value (vertices 5)", err.msg)

        # The raw terms no longer zero-fill either: a direct compute /
        # change_stat on incomplete data throws instead of returning a number
        for t in (NodeCov(:x), AbsDiff(:x))
            e = try; compute(t, part); nothing; catch err; err; end
            @test e isa ArgumentError && occursin("has no value", e.msg)
            e = try; change_stat(t, part, 3, 1); nothing; catch err; err; end
            @test e isa ArgumentError && occursin("vertex 3 has no value", e.msg)
        end
        # ... while the same term on complete data is unchanged
        @test compute(NodeCov(:wealth), flo) == 2168.0
        @test fit_ergm(flo, [Edges(), NodeCov(:wealth)]) isa ERGMResult

        # A String-valued attribute in a numeric-covariate term names the
        # offending value and points at the categorical terms, instead of
        # leaking the inner `convert` MethodError
        str = copy(flo)
        set_vertex_attribute!(str, :label, Dict(v => "fam$v" for v in 1:16))
        for t in (NodeCov(:label), AbsDiff(:label))
            e = try; fit_ergm(str, [Edges(), t]); nothing; catch err; err; end
            @test e isa ArgumentError
            @test occursin("NUMERIC", e.msg) && occursin("\"fam1\"", e.msg)
            @test occursin("String", e.msg) && occursin("NodeFactor", e.msg)
        end
    end

    @testset "Fail loudly: direction-incompatible terms" begin
        # Mutual on an undirected network is an error at model construction
        # (as in R ergm), not a structurally-zero statistic
        unet = fixture_undirected()
        err = try
            ERGMModel(ERGMFormula([Edges(), Mutual()]), unet)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("mutual", err.msg)
        @test occursin("directed", err.msg)
        @test_throws ArgumentError fit_ergm(unet, [Edges(), Mutual()])

        # Mutual on a directed network is fine
        dnet = fixture_directed()
        @test ERGMModel(ERGMFormula([Edges(), Mutual()]), dnet) isa ERGMModel

        # Kstar / GWDegree / Degree are undirected-only, as in R ergm; the
        # refusal names the directed variants to use instead. Before 0.2 a
        # directed network silently received OUT-stars under "kstar2".
        for (t, hint) in ((Kstar(2), "OStar(2)"), (Kstar(2), "IStar(2)"),
                          (GWDegree(0.5), "GWODegree(0.5)"), (GWDegree(0.5), "GWIDegree(0.5)"),
                          (Degree(1), "ODegree"), (Degree(1), "IDegree"))
            err = try
                ERGMModel(ERGMFormula([Edges(), t]), dnet)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("'$(ERGM.name(t))'", err.msg)
            @test occursin("undirected", err.msg)
            @test occursin(hint, err.msg)
            @test occursin("directed==TRUE", err.msg)   # R's own wording, quoted
            @test requires_undirected(t)
        end
        @test_throws ArgumentError fit_ergm(dnet, [Edges(), Kstar(2)])
        @test_throws ArgumentError fit_ergm(dnet, [Edges(), GWDegree(0.5)])

        # ... and the terms refuse a directed network themselves, so a caller
        # bypassing ERGMModel (a variant lifting a term) cannot get the old
        # silent out-star statistic either
        for t in (Kstar(2), GWDegree(0.5), Degree(1))
            @test_throws ArgumentError compute(t, dnet)
            @test_throws ArgumentError change_stat(t, dnet, 1, 2)
        end

        # The directed counterparts are directed-only, like IDegree/ODegree
        for t in (OStar(2), IStar(2))
            @test requires_directed(t)
            @test_throws ArgumentError ERGMModel(ERGMFormula([Edges(), t]), unet)
            @test ERGMModel(ERGMFormula([Edges(), t]), dnet) isa ERGMModel
        end
    end

    @testset "Public term-trait protocol" begin
        # Built-in declarations
        @test required_vertex_attributes(NodeCov(:age)) == (:age,)
        @test required_vertex_attributes(NodeMix(:group)) == (:group,)
        @test required_vertex_attributes(Edges()) == ()
        @test required_edge_attributes(Edges()) == ()
        @test requires_directed(Mutual()) && requires_directed(IDegree(1))
        @test !requires_directed(Edges())
        @test requires_undirected(Degree(1)) && !requires_undirected(ODegree(1))
        @test !is_dyad_dependent(Edges()) && is_dyad_dependent(Triangle())
        # Terms compute their statistic from face values; the missing-data
        # treatment lives in the estimators (mple), not in the terms
        @test !supports_missing(Edges())
        @test supports_missing(mple)

        # The pre-0.2 private names are gone: methods are declared on the
        # public traits (TERGM.jl's `requires_directed(::Delrecip) = true`)
        @test !isdefined(ERGM, :_requires_directed)
        @test !isdefined(ERGM, :_vertex_attribute)

        # Materialized twins keep their source term's declarations
        dnet = set_test_attrs!(fixture_directed())
        mat = ERGMModel(ERGMFormula([NodeCov(:age)]), dnet).formula.terms[1]
        @test required_vertex_attributes(mat) == (:age,)
        @test !is_dyad_dependent(mat)

        # A third-party term participates in validation on its declarations
        # alone: accepted when the network satisfies them ...
        set_edge_attribute!(dnet, :w, 1, 2, 1.0)
        term = ForeignTerm(:age, :w)
        model = ERGMModel(ERGMFormula([Edges(), term]), dnet)
        @test model.formula.terms.names == ["edges", "foreign.age"]

        # ... rejected, with the standard message, when a declared VERTEX
        # attribute is absent ...
        bad_v = try
            ERGMModel(ERGMFormula([ForeignTerm(:no_such, :w)]), dnet)
            nothing
        catch e
            e
        end
        @test bad_v isa ArgumentError
        @test occursin("vertex attribute :no_such", bad_v.msg)

        # ... when a declared EDGE attribute is absent ...
        bad_e = try
            ERGMModel(ERGMFormula([ForeignTerm(:age, :no_such_w)]), dnet)
            nothing
        catch e
            e
        end
        @test bad_e isa ArgumentError
        @test occursin("edge attribute :no_such_w", bad_e.msg)

        # ... and when its direction requirement is not met
        unet = set_test_attrs!(fixture_undirected())
        set_edge_attribute!(unet, :w, 1, 2, 1.0)
        bad_d = try
            ERGMModel(ERGMFormula([ForeignTerm(:age, :w)]), unet)
            nothing
        catch e
            e
        end
        @test bad_d isa ArgumentError
        @test occursin("directed", bad_d.msg)
    end

    @testset "Fail loudly: mcmc_diagnostics on an MPLE fit" begin
        flo = florentine_marriage()
        fit = fit_ergm(flo, [Edges()])  # MPLE, no MCMC samples
        err = try
            mcmc_diagnostics(fit)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("mple", err.msg)
        @test occursin("method=:mcmle", err.msg)

        # An MCMLE fit is accepted
        Random.seed!(1)
        mfit = fit_ergm(flo, [Edges()]; method=:mcmle, n_samples=200, maxiter=3,
                        max_n_samples=200)
        diag = mcmc_diagnostics(mfit)
        @test length(diag.autocorrelation) == 1
        @test diag.n_samples == 200

        # Geyer initial-sequence ESS and Geweke convergence fields
        @test diag isa MCMCDiagnostics
        @test length(diag.ess_geyer) == 1
        @test 2.0 <= diag.ess_geyer[1] <= 2 * diag.n_samples
        @test length(diag.geweke_z) == 1
        @test isfinite(diag.geweke_z[1])
        @test 0.0 < diag.geweke_p[1] <= 1.0
        @test diag.chain_lengths == [200]

        # `show` is the family's table, not a NamedTuple of 16-digit floats
        printed = sprint(show, diag)
        @test occursin("MCMC diagnostics: 200 draws in 1 chain\n", printed)
        @test occursin("lag-1 AC", printed) && occursin("ESS (lag-1)", printed)
        @test occursin("ESS (Geyer)", printed) && occursin("Geweke z", printed)
        @test occursin("Geweke p", printed)
        @test occursin("\nedges ", printed)
        @test !occursin("term_names", printed)

        # Several chains are diagnosed CHAIN BY CHAIN, never across the seams
        # of the concatenated sample: the Geyer ESS is the sum of
        # the per-chain ESSs — exactly the MCMLE's own n_eff estimator — the
        # lag-1 column is length-weighted, and the Geweke test is the chain
        # of largest |z|
        # (fixed-size TNT sampling: the chain bookkeeping below is that of
        # `n_samples` draws split over `n_chains`, which the ESS-adaptive
        # default replaces by a chain-length rule of its own)
        four = fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mcmle, n_samples=800,
                        n_chains=4, bridge_rungs=0, rng=Random.Xoshiro(5),
                        proposal=:tnt, effective_size=nothing)
        d4 = mcmc_diagnostics(four)
        @test d4.chain_lengths == four.chain_lengths == [200, 200, 200, 200]
        @test occursin("800 draws in 4 chains of 200, 200, 200, 200", sprint(show, d4))
        S = four.mcmc_samples
        blocks = [S[(200c - 199):(200c), :] for c in 1:4]
        for j in 1:2
            ge = sum(ERGM._geyer_ess(b[:, j]) for b in blocks)
            @test d4.ess_geyer[j] ≈ ge atol = 1e-9
            ρs = [cor(b[1:end-1, j], b[2:end, j]) for b in blocks]   # ≈ lag-1 AC per chain
            @test d4.autocorrelation[j] ≈ mean(ρs) atol = 0.02
            @test d4.effective_sample_size[j] ≈ sum(200 * (1 - ρ) / (1 + ρ) for ρ in ρs) rtol = 0.05
            zs = [ERGM._geweke_z(b[:, j]) for b in blocks]
            @test d4.geweke_z[j] ≈ zs[argmax(abs.(zs))] atol = 1e-9   # summation order may differ in the last bits
        end
        @test d4.ess_geyer[argmin(d4.ess_geyer)] ≈ four.mcmc_convergence.n_eff atol = 1e-9
        # A whole-sample computation would differ (the seams are real)
        @test !(mcmc_diagnostics(four).ess_geyer ≈ [ERGM._geyer_ess(S[:, j]) for j in 1:2])
        @test Base.propertynames(d4) == fieldnames(MCMCDiagnostics)
    end

    @testset "Geyer ESS and Geweke diagnostics (synthetic chains)" begin
        # iid chain: ESS ≈ n
        x = randn(Random.Xoshiro(2), 4000)
        @test 2500 < ERGM._geyer_ess(x) < 5500

        # AR(1) chain with ρ = 0.9: true ESS factor (1−ρ)/(1+ρ) ≈ 0.053,
        # so ESS ≈ 210 of 4000 — Geyer must see far less than n, and less
        # than the optimistic lag-1 estimate would ever be forced to admit
        rng = Random.Xoshiro(4)
        y = zeros(4000)
        y[1] = randn(rng)
        for t in 2:4000
            y[t] = 0.9 * y[t-1] + randn(rng)
        end
        @test 50 < ERGM._geyer_ess(y) < 800

        # Geweke: stationary chain passes, a drifting chain fails loudly
        @test abs(ERGM._geweke_z(x)) < 4
        drift = collect(range(0, 5; length=2000)) .+
                0.1 .* randn(Random.Xoshiro(5), 2000)
        @test abs(ERGM._geweke_z(drift)) > 5
        @test z_pvalues([ERGM._geweke_z(drift)])[1] < 1e-6

        # Constant chain degrades gracefully
        @test ERGM._geyer_ess(fill(3.0, 100)) == 100.0
        @test ERGM._geweke_z(fill(3.0, 100)) == 0.0
    end

    @testset "P-values use ccdf and never underflow to zero for finite z" begin
        # 2(1 − cdf) underflows to exactly 0 at |z| ≈ 8.3; ccdf is accurate
        @test z_pvalues([10.0])[1] > 0.0
        @test z_pvalues([10.0])[1] ≈ 1.5239706048320995e-23
        @test z_pvalues([30.0])[1] > 0.0

        # Beyond |z| ≈ 38 even the ccdf tail underflows Float64, so finite
        # z-statistics are floored at floatmin — a finite estimate never
        # reports an exact-zero p-value
        @test z_pvalues([40.0])[1] > 0.0
        @test z_pvalues([40.0])[1] == floatmin(Float64)

        # Sanity at the center and edge cases
        @test z_pvalues([0.0])[1] ≈ 1.0
        @test z_pvalues([-2.0]) == z_pvalues([2.0])
        @test isnan(z_pvalues([NaN])[1])
        @test z_pvalues([Inf])[1] == 0.0
    end

    @testset "Attribute snapshots (materialized terms) match the originals" begin
        for make_net in (fixture_undirected, fixture_directed)
            net = set_test_attrs!(make_net())
            raw = AbstractERGMTerm[Edges(), Triangle(),
                                   NodeFactor(:group), NodeFactor(:group; level="A"),
                                   NodeCov(:age), NodeCov(:age; transform=:log),
                                   NodeMatch(:group),
                                   NodeMatch(:group; diff=true, level="A"),
                                   NodeMatch(:group; diff=true, level="Z"),  # absent level
                                   NodeMismatch(:group), AbsDiff(:age),
                                   AbsDiff(:age; pow=2.0),
                                   NodeMix(:group, "A", "B")]
            model = ERGMModel(ERGMFormula(raw), net)
            mat = model.formula.terms

            # Nodal terms were actually materialized (typed snapshots), and
            # every name is preserved. The multi-level NodeFactor(:group)
            # resolves to its per-level statistics at model construction —
            # with levels {A, B} and the first level dropped (statnet's
            # default) that is the single statistic "nodefactor.group.B"
            @test any(t -> t isa ERGM.MaterializedNodeCov, mat.terms)
            @test any(t -> t isa ERGM.MaterializedNodeMatch, mat.terms)
            @test any(t -> t isa ERGM.MaterializedNodeMix, mat.terms)
            expected_names = [ERGM.name(t) for t in raw]
            expected_names[3] = "nodefactor.group.B"
            @test mat.names == expected_names

            # Statistics and change statistics agree exactly with the
            # original terms on every dyad
            @test compute_all(mat, net) == compute_all(TermSet(raw), net)
            n = nv(net)
            for i in 1:n, j in (is_directed(net) ? (1:n) : ((i+1):n))
                i == j && continue
                @test change_stat_all(mat, net, i, j) ==
                      change_stat_all(TermSet(raw), net, i, j)
            end

            # Dyad-dependence classification survives materialization
            @test is_dyad_dependent(mat.terms[2])          # Triangle
            @test !any(is_dyad_dependent, mat.terms[3:end]) # nodal terms
            @test !is_dyad_dependent(mat.terms[1])          # Edges
        end
    end

    @testset "mh_sample: public single-chain sampler" begin
        net = set_test_attrs!(fixture_undirected())
        model = ERGMModel(ERGMFormula([Edges(), NodeMatch(:group)]), net)
        θ = [-1.0, 0.5]

        out = mh_sample(model, θ; n_samples=10, burnin=200, interval=20,
                        rng=Random.Xoshiro(42), return_networks=true)
        @test size(out.stats) == (10, 2)
        @test length(out.networks) == 10

        # Recorded statistics are exactly the statistics of the recorded
        # networks, and attributes survive
        for k in 1:10
            @test out.stats[k, :] == compute_all(model.formula.terms, out.networks[k])
            @test get_vertex_attribute(out.networks[k], :group) ==
                  get_vertex_attribute(net, :group)
        end

        # networks === nothing unless requested
        out2 = mh_sample(model, θ; n_samples=5, burnin=100, interval=10)
        @test out2.networks === nothing
        @test size(out2.stats) == (5, 2)

        # Same rng seed => identical sample statistics
        a = mh_sample(model, θ; n_samples=20, burnin=200, interval=10,
                      rng=Random.Xoshiro(7)).stats
        b = mh_sample(model, θ; n_samples=20, burnin=200, interval=10,
                      rng=Random.Xoshiro(7)).stats
        @test a == b

        # The observed network is never mutated
        @test ne(net) == 8

        # θ-length mismatch errors
        @test_throws ArgumentError mh_sample(model, [-1.0]; n_samples=2)
    end

    @testset "RNG reproducibility across the sampling APIs" begin
        net = set_test_attrs!(fixture_undirected())
        model = ERGMModel(ERGMFormula([Edges(), NodeMatch(:group)]), net)
        θ = [-1.0, 0.5]

        # sample_networks: identical networks for identical seeds, for both
        # single- and multi-chain runs (chains are seeded deterministically
        # from the caller rng, so results are thread-count-independent)
        for n_chains in (1, 3)
            s1 = sample_networks(model, θ; n_sim=6, burnin=200, interval=20,
                                 rng=Random.Xoshiro(99), n_chains=n_chains)
            s2 = sample_networks(model, θ; n_sim=6, burnin=200, interval=20,
                                 rng=Random.Xoshiro(99), n_chains=n_chains)
            @test length(s1) == 6
            @test [as_matrix(a) for a in s1] == [as_matrix(b) for b in s2]
        end

        # gof with a seeded rng is fully reproducible
        fit = fit_ergm(net, [Edges()])
        g1 = gof(fit; n_sim=8, stats=[:degree], burnin=300, interval=30,
                 rng=Random.Xoshiro(3))
        g2 = gof(fit; n_sim=8, stats=[:degree], burnin=300, interval=30,
                 rng=Random.Xoshiro(3))
        @test g1.statistics[1].simulated == g2.statistics[1].simulated
        @test g1.statistics[1].p_values == g2.statistics[1].p_values

        # mcmle with a seeded rng gives identical fits
        flo = florentine_marriage()
        f1 = fit_ergm(flo, [Edges()]; method=:mcmle, n_samples=200, maxiter=3,
                      rng=Random.Xoshiro(1234), bridge_rungs=4, bridge_samples=100)
        f2 = fit_ergm(flo, [Edges()]; method=:mcmle, n_samples=200, maxiter=3,
                      rng=Random.Xoshiro(1234), bridge_rungs=4, bridge_samples=100)
        @test f1.coefficients == f2.coefficients
        @test f1.loglik == f2.loglik
        @test f1.mcmc_samples == f2.mcmc_samples
    end

    @testset "MPLE parametric-bootstrap standard errors" begin
        flo = florentine_marriage()

        # Dyad-independent model: the pseudo-likelihood is the likelihood,
        # so bootstrap SEs must roughly reproduce the (correct) Hessian SEs
        mh = fit_ergm(flo, [Edges()])
        mb = fit_ergm(flo, [Edges()]; se=:bootstrap, n_boot=60,
                      rng=Random.Xoshiro(2026))
        @test mh.se_type == :hessian
        @test mb.se_type == :bootstrap
        @test mb.coefficients == mh.coefficients        # same point estimates
        @test 0.5 < mb.std_errors[1] / mh.std_errors[1] < 2.0
        @test size(vcov(mb)) == (1, 1)
        @test sqrt(vcov(mb)[1, 1]) ≈ mb.std_errors[1] atol = 1e-12

        # Reproducible under the same seed
        mb2 = fit_ergm(flo, [Edges()]; se=:bootstrap, n_boot=60,
                       rng=Random.Xoshiro(2026))
        @test mb2.std_errors == mb.std_errors
        @test mb.boot_replicates isa Matrix{Float64} && size(mb.boot_replicates) == (60, 1)
        @test mh.boot_replicates === nothing
        @test isempty(approximations(mb))

        # Invalid se choice fails loudly
        @test_throws ArgumentError fit_ergm(flo, [Edges()]; se=:jackknife)

        # The dyad-dependent path the bootstrap exists for: a
        # simulated replicate with no triangle has no finite MPLE for
        # `triangle` (-Inf under R's drop), and its row used to enter
        # `cov` — every standard error, vcov entry and p-value came back NaN
        # on the README's and the estimation guide's own recipe. Such
        # replicates are now excluded, ONE warning says how many (about the
        # simulated replicates, never "observed statistic(s)"), and the
        # result records them.
        logs, tb = Test.collect_test_logs() do
            ergm(flo, [Edges(), Triangle()]; method=:mple, se=:bootstrap, n_boot=60, rng=Random.Xoshiro(1))
        end
        @test all(isfinite, stderror(tb)) && all(isfinite, vcov(tb)) && all(isfinite, confint(tb))
        @test all(isfinite, tb.p_values)
        @test size(tb.boot_replicates) == (60, 2)
        n_dropped = count(b -> !all(isfinite, tb.boot_replicates[b, :]), 1:60)
        @test n_dropped >= 1                        # this seed does hit the boundary
        @test length(logs) == 1
        @test occursin("$n_dropped of the 60 bootstrap refits", logs[1].message)
        @test occursin("simulated replicates", logs[1].message)
        @test !occursin("observed statistic(s)", logs[1].message)
        @test any(occursin("$n_dropped of the 60 parametric-bootstrap refits", a)
                  for a in approximations(tb))
        @test occursin("$n_dropped of the 60 parametric-bootstrap refits", sprint(show, tb))
        # ... and all three say what the exclusion does to the standard errors
        bias = "the excluded replicates are the extreme ones, so the standard errors are biased downward"
        @test occursin(bias, logs[1].message)
        @test any(occursin(bias, a) for a in approximations(tb))
        @test occursin(bias, sprint(show, tb))
        @test NetworkCore.check_statsapi(tb; strict=true) !== nothing
        # The covariance is that of the finite rows
        ok = [all(isfinite, tb.boot_replicates[b, :]) for b in 1:60]
        @test vcov(tb) ≈ cov(tb.boot_replicates[ok, :]) atol = 1e-12
        # The guide's recipe (edges + gwesp + nodematch) is finite too
        gnet = copy(flo)
        set_vertex_attribute!(gnet, :gender, Dict(v => (isodd(v) ? "F" : "M") for v in 1:16))
        gb = ergm(gnet, [Edges(), GWESP(0.5), NodeMatch(:gender)]; method=:mple, se=:bootstrap, n_boot=30,
                  rng=Random.Xoshiro(1))
        @test all(isfinite, stderror(gb)) && all(isfinite, vcov(gb))
    end

    @testset "show() prints a pseudo-likelihood caveat only under dyad dependence" begin
        flo = florentine_marriage()

        # Dyad-dependent formula + MPLE => caveat (the explicit naive opt-in;
        # the default withholds z and p and says so, pinned below)
        dep_fit = fit_ergm(flo, [Edges(), Triangle()]; method=:mple, se=:hessian)
        dep_out = sprint(show, dep_fit)
        @test occursin("pseudolikelihood", dep_out)
        @test occursin("suspect", dep_out)
        @test occursin("not reported (NaN)", sprint(show, fit_ergm(flo, [Edges(), Triangle()]; method=:mple)))

        # The coefficient table renders through the shared NetworkCore.jl
        # presentation layer (R-style columns and significance codes)
        @test occursin("Pr(>|z|)", dep_out)
        @test occursin("Signif. codes:", dep_out)
        @test occursin("edges", dep_out)

        # Dyad-independent formula => no caveat
        ind_fit = fit_ergm(flo, [Edges(), NodeCov(:wealth)])
        ind_out = sprint(show, ind_fit)
        @test !occursin("pseudolikelihood", ind_out)
        @test !occursin("suspect", ind_out)

        # Bootstrap-SE fit on a dyad-dependent model gets the softer note
        boot_fit = fit_ergm(flo, [Edges(), Triangle()]; method=:mple, se=:bootstrap, n_boot=10,
                            boot_burnin=500, boot_interval=50,
                            rng=Random.Xoshiro(8))
        boot_out = sprint(show, boot_fit)
        @test occursin("parametric-bootstrap", boot_out)
        @test !occursin("suspect", boot_out)
    end

    # ------------------------------------------------------------------
    # The shared optimizer and logistic kernel are NetworkCore.jl's: ERGM.jl re-exports them, so `using ERGM` is unchanged, and the
    # NUMERICS are pinned in NetworkCore's own "Shared Newton optimizer" testset.
    # What ERGM pins is the identity (there is exactly ONE definition), one
    # behavioural smoke test per function, and the allocation bound that the
    # variants' MPLEs rely on.
    # ------------------------------------------------------------------
    @testset "newton_fit is NetworkCore.newton_fit" begin
        @test newton_fit === NetworkCore.newton_fit
        @test ERGM.newton_fit === NetworkCore.newton_fit
        @test Base.isexported(ERGM, :newton_fit)
        @test !isdefined(ERGM, :Optim)        # the LBFGS dependency is gone

        # Poisson log-mean: ll(θ) = kθ − e^θ, maximum at log k, SE 1/√k
        k = 7.0
        pois(θ) = (k * θ[1] - exp(θ[1]), [k - exp(θ[1])], hcat(-exp(θ[1])))
        pfit = newton_fit(pois, [8.0])
        @test pfit.converged
        @test pfit.θ[1] ≈ log(k) atol = 1e-6
        @test pfit.se[1] ≈ 1 / sqrt(k) atol = 1e-6
    end

    @testset "logistic_derivatives is NetworkCore.logistic_derivatives" begin
        @test logistic_derivatives === NetworkCore.logistic_derivatives
        @test ERGM.logistic_derivatives === NetworkCore.logistic_derivatives
        @test Base.isexported(ERGM, :logistic_derivatives)

        rng = MersenneTwister(11)
        n, p = 400, 3
        X = randn(rng, n, p)
        βtrue = [0.7, -0.4, 0.2]
        y = [rand(rng) < 1 / (1 + exp(-dot(βtrue, X[r, :]))) for r in 1:n]
        d = logistic_derivatives(X, y)
        β = [0.1, 0.0, -0.1]
        ll, grad, hess = d(β)
        η = X * β
        pr = 1 ./ (1 .+ exp.(-η))
        @test ll ≈ sum(y[r] ? log(pr[r]) : log1p(-pr[r]) for r in 1:n) atol = 1e-9
        @test grad ≈ X' * (Float64.(y) .- pr) atol = 1e-9
        @test hess ≈ -X' * ((pr .* (1 .- pr)) .* X) atol = 1e-9
        fit = newton_fit(d, zeros(p))
        @test fit.converged
        @test norm(d(fit.θ)[2]) < 1e-6

        # ALLOCATION REGRESSION. The old per-package loops
        # allocated a p×p outer product and two broadcast temporaries PER ROW
        # PER EVALUATION — 470 KB on a 4200-row CMPLE design. An evaluation
        # allocates only the gradient and Hessian it returns: O(p²),
        # independent of the number of rows. The variants' MPLEs rely on it.
        function evaluation_allocs(rows)
            Xr = randn(MersenneTwister(3), rows, p)
            yr = rand(MersenneTwister(4), Bool, rows)
            f = logistic_derivatives(Xr, yr)
            f(β)                    # warm up — @allocated on a first call
            return @allocated f(β)  # would measure compilation
        end
        small = evaluation_allocs(50)
        big = evaluation_allocs(20_000)     # 400x the rows
        @test small <= 512
        @test big <= 512
        @test big <= small + 64             # ...and no growth with the rows
    end

    @testset "Bridge log-likelihood matches exhaustive enumeration (n = 6)" begin
        # Exact logZ of an edges+triangle ERGM on 6 nodes by enumerating all
        # 2^15 undirected graphs
        n = 6
        pairs = [(i, j) for i in 1:n for j in (i+1):n]
        n_dyads = length(pairs)
        θ = [-0.8, 0.4]

        function enum_stats(mask)
            adj = falses(n, n)
            for (k, (i, j)) in enumerate(pairs)
                if (mask >> (k - 1)) & 1 == 1
                    adj[i, j] = true
                    adj[j, i] = true
                end
            end
            tri = 0
            for a in 1:n, b in (a+1):n, c in (b+1):n
                adj[a, b] && adj[b, c] && adj[a, c] && (tri += 1)
            end
            return count_ones(mask), tri
        end

        vals = Vector{Float64}(undef, 2^n_dyads)
        for mask in 0:(2^n_dyads - 1)
            e, t = enum_stats(mask)
            vals[mask+1] = θ[1] * e + θ[2] * t
        end
        mx = maximum(vals)
        exact_logZ = mx + log(sum(exp.(vals .- mx)))

        net = network(n; directed=false)
        for (i, j) in [(1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (5, 6)]
            add_edge!(net, i, j)
        end
        model = ERGMModel(ERGMFormula([Edges(), Triangle()]), net)
        obs = compute_all(model.formula.terms, net)
        exact_ll = dot(θ, obs) - exact_logZ

        est = ERGM._bridge_loglik(model, θ, obs; nrungs=12, n_samples=600,
                                  burnin=3000, interval=15,
                                  rng=Random.Xoshiro(7))
        @test est ≈ exact_ll atol = 0.3

        # Dyad-independent model: the bridge collapses to the exact,
        # zero-Monte-Carlo-error log-likelihood
        m_ind = ERGMModel(ERGMFormula([Edges()]), net)
        obs_ind = compute_all(m_ind.formula.terms, net)
        θe = [-1.2]
        exact_ind = θe[1] * obs_ind[1] - n_dyads * log1p(exp(θe[1]))
        @test ERGM._bridge_loglik(m_ind, θe, obs_ind) ≈ exact_ind atol = 1e-10

        # MCMLE reports the bridge log-likelihood: for an edges-only model
        # it must be close to the exact log-likelihood at θ̂
        Random.seed!(31)
        flo = florentine_marriage()
        fit = fit_ergm(flo, [Edges()]; method=:mcmle, n_samples=400)
        exact_at = θ -> θ * 20 - 120 * log1p(exp(θ))
        @test fit.loglik ≈ exact_at(fit.coefficients[1]) atol = 1e-8
    end

    @testset "Term set" begin
        terms = [Edges(), Mutual(), Triangle()]
        ts = TermSet(terms)

        @test length(ts) == 3
        @test ts.names == ["edges", "mutual", "triangle"]

        net = network(3)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 1)

        stats = compute_all(ts, net)
        @test length(stats) == 3
        @test stats == [2.0, 1.0, 0.0]

        @test change_stat_all(ts, net, 3, 1) == [1.0, 0.0, 0.0]
    end

    @testset "Missing dyads: MPLE excludes masked dyads" begin
        rng = Random.Xoshiro(2026)
        n = 12
        net = network(n)
        for i in 1:n, j in 1:n
            i == j && continue
            rand(rng) < 0.25 && add_edge!(net, i, j)
        end
        set_vertex_attribute!(net, :gender,
                              Dict(v => (v % 2 == 0 ? "F" : "M") for v in 1:n))
        terms = [Edges(), NodeMatch(:gender)]

        full = fit_ergm(net, terms)
        @test nobs(full) == n * (n - 1)

        # Mask a mix of present and absent dyads
        masked_dyads = [(1, 2), (2, 1), (3, 7), (5, 6), (9, 10), (11, 4)]
        mnet = copy(net)
        for (i, j) in masked_dyads
            set_missing_dyad!(mnet, i, j)
        end
        fit = fit_ergm(mnet, terms)

        # nobs shrinks by exactly the number of masked dyads
        @test nobs(fit) == n * (n - 1) - length(masked_dyads)

        # The compressed design covers exactly the unmasked dyads
        ts = ERGM.TermSet(terms)
        _, n_tot, _ = ERGM._mple_data(mnet, ts, true)
        @test sum(n_tot) == n * (n - 1) - length(masked_dyads)

        # Reference fit: plain logistic regression on the manually
        # row-deleted dyad-level data (delete the masked dyads by hand)
        mset = Set(masked_dyads)
        X = Vector{Vector{Float64}}()
        y = Float64[]
        for i in 1:n, j in 1:n
            i == j && continue
            (i, j) in mset && continue
            push!(X, change_stat_all(ts, net, i, j))
            push!(y, has_edge(net, i, j) ? 1.0 : 0.0)
        end
        Xm = permutedims(reduce(hcat, X))
        β = zeros(2)
        for _ in 1:50   # Newton-Raphson IRLS
            η = Xm * β
            μ = 1 ./ (1 .+ exp.(-η))
            W = μ .* (1 .- μ)
            β += (Xm' * (W .* Xm)) \ (Xm' * (y .- μ))
        end
        @test fit.coefficients ≈ β atol = 1e-4

        # Masking changed the data, so (generically) the fit differs from
        # the full-network fit
        @test fit.coefficients != full.coefficients

        # No masked dyads: identical to the full fit
        clear_missing_dyads!(mnet)
        refit = fit_ergm(mnet, terms)
        @test refit.coefficients ≈ full.coefficients atol = 1e-10
        @test nobs(refit) == n * (n - 1)
    end

    @testset "Missing dyads: MH sampler never toggles masked dyads" begin
        n = 8
        net = network(n)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        set_missing_dyad!(net, 1, 2)   # masked, face value: edge present
        set_missing_dyad!(net, 3, 4)   # masked, face value: edge absent
        model = ERGMModel(ERGMFormula([Edges()]), net)

        # Strongly negative θ empties the free dyads, but the masked-present
        # dyad must survive; strongly positive θ fills the free dyads, but
        # the masked-absent dyad must stay empty.
        for θ in ([-4.0], [0.0], [3.0])
            out = mh_sample(model, θ; n_samples=40, burnin=500, interval=20,
                            rng=Random.Xoshiro(11), return_networks=true,
                            missing=:condition_on_face)
            @test length(out.networks) == 40
            for s in out.networks
                @test has_edge(s, 1, 2)
                @test !has_edge(s, 3, 4)
                @test is_missing_dyad(s, 1, 2)   # mask survives sampling copies
                @test is_missing_dyad(s, 3, 4)
            end
        end
        sims = sample_networks(model, [3.0]; n_sim=20, burnin=500, interval=20,
                               rng=Random.Xoshiro(5), n_chains=2,
                               missing=:condition_on_face)
        @test all(has_edge(s, 1, 2) && !has_edge(s, 3, 4) for s in sims)
        # ... and the free dyads did move under θ = 3
        @test mean(Float64(ne(s)) for s in sims) > 30

        # A fully masked network leaves the sampler nothing to toggle
        tiny = network(2)
        set_missing_dyad!(tiny, 1, 2)
        set_missing_dyad!(tiny, 2, 1)
        tiny_model = ERGMModel(ERGMFormula([Edges()]), tiny)
        @test_throws ArgumentError mh_sample(tiny_model, [0.0]; n_samples=1,
                                             burnin=10, interval=1,
                                             missing=:condition_on_face)
    end

    @testset "Missing dyads: samplers reject masked networks by default" begin
        n = 8
        net = network(n)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        set_missing_dyad!(net, 1, 2)   # masked, face value: edge present
        set_missing_dyad!(net, 3, 4)   # masked, face value: edge absent
        model = ERGMModel(ERGMFormula([Edges()]), net)

        # Every sampler entry point refuses to reinterpret the masked ties
        @test_throws ArgumentError mh_sample(model, [0.0]; n_samples=2,
                                             burnin=10, interval=1)
        @test_throws ArgumentError sample_networks(model, [0.0]; n_sim=2,
                                                   burnin=10, interval=1)
        err = try
            mh_sample(model, [0.0]; n_samples=2, burnin=10, interval=1)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("mh_sample", err.msg)        # names the routine
        @test occursin("2 masked dyads", err.msg)   # shared NetworkCore.jl message

        # The refusal names the policy ERGM ACCEPTS (`:condition_on_face`),
        # never the generic `:face` that ERGM's routines reject — for every
        # MCMC entry point.
        flo_m = florentine_marriage()
        set_missing_dyad!(flo_m, 1, 4)
        fit_m = fit_ergm(flo_m, [Edges()])
        refusals = [
            () -> mcmle(fit_m.model; n_samples=20, burnin=10, interval=1),
            () -> simulate_ergm(fit_m; n_sim=1, burnin=10, interval=1),
            () -> gof(fit_m; n_sim=1, burnin=10, interval=1),
            () -> mh_sample(fit_m.model, [0.0]; n_samples=1, burnin=10, interval=1),
            () -> sample_networks(fit_m.model, [0.0]; n_sim=1, burnin=10, interval=1),
        ]
        for (k, refuse) in enumerate(refusals)
            e = try
                refuse()
                nothing
            catch err
                err
            end
            @test e isa ArgumentError
            @test occursin("missing=:condition_on_face", e.msg)
            @test !occursin("missing=:face", e.msg)
            @test occursin("1 masked dyad", e.msg)
            # Only the estimator offers missing-data ML; the samplers never
            # advertise a policy they reject
            @test occursin("missing=:mle", e.msg) == (k == 1)
        end

        # `summary_stats` — statnet's `summary(net ~ terms)` — is a descriptive
        # statistic: it refuses a masked network by default (the shared
        # NetworkCore.jl message, which here DOES name `missing=:face` because
        # the routine takes it) and reads face values only on request
        e = try summary_stats(flo_m, [Edges(), GWESP(0.5)]); nothing catch err; err end
        @test e isa ArgumentError
        @test occursin("summary_stats", e.msg) && occursin("1 masked dyad", e.msg)
        @test occursin("missing=:face", e.msg)
        @test summary_stats(flo_m, [Edges(), GWESP(0.5)]; missing=:face) ==
              summary_stats(florentine_marriage(), [Edges(), GWESP(0.5)])
        @test_throws ArgumentError summary_stats(flo_m, [Edges()]; missing=:condition_on_face)
        @test missing_policies(summary_stats) == (:error, :face)
        # ... while the protocol-level raw evaluations carry no policy
        @test compute(Edges(), flo_m) == 20.0

        # ...and the vocabulary is declared, so tooling need not guess
        @test missing_policies(mcmle) == (:error, :condition_on_face, :mle)
        @test missing_policies(mh_sample) == (:error, :condition_on_face)
        @test missing_policies(sample_networks) == (:error, :condition_on_face)
        @test missing_policies(simulate_ergm) == (:error, :condition_on_face)
        @test NetworkCore.missing_policies(gof, ERGMResult) == (:error, :condition_on_face)
        @test missing_policies(mple) == (:error,)    # no keyword: it handles the mask
        @test supports_missing(mple)

        # An unknown policy is rejected outright (not silently ignored)
        @test_throws ArgumentError mh_sample(model, [0.0]; n_samples=2,
                                             burnin=10, interval=1,
                                             missing=:face)
        @test_throws ArgumentError mh_sample(model, [0.0]; n_samples=2,
                                             burnin=10, interval=1,
                                             missing=:nonsense)

        # Unmasked networks are unaffected by the guard
        clean = network(n)
        add_edge!(clean, 1, 2)
        clean_model = ERGMModel(ERGMFormula([Edges()]), clean)
        out = mh_sample(clean_model, [0.0]; n_samples=3, burnin=10, interval=1,
                        rng=Random.Xoshiro(3))
        @test size(out.stats) == (3, 1)
    end

    @testset "Missing dyads: MCMLE rejects by default, opts in explicitly" begin
        Random.seed!(77)
        flo = florentine_marriage()
        # One masked dyad whose face value is a tie, one whose face value is
        # a non-tie: both are unobserved, and MCMLE would score both as
        # stored.
        present = first((i, j) for i in 1:16, j in 1:16 if i < j && has_edge(flo, i, j))
        absent = first((i, j) for i in 1:16, j in 1:16 if i < j && !has_edge(flo, i, j))
        set_missing_dyad!(flo, present...)
        set_missing_dyad!(flo, absent...)
        @test n_missing_dyads(flo) == 2
        model = ERGMModel(ERGMFormula([Edges()]), flo)

        # Default: rejected, with the shared missing-data error message
        err = try
            mcmle(model; n_samples=50, burnin=100, interval=5, maxiter=2,
                  bridge_rungs=2, bridge_samples=50)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("mcmle", err.msg)
        @test occursin("2 masked dyads", err.msg)
        @test occursin("unobserved", err.msg)

        # ...and so is a bogus policy, or Network's generic :face spelling
        # (the ERGM opt-in is the specific :condition_on_face)
        @test_throws ArgumentError mcmle(model; missing=:face, n_samples=50,
                                         burnin=100, interval=5, maxiter=2,
                                         bridge_rungs=2, bridge_samples=50)

        # Explicit opt-in: fits, warns honestly, and records the method
        fit = @test_logs (:warn, r"conditions on them at their face value") match_mode=:any mcmle(
            model; missing=:condition_on_face, n_samples=200, burnin=1000,
            interval=20, maxiter=5, bridge_rungs=2, bridge_samples=100)
        @test length(fit.coefficients) == 1
        @test fit.missing_method === :condition_on_face
        @test nobs(fit) == 120 - 2
        # The treatment is visible in the printed output
        s = sprint(show, fit)
        @test occursin("Missing dyads: 2 masked", s)
        @test occursin("conditioned on face values", s)

        # An unmasked fit records :none and prints nothing about missingness
        clean_fit = mcmle(ERGMModel(ERGMFormula([Edges()]), florentine_marriage());
                          n_samples=200, burnin=1000, interval=20, maxiter=5,
                          bridge_rungs=2, bridge_samples=100)
        @test clean_fit.missing_method === :none
        @test !occursin("Missing dyads", sprint(show, clean_fit))
    end

    @testset "Missing dyads: MPLE declares support, records available-case" begin
        # MPLE's available-case pseudo-likelihood IS a principled treatment,
        # so it declares the ecosystem trait and takes no `missing` keyword.
        @test supports_missing(mple)
        @test supports_missing(mcmle)         # missing=:mle (the default still refuses)
        @test !supports_missing(simulate_ergm)
        @test !supports_missing(gof)

        flo = florentine_marriage()
        present = first((i, j) for i in 1:16, j in 1:16 if i < j && has_edge(flo, i, j))
        absent = first((i, j) for i in 1:16, j in 1:16 if i < j && !has_edge(flo, i, j))
        set_missing_dyad!(flo, present...)
        set_missing_dyad!(flo, absent...)

        fit = fit_ergm(flo, [Edges()])   # method=:mple
        @test fit.method === :mple
        @test fit.missing_method === :available_case
        @test nobs(fit) == 120 - 2
        s = sprint(show, fit)
        @test occursin("Missing dyads: 2 masked", s)
        @test occursin("available-case", s)

        # Unmasked: :none
        clean = fit_ergm(florentine_marriage(), [Edges()])
        @test clean.missing_method === :none
        @test !occursin("Missing dyads", sprint(show, clean))
    end

    @testset "Missing dyads: simulation and GOF cannot reinterpret masked ties" begin
        Random.seed!(99)
        flo = florentine_marriage()
        present = first((i, j) for i in 1:16, j in 1:16 if i < j && has_edge(flo, i, j))
        absent = first((i, j) for i in 1:16, j in 1:16 if i < j && !has_edge(flo, i, j))
        set_missing_dyad!(flo, present...)
        set_missing_dyad!(flo, absent...)

        # An MPLE fit is legitimate on a masked network, but SIMULATING from
        # it is a separate act that would freeze the unobserved ties at their
        # face value — so it must be asked for.
        fit = fit_ergm(flo, [Edges()])
        @test_throws ArgumentError simulate_ergm(fit; n_sim=2, burnin=100,
                                                 interval=10)
        @test_throws ArgumentError gof(fit; n_sim=2, stats=[:degree],
                                       burnin=100, interval=10)

        sims = @test_logs (:warn, r"conditions on them at their face value") match_mode=:any simulate_ergm(
            fit; n_sim=5, burnin=200, interval=10, rng=Random.Xoshiro(4),
            n_chains=2, missing=:condition_on_face)
        @test length(sims) == 5
        # The face values are frozen in every simulated network
        @test all(has_edge(s, present...) for s in sims)
        @test all(!has_edge(s, absent...) for s in sims)

        g = @test_logs (:warn, r"conditions on them at their face value") match_mode=:any gof(
            fit; n_sim=5, stats=[:degree], burnin=200, interval=10,
            rng=Random.Xoshiro(6), n_chains=2, missing=:condition_on_face)
        @test length(g.statistics) == 1

        # Unmasked fits keep working with the default policy
        clean_fit = fit_ergm(florentine_marriage(), [Edges()])
        @test length(simulate_ergm(clean_fit; n_sim=2, burnin=100, interval=10,
                                   rng=Random.Xoshiro(7))) == 2
    end

    @testset "Result metadata protocol" begin
        flo = florentine_marriage()

        # THE assertion this protocol exists for: ONE estimator (MPLE), two
        # formulas — exact ML on the dyad-independent one, an approximation on
        # the dyad-dependent one. `is_exact` is a property of the FIT.
        indep = fit_ergm(flo, [Edges(), NodeCov(:wealth)])
        dep = fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mple)

        md_indep = fit_metadata(indep)
        @test md_indep.estimand == :ergm
        @test md_indep.objective == :pseudolikelihood
        @test md_indep.is_exact          # MPLE of a dyad-independent formula IS the MLE
        @test md_indep.se_method == :hessian
        @test md_indep.missing_method == :none
        @test md_indep.tie_method == :not_applicable
        @test isempty(md_indep.approximations)

        md_dep = fit_metadata(dep)
        @test md_dep.objective == :pseudolikelihood     # same estimator
        @test !md_dep.is_exact                          # different formula
        @test md_dep.se_method == :hessian
        @test any(occursin("anticonservative", a) for a in md_dep.approximations)

        # The prose caveat in `show` and the protocol are driven by the same
        # predicate, so they agree on every fit.
        for (fit, exact) in ((indep, true), (dep, false))
            printed = sprint(show, fit)
            @test occursin("pseudolikelihood", printed) == !exact
            @test is_exact(fit) == exact
        end

        # Accessors are callable directly, not only through the collector
        @test estimand(indep) == :ergm
        @test objective(indep) == :pseudolikelihood
        @test missing_method(indep) == :none
        @test approximations(indep) == String[]

        # MCMLE: a Monte-Carlo approximation to the likelihood, never exact,
        # with inverse-Fisher standard errors from the MCMC sample
        mc = mcmle(ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo);
                   n_samples=200, burnin=200, interval=5, maxiter=3,
                   rng=Random.Xoshiro(11))
        md_mc = fit_metadata(mc)
        @test md_mc.objective == :mc_likelihood
        @test !md_mc.is_exact
        @test md_mc.se_method == :fisher
        @test any(occursin("Monte-Carlo error", a) for a in md_mc.approximations)

        # Bootstrap standard errors are reported as such
        boot = fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mple, se=:bootstrap, n_boot=5,
                        rng=Random.Xoshiro(12))
        @test se_method(boot) == :bootstrap

        # Masked dyads: MPLE drops them (available case), and the protocol says so
        masked = florentine_marriage()
        set_missing_dyad!(masked, 1, 4)
        mfit = fit_ergm(masked, [Edges()])
        @test missing_method(mfit) == :available_case
        @test fit_metadata(mfit).missing_method == :available_case
    end

    # ------------------------------------------------------------------
    # Golden fixture: a REAL statnet `ergm` fit, with provenance (issue #8).
    #
    # The suite used to carry R's Florentine numbers as bare literals in
    # comments ("Golden master vs R ergm", "Golden master: MCMLE matches R
    # statnet"). They were right, but they could not be regenerated, nobody
    # could tell which ergm produced them, and the atols beside them were
    # chosen by hand. Every one of those numbers now comes from the
    # provenanced TOML fixture (test/fixtures/flomarriage_ergm.toml), generated
    # by test/fixtures/r/flomarriage_ergm.R against a stated ergm version and
    # seed, with every tolerance justified in the fixture itself; the literal
    # testsets are gone.
    #
    # It deliberately covers BOTH kinds of ERGM fit, because they are different
    # kinds of number and a single tolerance for both would be dishonest:
    #   - dyad-independent: MPLE IS the exact MLE. Compared at 1e-6.
    #   - dyad-dependent:   MCMLE. Compared against R's OWN measured seed-to-seed
    #                       spread, which the fixture records.
    # ------------------------------------------------------------------
    @testset "Golden fixture: statnet ergm on flomarriage (provenanced)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "flomarriage_ergm.toml"))
        @test g.provenance["ergm_version"] == "4.12.0"
        flo = florentine_marriage()

        # --- deterministic: summary statistics ---------------------------
        # A function of the observed graph alone. Any disagreement is a bug in
        # a term formula; there is no Monte Carlo to hide behind.
        @test nv(flo) == 16
        @test ne(flo) == 20
        @test g.values["summary_statistic_names"] ==
              ["edges", "nodecov.wealth", "gwesp.fixed.0.5", "triangle"]
        stats = [compute(Edges(), flo), compute(NodeCov(:wealth), flo),
                 compute(GWESP(0.5), flo), compute(Triangle(), flo)]
        @test check_golden(g, "summary_statistics", stats) ||
              error(golden_report(g, "summary_statistics", stats))

        # --- (a) dyad-independent: MPLE == exact ML ----------------------
        # edges + nodecov("wealth") factorizes over dyads, so both packages are
        # solving the SAME convex logistic regression. Agreement is asserted at
        # 1e-6 — optimizer precision, not "close enough". Observed: 6.6e-12 on
        # the coefficients, 4.8e-8 on the standard errors.
        @test g.values["di_terms"] == ["edges", "nodecov.wealth"]
        di = fit_ergm(flo, [Edges(), NodeCov(:wealth)]; method=:mple)
        @test check_golden(g, "di_coefficients", di.coefficients) ||
              error(golden_report(g, "di_coefficients", di.coefficients))
        @test check_golden(g, "di_std_errors", di.std_errors) ||
              error(golden_report(g, "di_std_errors", di.std_errors))
        # The exact log-likelihood and AIC follow, and are exact for the same
        # reason (no bridge sampler is involved in a dyad-independent fit).
        @test di.loglik ≈ g.values["di_loglik"] atol = 1e-6
        @test di.aic ≈ g.values["di_aic"] atol = 1e-6
        # vcov is the full covariance matrix, consistent with the SEs
        V = vcov(di)
        @test size(V) == (2, 2)
        @test sqrt.([V[1, 1], V[2, 2]]) ≈ di.std_errors atol = 1e-8
        @test V[1, 2] ≈ V[2, 1] atol = 1e-12

        # --- gof() observed panels: R's obs.deg / obs.esp / obs.dist ------
        # Deterministic functions of the graph. The distance panel counts
        # UNORDERED pairs on an undirected network and ends with the
        # unreachable pairs under "Inf" (it used to count ordered pairs,
        # twice R, and dropped the Inf row). ERGM.jl trims trailing all-zero
        # finite levels, so the panels are compared zero-padded to R's length.
        gd = gof(di; n_sim=2, burnin=10, interval=1, stats=[:degree, :esp, :distance],
                 rng=Random.Xoshiro(3))
        padded(v, len) = (@assert length(v) <= len; vcat(v, zeros(len - length(v))))
        panel(name) = only(s for s in gd.statistics if s.name == name)
        @test check_golden(g, "gof_degree", padded(panel("degree").observed, 16)) ||
              error(golden_report(g, "gof_degree", panel("degree").observed))
        @test check_golden(g, "gof_esp", padded(panel("esp").observed, 15)) ||
              error(golden_report(g, "gof_esp", panel("esp").observed))
        dist = panel("distance")
        @test dist.labels[end] == "Inf" && g.values["gof_distance_labels"][end] == "Inf"
        dist_full = vcat(padded(dist.observed[1:end-1], 15), dist.observed[end])
        @test check_golden(g, "gof_distance", dist_full) ||
              error(golden_report(g, "gof_distance", dist_full))
        @test dist.observed[end] == 15.0            # Pucci: 15 unreachable pairs
        @test sum(dist.observed) == 120             # every unordered dyad, once
        @test size(dist.simulated, 2) == length(dist.labels)
        @test all(sum(dist.simulated; dims=2) .== 120)

        # --- (a') the Bernoulli model: MLE = logit(density), exact ------
        eo = fit_ergm(flo, [Edges()])
        @test eo.coefficients[1] ≈ log((20 / 120) / (100 / 120)) atol = 1e-8
        @test check_golden(g, "eo_coefficients", eo.coefficients) ||
              error(golden_report(g, "eo_coefficients", eo.coefficients))
        @test check_golden(g, "eo_std_errors", eo.std_errors) ||
              error(golden_report(g, "eo_std_errors", eo.std_errors))
        # MCMLE of the same dyad-independent model must land on the same
        # exact MLE: it starts at the MPLE (= the MLE) and its convergence
        # test passes there, so the point estimate is that MLE up to the
        # Monte-Carlo test's verdict (0.05 is ~1/5 of the standard error)
        eo_mc = fit_ergm(flo, [Edges()]; method=:mcmle, n_samples=600,
                         rng=Random.Xoshiro(42))
        @test eo_mc.method == :mcmle
        @test eo_mc.converged
        @test eo_mc.coefficients[1] ≈ Float64(g.values["eo_coefficients"][1]) atol = 0.05
        @test !isnothing(eo_mc.mcmc_samples)

        # --- (b) dyad-dependent: MCMLE -----------------------------------
        # edges + gwesp(0.5, fixed=TRUE). Both sides are Monte Carlo, so we
        # compare the MEAN of five ERGM.jl fits at declared seeds against the
        # frozen R fit, at a tolerance the fixture justifies from R's own
        # seed-to-seed spread (`mcmle_seed_sd`).
        @test g.values["dd_terms"] == ["edges", "gwesp.fixed.0.5"]
        dd_fits = [fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mcmle,
                            n_samples=4096, rng=Random.Xoshiro(s))
                   for s in (101, 202, 303, 404, 505)]
        @test all(f.converged for f in dd_fits)
        dd_coef = mean(f.coefficients for f in dd_fits)
        dd_se = mean(f.std_errors for f in dd_fits)
        @test check_golden(g, "dd_coefficients", dd_coef) ||
              error(golden_report(g, "dd_coefficients", dd_coef))
        @test check_golden(g, "dd_std_errors", dd_se) ||
              error(golden_report(g, "dd_std_errors", dd_se))

        # The agreement above is closer than R's agreement with ITSELF: the
        # gap to R is smaller than R's own seed-to-seed sd on both coefficients.
        r_sd = Float64.(g.values["mcmle_seed_sd"])
        gap = abs.(dd_coef .- Float64.(g.values["dd_coefficients"]))
        @test all(gap .< 3 .* r_sd)

        # statnet's order of operations (the pre-0.2 N7 difference is gone):
        # MCMLE takes a Monte-Carlo Newton step at EVERY iteration, the first
        # included, so no fit returns the MPLE unchanged, and the point
        # estimates vary from seed to seed like R's own.
        mple_dd = fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mple)
        @test !any(f.coefficients == mple_dd.coefficients for f in dd_fits)
        @test std(f.coefficients[2] for f in dd_fits) > 1e-6
        # ...and the standard errors come from the MCMC sample, and vary.
        @test std(f.std_errors[2] for f in dd_fits) > 1e-6
    end

    # ------------------------------------------------------------------
    # Term hygiene and R parity for migrants
    # ------------------------------------------------------------------
    @testset "GW terms at decay = 0" begin
        flo = florentine_marriage()

        # GWESP(0): eᵅ(1 − (1 − e⁻ᵅ)ˢ) at α = 0 is 1 for s ≥ 1 and 0 for s = 0
        # (0^0 == 1), i.e. the number of edges with ≥ 1 shared partner
        n_esp = count(!isempty(intersect(neighbors(flo, i), neighbors(flo, j)))
                      for i in 1:16 for j in neighbors(flo, i) if i < j)
        @test compute(GWESP(0.0), flo) == n_esp == 8.0

        # GWDegree(0): the number of non-isolates
        @test compute(GWDegree(0.0), flo) == count(v -> degree(flo, v) > 0, 1:16) == 15.0

        # GWDSP(0): the number of dyads (tied or not) with ≥ 1 shared partner
        n_dsp = count(!isempty(intersect(neighbors(flo, i), neighbors(flo, j)))
                      for i in 1:16 for j in (i+1):16)
        @test compute(GWDSP(0.0), flo) == n_dsp == 43.0

        # Directed: decay-0 in-/out-degree terms count the vertices with
        # in-/out-degree ≥ 1
        sl = samplike()
        @test compute(GWIDegree(0.0), sl) == count(v -> indegree(sl, v) > 0, 1:18)
        @test compute(GWODegree(0.0), sl) == count(v -> outdegree(sl, v) > 0, 1:18)

        # Every GW term takes 0.0 (also as an Int) and refuses a negative decay
        for T in (GWESP, GWDSP, GWDegree, GWIDegree, GWODegree)
            @test T(0.0).decay == 0.0
            @test T(0).decay == 0.0
            e = try; T(-0.1); nothing; catch err; err; end
            @test e isa ArgumentError && occursin("non-negative", e.msg)
            @test_throws ArgumentError T(NaN)
        end

        # A decay-0 model fits and simulates like any other
        f = fit_ergm(flo, [Edges(), GWESP(0.0)]; method=:mple)
        @test coef(f) |> length == 2 && all(isfinite, coef(f))
        @test f.model.formula.terms.names == ["edges", "gwesp.fixed.0"]
    end

    @testset "R-style decay labels" begin
        # R prints an integer-valued decay without a decimal point:
        # gwesp(0, fixed=TRUE) is "gwesp.fixed.0", gwesp(1, fixed=TRUE)
        # "gwesp.fixed.1" — `string(1.0)` would give "gwesp.fixed.1.0" and
        # break every by-name comparison with a statnet fit.
        @test name(GWESP(0.0)) == "gwesp.fixed.0"
        @test name(GWESP(0.5)) == "gwesp.fixed.0.5"
        @test name(GWESP(1.0)) == "gwesp.fixed.1"
        @test name(GWESP(1.25)) == "gwesp.fixed.1.25"
        @test name(GWESP(0.0; type=:OSP)) == "gwesp.OSP.fixed.0"
        @test name(GWESP(2.0; type=:union)) == "gwesp.union.fixed.2"
        @test name(GWDSP(0.0)) == "gwdsp.fixed.0"
        @test name(GWDSP(0.5; type=:ITP)) == "gwdsp.ITP.fixed.0.5"
        # gwdegree is labelled "gwdeg" in R, like gwideg/gwodeg
        @test name(GWDegree(0.0)) == "gwdeg.fixed.0"
        @test name(GWDegree(1.0)) == "gwdeg.fixed.1"
        @test name(GWDegree(0.5)) == "gwdeg.fixed.0.5"
        @test name(GWIDegree(1.0)) == "gwideg.fixed.1"
        @test name(GWODegree(0.0)) == "gwodeg.fixed.0"
        @test ERGM._decay_label(0.0) == "0"
        @test ERGM._decay_label(3.0) == "3"
        @test ERGM._decay_label(0.75) == "0.75"
        # absdiff(pow=2) is labelled absdiff2.<attr>, and R's integer spelling
        # `absdiff("wealth", pow=2)` is accepted (it used to take `pow::Float64`
        # threw a raw TypeError on `pow=2`)
        @test name(AbsDiff(:wealth; pow=2)) == "absdiff2.wealth"
        @test name(AbsDiff(:wealth; pow=2.0)) == "absdiff2.wealth"
        @test name(AbsDiff(:wealth; pow=1)) == "absdiff.wealth"
        @test AbsDiff(:wealth; pow=2).pow === 2.0
        flo = florentine_marriage()
        @test compute(AbsDiff(:wealth; pow=2), flo) == compute(AbsDiff(:wealth; pow=2.0), flo)
    end

    @testset "OStar / IStar (statnet ostar/istar) and Kstar on undirected" begin
        # Arcs 1→2, 1→3, 2→3: out-degrees [2,1,0], in-degrees [0,1,2]
        net = network(3; directed=true)
        for (i, j) in [(1, 2), (1, 3), (2, 3)]
            add_edge!(net, i, j)
        end
        @test compute(OStar(2), net) == 1.0
        @test compute(IStar(2), net) == 1.0
        @test compute(OStar(3), net) == 0.0
        @test name(OStar(2)) == "ostar2"
        @test name(IStar(3)) == "istar3"
        @test_throws ArgumentError OStar(1)
        @test_throws ArgumentError IStar(0)
        @test is_dyad_dependent(OStar(2)) && is_dyad_dependent(IStar(2))

        # Σᵥ C(outdeg(v), k) / Σᵥ C(indeg(v), k) on Sampson's monastery
        sl = samplike()
        @test compute(OStar(2), sl) == sum(binomial(outdegree(sl, v), 2) for v in 1:18) == 178.0
        @test compute(IStar(2), sl) == sum(binomial(indegree(sl, v), 2) for v in 1:18) == 233.0
        @test compute(OStar(3), sl) == sum(binomial(outdegree(sl, v), 3) for v in 1:18)

        # O(1) change statistics from the endpoint degree, and allocation-free
        @test change_stat(OStar(2), sl, 1, 2) == binomial(outdegree(sl, 1), 2) - binomial(outdegree(sl, 1) - 1, 2)
        @test change_stat(IStar(2), sl, 5, 6) == binomial(indegree(sl, 6) + 1, 2) - binomial(indegree(sl, 6), 2)
        for t in (OStar(2), IStar(2))
            change_stat(t, sl, 3, 4)
            @test @allocated(change_stat(t, sl, 3, 4)) == 0
        end

        # Kstar on undirected is Σᵥ C(deg(v), k)
        flo = florentine_marriage()
        @test compute(Kstar(2), flo) == sum(binomial(degree(flo, v), 2) for v in 1:16) == 47.0
        @test compute(Kstar(3), flo) == sum(binomial(degree(flo, v), 3) for v in 1:16)

        # ... and a directed model with the star/degree terms fits
        f = fit_ergm(sl, [Edges(), Mutual(), OStar(2), IStar(2)]; method=:mple)
        @test f.model.formula.terms.names == ["edges", "mutual", "ostar2", "istar2"]
        @test all(isfinite, coef(f))
    end

    @testset "Degree(0:2) is one expanding term" begin
        flo = florentine_marriage()

        # [Edges(), Degree(0:2)] is a Vector{<:AbstractERGMTerm} and fits with
        # one coefficient per degree, equal to three explicit Degree(d)
        f = fit_ergm(flo, [Edges(), Degree(0:2)]; method=:mple)
        @test f.model.formula.terms.names == ["edges", "degree0", "degree1", "degree2"]
        f3 = fit_ergm(flo, [Edges(), Degree(0), Degree(1), Degree(2)]; method=:mple)
        @test coef(f) == coef(f3)
        @test stderror(f) == stderror(f3)
        @test coef(fit_ergm(flo, [Edges(); Degree(0:2)]; method=:mple)) == coef(f)
        @test summary_stats(flo, [Edges(), Degree(0:2)]) ==
              (edges=20.0, degree0=1.0, degree1=4.0, degree2=2.0)

        # Same for IDegree/ODegree on a directed network
        sl = samplike()
        fi = fit_ergm(sl, [Edges(), IDegree(2:4), ODegree([3, 4])]; method=:mple)
        @test fi.model.formula.terms.names ==
              ["edges", "idegree2", "idegree3", "idegree4", "odegree3", "odegree4"]
        @test coef(fi) == coef(fit_ergm(sl, [Edges(), IDegree(2), IDegree(3), IDegree(4),
                                             ODegree(3), ODegree(4)]; method=:mple))

        # A single degree is a statistic; a multi-degree term is a
        # specification that must be expanded
        @test Degree(2) == Degree([2]) && name(Degree(2)) == "degree2"
        @test compute(Degree(2), flo) == 2.0
        @test name(Degree(0:2)) == "degree(0,1,2)"
        for (t, net) in ((Degree(0:2), flo), (IDegree(1:2), sl), (ODegree([0, 3]), sl))
            e = try; compute(t, net); nothing; catch err; err; end
            @test e isa ArgumentError
            @test occursin("not a single statistic", e.msg)
            @test occursin("ERGMModel", e.msg)
            @test_throws ArgumentError change_stat(t, net, 1, 2)
        end
        # ... and the expansion is what materialize does
        @test ERGM.Extension.materialize(Degree(0:2), flo) == [Degree(0), Degree(1), Degree(2)]
        @test ERGM.Extension.materialize(Degree(1), flo) === Degree(1) || ERGM.Extension.materialize(Degree(1), flo) == Degree(1)
    end

    @testset "Golden fixture: statnet ergm term parity (provenanced)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "ergm_terms.toml"))
        @test g.provenance["ergm_version"] == "4.12.0"
        flo = florentine_marriage()
        sl = samplike()

        # --- flomarriage: decay-0 GW terms, R's labels, degree(0:2) --------
        # Names are compared EXACTLY: a coefficient that R calls
        # "gwesp.fixed.0" must be called that here, or by-name comparison of
        # two fits is impossible.
        flo_terms = [Edges(), GWESP(0.0), GWDSP(0.0), GWDegree(0.0), GWDegree(1.0),
                     Degree(0:2), AbsDiff(:wealth; pow=2.0), AbsDiff(:wealth)]
        stats = summary_stats(flo, flo_terms)
        @test String.(collect(keys(stats))) == g.values["flo_summary_names"] ==
              ["edges", "gwesp.fixed.0", "gwdsp.fixed.0", "gwdeg.fixed.0",
               "gwdeg.fixed.1", "degree0", "degree1", "degree2",
               "absdiff2.wealth", "absdiff.wealth"]
        @test check_golden(g, "flo_summary", collect(values(stats))) ||
              error(golden_report(g, "flo_summary", collect(values(stats))))
        # ... and the same numbers through a model (materialized formula)
        m = ERGMModel(ERGMFormula(flo_terms), flo)
        @test m.formula.terms.names == g.values["flo_summary_names"]
        @test check_golden(g, "flo_summary", compute_all(m.formula.terms, flo))

        # --- MPLE vs MPLE on edges + degree(0:2) (dyad-dependent, so this
        # pins the pseudo-likelihood and the Degree(0:2) expansion, not the
        # MLE): both sides maximize the same logistic-regression objective,
        # compared at 1e-6. Observed: ~1e-13 on coefficients.
        @test g.values["flo_mple_terms"] == ["edges", "degree0", "degree1", "degree2"]
        fit = fit_ergm(flo, [Edges(), Degree(0:2)]; method=:mple)
        @test fit.model.formula.terms.names == g.values["flo_mple_terms"]
        @test check_golden(g, "flo_mple_coefficients", fit.coefficients) ||
              error(golden_report(g, "flo_mple_coefficients", fit.coefficients))
        @test check_golden(g, "flo_mple_std_errors", fit.std_errors) ||
              error(golden_report(g, "flo_mple_std_errors", fit.std_errors))

        # --- samplike (directed): ostar/istar, gwidegree/gwodegree, triangle,
        # then (positions 9-16) every directed shared-partner type: gwesp/gwdsp
        # with their default type — R's DIRECTED labels carry it,
        # `gwesp.OTP.fixed.0.5` — and the typed dgwesp/dgwdsp ITP/OSP/ISP rows,
        # all summed over ORDERED dyads in R's C code (gwdsp OSP/ISP
        # used to come out at exactly half of R's). `ttriple` (row 8) has no
        # ERGM.jl term and is not compared; Triangle() on a directed network
        # is statnet's `triangle`.
        sl_terms = [Edges(), Mutual(), OStar(2), IStar(2), GWIDegree(0.5),
                    GWODegree(0.5), Triangle()]
        sl_sp = [GWESP(0.5), GWDSP(0.5), GWESP(0.5; type=:ITP), GWESP(0.5; type=:OSP),
                 GWESP(0.5; type=:ISP), GWDSP(0.5; type=:ITP), GWDSP(0.5; type=:OSP),
                 GWDSP(0.5; type=:ISP)]
        @test g.values["samplike_summary_names"][1:7] == [name(t) for t in sl_terms]
        @test g.values["samplike_summary_names"][8] == "ttriple"
        @test g.values["samplike_summary_names"][9:16] ==
              ["gwesp.OTP.fixed.0.5", "gwdsp.OTP.fixed.0.5", "gwesp.ITP.fixed.0.5",
               "gwesp.OSP.fixed.0.5", "gwesp.ISP.fixed.0.5", "gwdsp.ITP.fixed.0.5",
               "gwdsp.OSP.fixed.0.5", "gwdsp.ISP.fixed.0.5"] ==
              [name(t, sl) for t in sl_sp]
        sl_stats = vcat(compute_all(TermSet(sl_terms), sl), g.values["samplike_summary"][8],
                        compute_all(TermSet(sl_sp), sl))
        @test check_golden(g, "samplike_summary", sl_stats) ||
              error(golden_report(g, "samplike_summary", sl_stats))
        # ttriple ≤ triangle, since triangle = ttriple + ctriple
        @test g.values["samplike_summary"][8] <= compute(Triangle(), sl)
        # ... the same numbers AND labels through a model and summary_stats
        sm = ERGMModel(ERGMFormula(sl_sp), sl)
        @test sm.formula.terms.names == g.values["samplike_summary_names"][9:16]
        @test check_golden(g, "samplike_summary",
                           vcat(sl_stats[1:8], compute_all(sm.formula.terms, sl)))
        @test String.(collect(keys(summary_stats(sl, sl_sp)))) ==
              g.values["samplike_summary_names"][9:16]
        # OSP/ISP change statistics agree with brute force on samplike (the
        # halved statistic came with a consistently halved change statistic,
        # invisible to the randomized cross-validation)
        for t in (:OSP, :ISP), (i, j) in ((1, 5), (7, 3), (12, 16), (4, 11))
            term = GWDSP(0.5; type=t)
            plus = copy(sl); has_edge(plus, i, j) || add_edge!(plus, i, j)
            minus = copy(sl); has_edge(minus, i, j) && rem_edge!(minus, i, j)
            @test change_stat(term, sl, i, j) ≈ compute(term, plus) - compute(term, minus) atol = 1e-10
        end

        # --- gof() observed panels on the directed samplike: ORDERED pairs
        # for the distance panel (as R), Inf row = 0 (strongly connected),
        # idegree/odegree over 0:17, esp (OTP) over 0:16 -------------------
        gs = gof(fit_ergm(sl, [Edges()]); n_sim=2, burnin=10, interval=1,
                 stats=[:degree, :esp, :distance], rng=Random.Xoshiro(4))
        spanel(name) = only(s for s in gs.statistics if s.name == name)
        spad(v, len) = vcat(v, zeros(len - length(v)))
        @test [s.name for s in gs.statistics] == ["idegree", "odegree", "esp", "distance"]
        @test check_golden(g, "samplike_gof_idegree", spad(spanel("idegree").observed, 18)) ||
              error(golden_report(g, "samplike_gof_idegree", spanel("idegree").observed))
        @test check_golden(g, "samplike_gof_odegree", spad(spanel("odegree").observed, 18)) ||
              error(golden_report(g, "samplike_gof_odegree", spanel("odegree").observed))
        @test g.values["samplike_gof_esp_labels"][1] == ".OTP0"
        @test check_golden(g, "samplike_gof_esp", spad(spanel("esp").observed, 17)) ||
              error(golden_report(g, "samplike_gof_esp", spanel("esp").observed))
        sd = spanel("distance")
        sd_full = vcat(spad(sd.observed[1:end-1], 17), sd.observed[end])
        @test check_golden(g, "samplike_gof_distance", sd_full) ||
              error(golden_report(g, "samplike_gof_distance", sd_full))
        @test sum(sd.observed) == 18 * 17 && sd.observed[end] == 0

        # --- R refuses these too, and the fixture says how ------------------
        @test occursin("directed==TRUE", g.values["r_error_kstar_directed"])
        @test occursin("directed==TRUE", g.values["r_error_gwdegree_directed"])
        @test occursin("missing data", g.values["r_error_nodecov_na"])
        @test_throws ArgumentError ERGMModel(ERGMFormula([Edges(), Kstar(2)]), sl)
        @test_throws ArgumentError ERGMModel(ERGMFormula([Edges(), GWDegree(0.5)]), sl)
        na = copy(flo)
        x = Dict(v => Float64(get_vertex_attribute(flo, :wealth, v)) for v in 1:16)
        delete!(x, 3)
        set_vertex_attribute!(na, :x, x)
        @test_throws ArgumentError ERGMModel(ERGMFormula([Edges(), NodeCov(:x)]), na)
    end

    # ------------------------------------------------------------------
    # Seams and contracts
    # ------------------------------------------------------------------
    @testset "Re-exports: Network is usable after `using ERGM`" begin
        # (a) Fresh process: `using ERGM` alone gives the network constructor
        # and the descriptive verbs, and does NOT leak the developer tooling.
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(@__DIR__)) -e 'using ERGM; @assert Network(5) isa Network; @assert degree(Network(3)) == zeros(Int, 3); @assert !isdefined(Main, :load_golden); @assert !isdefined(Main, :bootstrap_cov); @assert !isdefined(Main, :NetworkCore)'`
        @test success(cmd)

        # (b) Drift pin: the curated list mirrors NetworkCore's frozen export
        # inventory exactly, minus the deliberate exclusions. A new NetworkCore
        # export not added here (or an exclusion silently re-exported) fails.
        exported_networks = filter(n -> Base.isexported(NetworkCore, n), names(NetworkCore))
        @test Set(setdiff(exported_networks, names(ERGM))) ==
              Set([:GoldenFixture, :load_golden, :check_golden, :golden_report,
                   :golden_tolerance, :bootstrap_cov, :record_drop!, :check_se,
                   :check_statsapi, :NetworkCore])
        @test Base.isexported(ERGM, :Network)
        @test Base.isexported(ERGM, :degree)
        @test Base.isexported(ERGM, :src) && Base.isexported(ERGM, :dst)   # with `edges`
        @test ERGM.src === Graphs.src && ERGM.dst === Graphs.dst
        @test Base.isexported(ERGM, :coeftable)
        @test ERGM.coeftable === NetworkCore.coeftable === StatsBase.coeftable
        @test Base.isexported(ERGM, :coefnames)
        @test ERGM.coefnames === NetworkCore.coefnames === StatsBase.coefnames
        @test ERGM.z_pvalues === NetworkCore.z_pvalues
        @test !Base.isexported(ERGM, :check_se)
    end

    @testset "Extension API surface" begin
        # `ERGM.Extension` is the stable API for packages built on ERGM.jl.
        # Its inventory is pinned as a literal, so adding or removing a name
        # is a decision, not an accident.
        EXT = sort([:attainable_range, :boundary_columns, :bridge_integrate,
                    :collect_terms, :confidence_test, :ess_sample, :expand_terms,
                    :extreme_statistics, :materialize, :mcmc_defaults,
                    :mcmle_covariance, :mcmle_sampler, :mcmle_solve, :mple_fit_design,
                    :n_observed_dyads, :require_supported_network,
                    :validate_formula, :warn_boundary])
        ext_names = sort([n for n in names(ERGM.Extension) if n !== :Extension])
        @test ext_names == EXT
        @test all(n -> Base.isexported(ERGM.Extension, n), EXT)
        @test all(n -> !startswith(string(n), "_"), EXT)
        # `ERGM` exports nothing new and keeps the submodule a public name
        @test Base.ispublic(ERGM, :Extension) && !Base.isexported(ERGM, :Extension)
        @test all(n -> !Base.isexported(ERGM, n), EXT)
        # The generic functions are owned by the submodule (a package adds
        # methods to `ERGM.Extension.attainable_range`, not to a copy)
        for n in EXT
            @test parentmodule(getfield(ERGM.Extension, n)) === ERGM.Extension
        end
        # `using ERGM.Extension` brings every name in, unqualified
        m = Module(:ExtensionUser)
        Core.eval(m, :(using ERGM; using ERGM.Extension))
        @test all(n -> isdefined(m, n), EXT)

        # No underscore name of ERGM is public: an underscore says private
        @test isempty([n for n in names(ERGM) if startswith(string(n), "_")])
        # The underscore names the extension API replaced are gone, the two
        # refusals behind `require_supported_network` stay internal, and the
        # dead helpers are deleted
        for n in (:_boundary_columns_iterated, :_validate_formula, :_materialize,
                  :_expand_terms, :_collect_terms, :_n_dyads, :_mcmc_defaults,
                  :_confidence_test, :_mcmle_solve, :_bridge_integrate, :_ess_sample,
                  :_warn_boundary, :_mple_fit_design, :_mcmle_covariance,
                  :_extreme_statistics, :_attainable_range, :_separated,
                  :_warn_separated, :_boundary_columns)
            @test !isdefined(ERGM, n)
        end
        for n in (:_refuse_two_mode, :_refuse_self_loops, :_hummel_step, :_bridge_quadrature,
                  :_warn_degenerate_stats, :_bridge_logZ, :_boundary_columns_once,
                  :_copy_network, :_warn_aliased)
            @test isdefined(ERGM, n) && !Base.ispublic(ERGM, n)
        end
        # The 0.1-era private aliases are gone (their public names are the API)
        for n in (:_requires_directed, :_requires_undirected, :_vertex_attribute,
                  :_has_dyad_dependent, :_z_pvalues)
            @test !isdefined(ERGM, n)
        end

        # Every Extension name has a consumer in some variant's `src/`, or is
        # an extension point documented as such (a method a package adds for
        # its own types). The scan runs where the variants sit beside ERGM.jl
        # (the ecosystem layout CI reconstructs); elsewhere it is skipped.
        EXTENSION_POINTS = (:attainable_range, :materialize)
        for n in EXTENSION_POINTS
            b = Base.Docs.Binding(ERGM.Extension, n)
            doc = join((join(string.(ds.text), "") for m in (ERGM, ERGM.Extension)
                        if haskey(Base.Docs.meta(m), b)
                        for ds in values(Base.Docs.meta(m)[b].docs)), "\n")
            @test occursin("extension point", doc)
        end
        variants = ("TERGM", "ERGMCount", "ERGMRank", "ERGMEgo", "ERGMMulti", "ERGMUserterms")
        roots = [joinpath(dirname(pkgdir(ERGM)), v * ".jl", "src") for v in variants]
        if all(isdir, roots)
            code = join((read(joinpath(r, f), String) for r in roots
                         for f in readdir(r) if endswith(f, ".jl")), "\n")
            for n in EXT
                used = occursin(Regex("\\b(?:Extension\\.)?" * string(n) * "\\b"), code)
                @test used || n in EXTENSION_POINTS
            end
            # ... and no variant reaches into an underscore name of ERGM for
            # something the extension API provides
            for n in (:_boundary_columns_iterated, :_validate_formula, :_materialize,
                      :_expand_terms, :_collect_terms, :_n_dyads, :_mcmc_defaults,
                      :_confidence_test, :_mcmle_solve, :_bridge_integrate, :_ess_sample,
                      :_warn_boundary, :_mple_fit_design, :_mcmle_covariance,
                      :_extreme_statistics, :_attainable_range, :_refuse_two_mode,
                      :_refuse_self_loops)
                @test !occursin("ERGM." * string(n), code)
            end
        else
            @info "Extension consumer scan skipped: the variant packages are not beside ERGM.jl"
        end

        # R's default-estimator rule is public, for the variants
        @test Base.ispublic(ERGM, :resolve_method) && !Base.isexported(ERGM, :resolve_method)
        @test Base.isexported(ERGM, :has_dyad_dependent)
        @test Base.ispublic(ERGM, :TriadCensus) && !Base.isexported(ERGM, :TriadCensus)

        # `expand_terms`: the specification — ERGM's expansion of levels,
        # cells and degrees against the network — as plain terms without a
        # snapshot; the same statistics, in the same order, under the same
        # names as `materialize`
        fmh = load_dataset(:faux_mesa_high)
        raw = [Edges(), NodeFactor(:Grade), NodeMix(:Sex), Degree(0:2), GWESP(0.5)]
        spec = ERGM.Extension.expand_terms(raw, fmh)
        mat = ERGM.Extension.materialize(TermSet(raw), fmh)
        @test spec isa Vector{AbstractERGMTerm}
        # 1 + 5 non-base Grade levels + 2 Sex cells (F-F is the base) + 3 degrees + 1
        @test length(spec) == length(mat.terms) == 1 + 5 + 2 + 3 + 1
        # plain terms, never a materialized twin (a twin's `base` is a term;
        # NodeFactor's own `base` field is its vector of base levels)
        plain(t) = !(hasfield(typeof(t), :base) && fieldtype(typeof(t), :base) <: AbstractERGMTerm)
        @test all(plain, spec)
        @test !all(plain, collect(mat.terms))
        @test [name(t, fmh) for t in spec] == mat.names
        @test [compute(t, fmh) for t in spec] == [compute(t, fmh) for t in mat.terms]
        @test [name(t) for t in ERGM.Extension.expand_terms(TermSet(raw), fmh)] == [name(t) for t in spec]
        @test ERGM.Extension.expand_terms(Edges(), fmh) == [Edges()]
        # Materializing the specification is the materialized formula
        @test ERGM.Extension.materialize(TermSet(spec), fmh).names == mat.names

        # `mple_fit_design(...; context=)` prefixes both R sentences with the
        # caller's name (TERGM's "cmple"), so a variant needs neither `_warn_*`
        Xb = [1.0 0.0; 1.0 2.0]
        @test_logs (:warn, r"^cmple: observed statistic\(s\) b are at their smallest") (:warn, r"^cmple: observed statistic\(s\) a are at their largest") ERGM.Extension.mple_fit_design(
            Xb, [3.0, 2.0], [3.0, 0.0], ["a", "b"]; context="cmple")
        # ... and the sentence's variable parts: `note` replaces the R parenthesis,
        # `noun` the rows the rest is fitted on
        @test_logs (:warn, r"exists; ergm.rank has no drop\)\. The remaining coefficients are estimated on the swap comparisons these") ERGM.Extension.warn_boundary(
            ["a"], [(1, :max)]; context="fit_ergm_rank", noun="swap comparisons",
            note="ergm.rank has no drop")

        # `note=` words every closing sentence of `mple_fit_design`'s warnings
        # for a package whose R counterpart behaves differently; a key not
        # given keeps R ergm's sentence, an unknown key is refused
        @test_logs (:warn, r"^multi: observed statistic\(s\) b .*exists; ergm.multi differs\)\. The remaining coefficients are estimated on the layer dyads") (:warn, r"^multi: observed statistic\(s\) a .*exists; ergm.multi differs\)") ERGM.Extension.mple_fit_design(
            Xb, [3.0, 2.0], [3.0, 0.0], ["a", "b"]; context="multi", noun="layer dyads",
            note=(boundary="ergm.multi differs",))
        Za = [1.0 0.0 0.0 2.0; 1.0 0.0 1.0 2.0]       # x2 never varies, x4 = 2·edges
        @test_logs (:warn, r"^v: statistic\(s\) x2 do not vary on the comparisons fitted .*\(not varying here\)") (:warn, r"^v: statistic\(s\) x4 are linear combinations.*the swap-MPLE without them \(dependent here\)") ERGM.Extension.mple_fit_design(
            Za, [10.0, 10.0], [3.0, 6.0], ["edges", "x2", "x3", "x4"]; context="v",
            estimate="swap-MPLE", noun="comparisons",
            note=(not_varying="not varying here", linear_dependence="dependent here"))
        @test_logs (:warn, r"^v: statistic\(s\) x2 do not vary on the dyads fitted .*R ergm warns \"Model statistics") (:warn, r"R's glm reports NA") ERGM.Extension.mple_fit_design(
            Za, [10.0, 10.0], [3.0, 6.0], ["edges", "x2", "x3", "x4"]; context="v")
        @test_throws ArgumentError ERGM.Extension.mple_fit_design(
            Xb, [3.0, 2.0], [1.0, 1.0], ["a", "b"]; note=(boundry="typo",))
        @test_throws ArgumentError ERGM.Extension.mple_fit_design(
            Xb, [3.0, 2.0], [1.0, 1.0], ["a", "b"]; note=(boundary=1,))
        # A separated design: the closing sentence is replaced too, and the
        # verdict comes back with the fit, so no caller recomputes it
        # (ties exactly where x > z: separated by x − z, while no single
        # column is at its boundary)
        Xs = [1.0 1.0 0.0; 1.0 0.0 1.0; 1.0 1.0 1.0; 1.0 0.0 0.0]
        ts_, os_ = fill(5.0, 4), [5.0, 0.0, 2.0, 2.0]
        @test isempty(ERGM.Extension.boundary_columns(Xs, ts_, os_))
        rs = @test_logs (:warn, r"^v: the MPLE does not exist \(separation\).* my package warns too\.$"s) match_mode=:any ERGM.Extension.mple_fit_design(
            Xs, ts_, os_, ["edges", "x", "z"]; context="v",
            note=(separation="my package warns too.",))
        @test rs.verdict isa NetworkCore.SeparationVerdict && rs.verdict.separated
        v_again = NetworkCore.logistic_separation(Xs[rs.kept_rows, rs.fitted], ts_, os_)
        @test rs.verdict.separated == v_again.separated && rs.verdict.terms == v_again.terms
        @test rs.separated && !rs.converged && rs.fitted == [1, 2, 3]
        @test rs.separated_terms == ["edges", "x", "z"][rs.fitted][rs.verdict.terms]
        @test issubset(["x", "z"], rs.separated_terms)
        # The result names what was dropped and aliased, the columns fitted
        # and the rows they were fitted on, consistently with the coefficients
        za = ERGM.Extension.mple_fit_design(Za, [10.0, 10.0], [3.0, 6.0],
                                            ["edges", "x2", "x3", "x4"]; warn=false)
        @test za.aliased == [2, 4] && isempty(za.boundary) && za.fitted == [1, 3]
        @test za.kept_rows == [1, 2] && !za.verdict.separated
        @test isnan(za.coefficients[2]) && isnan(za.coefficients[4])
        @test za.coefficients[1] ≈ log(3 / 7) && za.coefficients[3] ≈ log(6 / 4) - log(3 / 7)
        zb = ERGM.Extension.mple_fit_design(Xb, [3.0, 2.0], [3.0, 0.0], ["a", "b"]; warn=false)
        @test zb.boundary == [(1, :max), (2, :min)] && isempty(zb.fitted)
        @test zb.verdict === nothing && zb.coefficients == [Inf, -Inf]

        # `attainable_range` is a generic a package extends for its own term
        # or network type; `extreme_statistics` dispatches through it (and
        # `compute`), and ERGM's own methods never claim a bound on a foreign
        # network type
        mat10 = network(10; directed=false)
        for i in 1:2:9
            add_edge!(mat10, i, i + 1)
        end
        @test ERGM.Extension.extreme_statistics([Edges(), Triangle()], mat10) == [(2, :min)]
        @test ERGM.Extension.extreme_statistics(TermSet([Edges(), Triangle()]), mat10) == [(2, :min)]
        @test ERGM.Extension.extreme_statistics([Offset(Triangle(), -1.0), Edges()], mat10) == Tuple{Int,Symbol}[]
        @test ERGM.Extension.attainable_range(Edges(), mat10) == (0.0, 45.0)
        @test ERGM.Extension.attainable_range(Offset(Edges(), 1.0), mat10) == (0.0, 45.0)
        @test ERGM.Extension.attainable_range(Edges(), ExtForeignNetwork()) == (-Inf, Inf)
        @test ERGM.Extension.extreme_statistics([ExtIsolatesUnranged()], mat10) == Tuple{Int,Symbol}[]
        @test ERGM.Extension.extreme_statistics([ExtIsolates()], mat10) == [(1, :min)]
        @test ERGM.Extension.extreme_statistics([ExtIsolates()], network(4; directed=false)) == [(1, :max)]

        # `require_supported_network` gives the model's own refusals
        bp = network(5; bipartite=2)
        e1 = try ERGM.Extension.require_supported_network(bp) catch err; err end
        e2 = try ERGMModel(ERGMFormula([Edges()]), bp) catch err; err end
        @test e1 isa ArgumentError && e1.msg == e2.msg
        loopy = network(3; loops=true); add_edge!(loopy, 2, 2)
        e3 = try ERGM.Extension.require_supported_network(loopy) catch err; err end
        e4 = try ERGMModel(ERGMFormula([Edges()]), loopy) catch err; err end
        @test e3 isa ArgumentError && e3.msg == e4.msg && occursin("at vertex 2", e3.msg)
        ok = network(3; loops=true)
        @test ERGM.Extension.require_supported_network(ok) === ok

        # `_fmt3` prints three significant digits, never a rounded Float64's
        # decimal expansion (`6.969999999999999e-32`)
        @test ERGM._fmt3(6.97e-32) == "6.97e-32"
        @test ERGM._fmt3(1.67e33) == "1.67e+33"
        @test ERGM._fmt3(0.12345) == "0.123" && ERGM._fmt3(Inf) == "Inf" && ERGM._fmt3(NaN) == "NaN"

        flo = florentine_marriage()
        @test !has_dyad_dependent(ERGMModel(ERGMFormula([Edges(), NodeCov(:wealth)]), flo))
        @test has_dyad_dependent(ERGMModel(ERGMFormula([Edges(), Triangle()]), flo))
    end

    @testset "ergm is a const alias of fit_ergm" begin
        @test ergm === fit_ergm
        @test occursin("fit_ergm", string(@doc ergm))
        @test occursin("ergm", string(@doc fit_ergm))
        flo = florentine_marriage()
        @test coef(ergm(flo, [Edges()])) == coef(fit_ergm(flo, [Edges()]))
    end

    @testset "Keyword vocabulary: maxiter" begin
        flo = florentine_marriage()
        kw = Base.kwarg_decl(first(methods(mcmle)))
        @test :maxiter in kw
        @test :maxiter in Base.kwarg_decl(first(methods(mple)))
        # The 0.2 additions share the ecosystem vocabulary: n_chains, n_samples,
        # rng, init (statnet's control.ergm(init=)), bridge_rungs
        for name in (:n_chains, :n_samples, :rng, :init, :bridge_rungs, :burnin, :interval)
            @test name in kw
        end

        @test !(:max_iter in kw) && !(:tol in kw)

        # An unconverged MCMLE is loud: warned, recorded, and listed
        unconv = @test_logs (:warn, r"did not converge") match_mode=:any fit_ergm(
            flo, [Edges(), GWESP(0.5)]; method=:mcmle, n_samples=30, burnin=50,
            interval=2, maxiter=1, gamma0=0.01, bridge_rungs=1, bridge_samples=10,
            rng=Random.Xoshiro(9))
        @test !unconv.converged
        @test any(occursin("did not converge", a) for a in approximations(unconv))
    end

    @testset "ERGMModel{T,D}: directedness is a type parameter" begin
        for make in (fixture_undirected, fixture_directed)
            net = make()
            m = ERGMModel(ERGMFormula([Edges(), Triangle()]), net)
            @test m isa ERGMModel{Int, is_directed(net)}
            @test isconcretetype(fieldtype(typeof(m), :network))
            @test is_directed(m) == is_directed(net)
            @test is_directed(typeof(m)) == is_directed(net)
            @test !hasfield(typeof(m), :directed)
            @test ERGM.Extension.n_observed_dyads(m) == (is_directed(net) ? 42 : 21)
            fit = mple(m)
            @test fit isa ERGMResult{Int, is_directed(net)}
            sims = sample_networks(m, coef(fit); n_sim=3, burnin=50, interval=5,
                                   rng=Random.Xoshiro(1))
            @test eltype(sims) === Network{Int, is_directed(net)}
            @test eltype(sample_networks(m, coef(fit); n_sim=0)) === Network{Int, is_directed(net)}
            @test eltype(simulate_ergm(fit; n_sim=2, burnin=50, interval=5)) ===
                  Network{Int, is_directed(net)}
        end
    end

    @testset "StatsAPI surface is complete: confint and coeftable" begin
        flo = florentine_marriage()
        fits = (fit_ergm(flo, [Edges(), NodeCov(:wealth)]),
                fit_ergm(flo, [Edges(), NodeCov(:wealth)]; method=:mcmle, n_samples=100,
                         burnin=200, interval=5, maxiter=2, bridge_rungs=2,
                         bridge_samples=20, rng=Random.Xoshiro(4)),
                fit_ergm(flo, [Edges()]; se=:bootstrap, n_boot=5, boot_burnin=200,
                         boot_interval=20, rng=Random.Xoshiro(5)))
        for fit in fits
            @test NetworkCore.check_statsapi(fit; strict=true) !== nothing
            # `coefnames` (R's names(coef(fit))): required, the StatsAPI
            # binding, and the coefficient table's labels
            @test all(values(NetworkCore.check_statsapi(fit;
                required=(NetworkCore.STATSAPI_VERBS..., :coefnames), strict=true)))
            @test coefnames(fit) == coeftable(fit).names == fit.model.formula.terms.names
            @test coefnames === StatsAPI.coefnames === NetworkCore.coefnames
            c = coefnames(fit); c[1] = "x"
            @test coefnames(fit)[1] != "x"          # a copy
            ci = confint(fit)
            @test size(ci) == (length(coef(fit)), 2)
            @test all(ci[:, 1] .< coef(fit) .< ci[:, 2])
            ci90 = confint(fit; level=0.9)
            @test all(ci90[:, 1] .> ci[:, 1]) && all(ci90[:, 2] .< ci[:, 2])
            @test ci[:, 2] .- coef(fit) ≈ 1.959963984540054 .* stderror(fit) atol = 1e-12
            tbl = coeftable(fit)
            @test tbl isa CoefficientTable
            @test tbl.names == fit.model.formula.terms.names
            @test tbl.estimates == coef(fit)
            @test tbl.std_errors == stderror(fit)
            @test tbl.p_values == fit.p_values
            # The printed table IS the inspected one
            @test occursin(sprint(show, tbl), sprint(show, fit))
        end
        @test_throws ArgumentError confint(fits[1]; level=1.5)
        @test coeftable(fits[1])["edges"].estimate == coef(fits[1])[1]
    end

    @testset "Actionable errors: single term, swapped arguments, non-terms" begin
        flo = florentine_marriage()

        # A single term needs no brackets
        @test coef(fit_ergm(flo, Edges())) == coef(fit_ergm(flo, [Edges()]))

        # An unknown NodeCov transform is refused at construction (it used to
        # be accepted: `transform=:exp` silently computed the untransformed statistic)
        e = try NodeCov(:wealth; transform=:exp); nothing catch err; err end
        @test e isa ArgumentError
        @test occursin(":exp", e.msg) && occursin(":log", e.msg) && occursin(":sqrt", e.msg)
        @test NodeCov(:wealth; transform=:sqrt).transform === :sqrt

        # Swapped positional arguments
        for bad in (() -> fit_ergm([Edges()], flo), () -> fit_ergm(Edges(), flo),
                    () -> ergm([Edges(), Triangle()], flo))
            e = try bad(); nothing catch err; err end
            @test e isa ArgumentError
            @test occursin("arguments are swapped", e.msg)
            @test occursin("fit_ergm(net, terms)", e.msg)
        end

        # A Vector{Any} of terms is fine; a non-term element is named with its
        # position, and a term TYPE gets the "did you mean Edges()?" hint
        @test coef(fit_ergm(flo, Any[Edges(), NodeCov(:wealth)])) ==
              coef(fit_ergm(flo, [Edges(), NodeCov(:wealth)]))
        e = try fit_ergm(flo, [Edges(), 3]); nothing catch err; err end
        @test e isa ArgumentError
        @test occursin("element 2", e.msg) && occursin("Int64", e.msg)
        e = try fit_ergm(flo, [Edges]); nothing catch err; err end
        @test e isa ArgumentError
        @test occursin("did you mean `Edges()`", e.msg)
        @test_throws ArgumentError fit_ergm(flo, AbstractERGMTerm[])
        @test_throws ArgumentError fit_ergm(flo, [Edges()]; method=:nope)

        # `Degree(0:2)` is one expanding term (statnet `edges + degree(0:2)`):
        # the natural spelling is a Vector{<:AbstractERGMTerm} and fits
        f = fit_ergm(flo, [Edges(), Degree(0:2)]; method=:mple)
        @test f.model.formula.terms.names == ["edges", "degree0", "degree1", "degree2"]
        @test summary_stats(flo, [Edges(), Degree(0:2)]).degree0 == 1.0   # Pucci
        @test [Edges(), Degree(0:2)] isa Vector{<:AbstractERGMTerm}
        @test ERGMFormula([Edges(), Degree(0:2)]).terms.names == ["edges", "degree(0,1,2)"]
        # Nested vectors of terms are still spliced for programmatic callers
        @test ERGMFormula([Edges(), [Degree(0), Degree(1)]]).terms.names ==
              ["edges", "degree0", "degree1"]

        # An EdgeCov matrix built for another network is a formula error
        # (was a raw BoundsError at fit time)
        e = try ergm(flo, [Edges(), EdgeCov(rand(5, 5))]); nothing catch err; err end
        @test e isa ArgumentError
        @test occursin("5×5", e.msg) && occursin("16 vertices", e.msg) && occursin("n×n", e.msg)
        @test_throws ArgumentError compute(EdgeCov(rand(5, 5)), flo)
        @test_throws ArgumentError change_stat(EdgeCov(rand(5, 5)), flo, 1, 2)
        @test_throws ArgumentError ERGM.Extension.validate_formula(TermSet([EdgeCov(zeros(17, 17))]), flo)
        M = [Float64(i + j) for i in 1:16, j in 1:16]
        @test coef(ergm(flo, [Edges(), EdgeCov(M)])) isa Vector{Float64}

        # gof validates its `stats` before simulating anything: a typo names
        # the valid symbols instead of running every simulation and then
        # failing on an empty result (or silently dropping the panel)
        fe = fit_ergm(flo, [Edges()])
        e = try gof(fe; stats=[:degre], n_sim=2, burnin=1, interval=1); nothing catch err; err end
        @test e isa ArgumentError
        @test occursin(":degre", e.msg) && occursin(":idegree", e.msg) && occursin(":distance", e.msg)
        @test_throws ArgumentError gof(fe; stats=[:degree, :esp, :nope], n_sim=2, burnin=1, interval=1)
        @test_throws ArgumentError gof(fe; stats=Symbol[], n_sim=2, burnin=1, interval=1)

        # `show` of a model or formula prints the formula, never the struct
        # with its materialized attribute vectors
        mshow = sprint(show, ERGMModel(ERGMFormula([Edges(), NodeCov(:wealth)]), flo))
        @test mshow == "ERGMModel{Int64,false}: 16 vertices, 20 edges (undirected); terms: edges + nodecov.wealth"
        @test !occursin("[10.0", mshow) && !occursin("Materialized", mshow)
        @test sprint(show, ERGMFormula([Edges(), NodeCov(:wealth)])) ==
              "ERGMFormula: edges + nodecov.wealth"
        mk = copy(flo); set_missing_dyad!(mk, 1, 9)
        @test occursin("1 masked dyad;", sprint(show, ERGMModel(ERGMFormula([Edges()]), mk)))
    end

    @testset "Refused, not mis-fit: self-loops" begin
        # A self-loop would be counted by the statistics but never by the
        # pseudo-likelihood, nobs, the proposal or a simulation (R ergm warns
        # "This network contains loops"); ERGM.jl refuses it
        ln = network(6; directed=false, loops=true)
        add_edge!(ln, 1, 2); add_edge!(ln, 2, 3); add_edge!(ln, 3, 3)
        @test has_edge(ln, 3, 3)
        @test compute(Edges(), ln) == 3.0            # the raw statistic counts it
        e = try ERGMModel(ERGMFormula([Edges()]), ln); nothing catch err; err end
        @test e isa ArgumentError
        @test occursin("1 self-loop", e.msg) && occursin("vertex 3", e.msg)
        @test occursin("contains loops", e.msg)      # R's own words
        @test_throws ArgumentError fit_ergm(ln, [Edges()])
        dl = network(4; directed=true, loops=true)
        add_edge!(dl, 1, 2); add_edge!(dl, 2, 2); add_edge!(dl, 4, 4)
        e = try fit_ergm(dl, Edges()); nothing catch err; err end
        @test e isa ArgumentError && occursin("2 self-loops", e.msg)
        # A loops-permitted network WITHOUT a loop is the loop-free network
        # it is: modelled on its n(n−1)/2 off-diagonal dyads
        ln0 = network(6; directed=false, loops=true)
        add_edge!(ln0, 1, 2); add_edge!(ln0, 2, 3)
        f = fit_ergm(ln0, [Edges()])
        @test nobs(f) == 15
        @test coef(f)[1] ≈ log((2 / 15) / (13 / 15)) atol = 1e-8
        sims = simulate_ergm(f; n_sim=3, burnin=50, interval=1, rng=Random.Xoshiro(1))
        @test all(!has_edge(s, v, v) for s in sims for v in 1:6)
    end

    @testset "Boundary statistics: R's drop semantics (perfect separation)" begin
        # The docs' former Quick Start network: 12 edges, every one between
        # different `mod1(v, 3)` groups, so nodematch.group = 0 is at its
        # smallest attainable value. R ergm 4.12.0: "Observed statistic(s)
        # nodematch.group are at their smallest attainable values. Their
        # coefficients will be fixed at -Inf."; coef = (-3.178054, -Inf),
        # SE = (0.2946269, 0), logLik -50.38324 — the edges coefficient is
        # logit(12/300): the fit on the 300 dyads the dropped term does not
        # touch, the exact limit of the pseudo-likelihood (NOT logit(12/435)
        # = -3.562, the edges-only fit). Newton on the full design used to
        # "converge" at -21 with an SE of 21,000 and no warning.
        # Every R number here is a provenanced row of ergm_terms.toml
        # (section (e) of test/fixtures/r/ergm_terms.R), including the
        # df/nobs rule behind AIC/BIC: df counts the FINITE coefficients and
        # logLik's nobs is the 300 dyads the dropped term does not touch,
        # while nobs(fit) stays every dyad (435).
        g = load_golden(joinpath(@__DIR__, "fixtures", "ergm_terms.toml"))
        net = network(30; directed=false)
        for (i, j) in [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5),
                       (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
            add_edge!(net, i, j)
        end
        set_vertex_attribute!(net, :group, Dict(v => ("A", "B", "C")[mod1(v, 3)] for v in 1:30))
        @test summary_stats(net, [NodeMatch(:group)]).var"nodematch.group" == 0.0
        fit = @test_logs (:warn, r"nodematch.group are at their smallest attainable values") match_mode=:any ergm(
            net, [Edges(), NodeMatch(:group)])
        @test g.values["sep_terms"] == fit.model.formula.terms.names
        @test check_golden(g, "sep_coefficients", coef(fit)) ||
              error(golden_report(g, "sep_coefficients", coef(fit)))
        @test coef(fit)[1] ≈ log((12 / 300) / (288 / 300)) atol = 1e-8
        @test check_golden(g, "sep_std_errors", stderror(fit)) ||
              error(golden_report(g, "sep_std_errors", stderror(fit)))
        @test check_golden(g, "sep_loglik", loglikelihood(fit))
        @test dof(fit) == g.values["sep_df"] == 1
        @test nobs(fit) == g.values["sep_nobs"] == 435
        @test g.values["sep_loglik_nobs"] == 300
        @test check_golden(g, "sep_aic", aic(fit)) || error(golden_report(g, "sep_aic", aic(fit)))
        @test check_golden(g, "sep_bic", bic(fit)) || error(golden_report(g, "sep_bic", bic(fit)))
        @test bic(fit) ≈ -2 * loglikelihood(fit) + log(300) atol = 1e-10
        @test fit.z_values[2] == -Inf && fit.p_values[2] == 0.0
        @test fit.converged
        # An extended-value estimate is not the exact MLE
        @test !is_exact(fit)
        # ... and cannot be bootstrapped (no network can be simulated at -Inf)
        e = try ergm(net, [Edges(), NodeMatch(:group)]; se=:bootstrap, n_boot=5); nothing catch err; err end
        @test e isa ArgumentError
        @test occursin("se=:bootstrap", e.msg) && occursin("nodematch.group", e.msg)
        @test vcov(fit)[2, :] == zeros(2) && vcov(fit)[:, 2] == zeros(2)
        @test any(occursin("fixed at -Inf", a) for a in approximations(fit))
        printed = sprint(show, fit)
        @test occursin("-Inf", printed) && occursin("nodematch.group fixed at -Inf", printed)
        # The largest attainable value, symmetrically: every within-group
        # dyad tied
        g2 = network(6; directed=false)
        set_vertex_attribute!(g2, :s, Dict(1 => "a", 2 => "a", 3 => "b", 4 => "b", 5 => "c", 6 => "c"))
        for (i, j) in [(1, 2), (3, 4), (5, 6), (1, 3)]
            add_edge!(g2, i, j)
        end
        f2 = @test_logs (:warn, r"largest attainable values") match_mode=:any ergm(g2, [Edges(), NodeMatch(:s)])
        @test coef(f2)[2] == Inf && stderror(f2)[2] == 0.0
        @test coef(f2)[1] ≈ log((1 / 12) / (11 / 12)) atol = 1e-8   # 1 of the 12 between-group dyads
        # Every coefficient at the boundary: the empty network under `edges`
        empty6 = network(6; directed=false)
        f3 = @test_logs (:warn, r"edges are at their smallest") match_mode=:any ergm(empty6, Edges())
        @test coef(f3) == [-Inf] && loglikelihood(f3) == 0.0 && dof(f3) == 0
        # R's attainable-range test: a column is at its minimum iff every
        # positive-change dyad is a non-tie AND every negative-change dyad is
        # a tie — the sign pattern is irrelevant (a mixed-sign
        # column used to be skipped as "no monotone direction")
        @test ERGM._boundary_columns_once([1.0 -1.0; 1.0 1.0], [3.0, 2.0], [0.0, 0.0]) == [(1, :min)]
        @test isempty(ERGM._boundary_columns_once(reshape([1.0, -1.0], 2, 1), [3.0, 2.0], [0.0, 0.0]))
        @test ERGM._boundary_columns_once(reshape([1.0, -1.0], 2, 1), [3.0, 2.0], [0.0, 2.0]) == [(1, :min)]
        @test ERGM._boundary_columns_once(reshape([1.0, -1.0], 2, 1), [3.0, 2.0], [3.0, 0.0]) == [(1, :max)]
        @test isempty(ERGM._boundary_columns_once(reshape([1.0, -1.0], 2, 1), [3.0, 2.0], [3.0, 2.0]))
        @test ERGM._boundary_columns_once([1.0 0.0; 1.0 2.0], [3.0, 2.0], [0.0, 0.0]) ==
              [(1, :min), (2, :min)]
        @test ERGM._boundary_columns_once([1.0 -2.0; 1.0 0.0], [3.0, 2.0], [3.0, 2.0]) ==
              [(1, :max), (2, :min)]
        # ... iterated on the reduced design: column 2 is at its minimum (its
        # only row is empty); once it is dropped the untouched row (row 1) is
        # all ties, so column 1 — not a boundary of the FULL design — is at
        # its maximum on the reduced one
        @test ERGM._boundary_columns_once([1.0 0.0; 1.0 2.0], [3.0, 2.0], [3.0, 0.0]) == [(2, :min)]
        @test ERGM.Extension.boundary_columns([1.0 0.0; 1.0 2.0], [3.0, 2.0], [3.0, 0.0]) ==
              [(1, :max), (2, :min)]
        @test isempty(ERGM.Extension.boundary_columns([1.0 0.0; 1.0 2.0], [3.0, 2.0], [1.0, 1.0]))

        # The mixed-sign case (fixture section (f)): 8 nodes, x =
        # -4..3, tie iff x_i + x_j ≤ 0, so nodecov.x = -50 is the smallest
        # value the statistic can attain although its change statistic
        # x_i + x_j takes both signs. R: "The MPLE does not exist!". ERGM.jl
        # used to return coef = [20.8, -41.8], SE = [19177, 28256], p = 0.999
        # and converged == true with no warning at all.
        mix = network(Int(g.values["mix_n"]); directed=false)
        mx = Int.(g.values["mix_x"])
        for i in 1:7, j in (i + 1):8
            mx[i] + mx[j] <= 0 && add_edge!(mix, i, j)
        end
        set_vertex_attribute!(mix, :x, Dict(v => mx[v] for v in eachindex(mx)))
        @test g.values["mix_summary_names"] == ["edges", "nodecov.x"]
        @test check_golden(g, "mix_summary", collect(values(summary_stats(mix, [Edges(), NodeCov(:x)]))))
        @test occursin("MPLE does not exist", g.values["r_warning_mple_nonexistent"])
        fmix = @test_logs (:warn, r"nodecov.x are at their smallest attainable values") (:warn, r"edges are at their largest attainable values") match_mode=:any ergm(
            mix, [Edges(), NodeCov(:x)])
        # The exact limit of the pseudo-likelihood: nodecov.x → -Inf, and on
        # the three dyads it does not touch (x_i + x_j = 0, all tied) edges → +Inf
        @test coef(fmix) == [Inf, -Inf] && stderror(fmix) == [0.0, 0.0]
        @test loglikelihood(fmix) == 0.0 && dof(fmix) == 0
        @test !is_exact(fmix)
        @test any(occursin("fixed at -Inf", a) for a in approximations(fmix))
        # MCMLE under drop=false refuses BEFORE any MPLE start is computed:
        # the refusal and nothing else — no `mple:` warning on the way (the
        # design used to be built twice and the drop warning preceded the
        # ArgumentError)
        logs, emix = Test.collect_test_logs() do
            try ergm(mix, [Edges(), NodeCov(:x)]; method=:mcmle, n_samples=20,
                     burnin=10, interval=1, drop=false); nothing catch err; err end
        end
        @test emix isa ArgumentError && occursin("nodecov.x", emix.msg) &&
              occursin("drop=false", emix.msg)
        @test isempty(logs)
        # Under the default drop=true both columns are fixed (R's drop), and
        # then there is nothing left to estimate: said, naming them
        emix2 = try ergm(mix, [Edges(), NodeCov(:x)]; method=:mcmle, n_samples=20,
                         burnin=10, interval=1); nothing catch err; err end
        @test emix2 isa ArgumentError && occursin("nothing to estimate", emix2.msg) &&
              occursin("nodecov.x", emix2.msg)

        # Separated by a COMBINATION of columns (fixture section (f')): no
        # single nodecov column is at its boundary, yet nodecov.x + nodecov.z
        # predicts every tie. R's LP warns "The MPLE does not exist!"; so does
        # the shared NetworkCore verdict (an exact linear programme), which
        # names the two columns that carry the separating direction. The
        # ecosystem's policy: warn, converged = false, flag the terms,
        # withhold z, p and intervals.
        sep2 = network(Int(g.values["sep2_n"]); directed=false)
        sx, sz = Int.(g.values["sep2_x"]), Int.(g.values["sep2_z"])
        for i in 1:5, j in (i + 1):6
            (sx[i] + sz[i]) + (sx[j] + sz[j]) > 0 && add_edge!(sep2, i, j)
        end
        set_vertex_attribute!(sep2, :x, Dict(v => sx[v] for v in 1:6))
        set_vertex_attribute!(sep2, :z, Dict(v => sz[v] for v in 1:6))
        @test check_golden(g, "sep2_summary",
                           collect(values(summary_stats(sep2, [Edges(), NodeCov(:x), NodeCov(:z)]))))
        @test occursin("MPLE does not exist", g.values["r_warning_mple_nonexistent_combination"])
        m2 = ERGMModel(ERGMFormula([Edges(), NodeCov(:x), NodeCov(:z)]), sep2)
        @test isempty(ERGM.Extension.boundary_columns(ERGM._mple_data(sep2, m2.formula.terms, false)...))
        fsep2 = @test_logs (:warn, r"^mple: the MPLE does not exist \(separation\).*`nodecov.x`, `nodecov.z`.*The MPLE does not exist!") match_mode=:any ergm(
            sep2, [Edges(), NodeCov(:x), NodeCov(:z)])
        @test !fsep2.converged
        @test !is_exact(fsep2)
        @test fsep2.separated_terms == ["nodecov.x", "nodecov.z"]
        @test all(isfinite, coef(fsep2))       # the last iterate, returned but flagged
        @test all(isnan, fsep2.z_values) && all(isnan, fsep2.p_values)   # inference withheld
        @test all(isnan, confint(fsep2))
        @test all(isfinite, stderror(fsep2))   # kept, for diagnosis
        e3 = try ergm(sep2, [Edges(), NodeCov(:x), NodeCov(:z)]; method=:mple,
                      se=:bootstrap, n_boot=4); nothing catch err; err end
        @test e3 isa ArgumentError && occursin("nodecov.x", e3.msg)
        # The verdict is the shared one, on the design actually fitted
        X2, t2, o2 = ERGM._mple_data(sep2, m2.formula.terms, false)
        v2 = NetworkCore.logistic_separation(X2, t2, o2)
        @test v2.separated && v2.certified
        @test m2.formula.terms.names[v2.terms] == fsep2.separated_terms
        @test any(occursin("does not exist", a) for a in approximations(fsep2))
        @test occursin("Converged: false", sprint(show, fsep2))
        @test occursin("MPLE does not exist", sprint(show, fsep2))
        @test occursin("`nodecov.x`, `nodecov.z`", sprint(show, fsep2))
        e2 = try ergm(sep2, [Edges(), NodeCov(:x), NodeCov(:z)]; method=:mcmle, n_samples=20,
                      burnin=10, interval=1); nothing catch err; err end
        @test e2 isa ArgumentError && occursin("does not exist", e2.msg) && occursin("init=", e2.msg)
        # A finite maximum with an extreme but well-determined dyad class is
        # NOT flagged (the verdict depends on the data, not on how far Newton
        # ran), and a single column at its boundary is R's drop, not
        # separation
        X1 = [1.0 0.0; 1.0 30.0]
        f1 = ERGM.Extension.mple_fit_design(X1, [100.0, 5.0], [50.0, 1.0], ["edges", "x"]; warn=false)
        @test f1.converged && !f1.separated && isempty(f1.separated_terms)
        @test all(isfinite, f1.coefficients)
        f0 = ERGM.Extension.mple_fit_design(X1, [100.0, 5.0], [50.0, 0.0], ["edges", "x"]; warn=false)
        @test f0.converged && !f0.separated && f0.coefficients[2] == -Inf

        # MCMLE does what R's ergm() does (drop=TRUE): the coefficient fixed
        # at -Inf with R's sentence, the rest estimated — on a dyad-independent
        # formula the MLE on the 300 untouched dyads, which is the MPLE above
        fm = @test_logs (:warn, r"^mcmle: observed statistic\(s\) nodematch.group are at their smallest attainable values. Their coefficients will be fixed at -Inf") match_mode=:any ergm(
            net, [Edges(), NodeMatch(:group)]; method=:mcmle, n_samples=50, burnin=100,
            interval=10, rng=Xoshiro(3), bridge_rungs=2)
        @test coef(fm)[2] == -Inf && stderror(fm)[2] == 0.0 && fm.p_values[2] == 0.0
        @test coef(fm)[1] ≈ coef(fit)[1] atol = 1e-8
        @test dof(fm) == 1
        @test any(occursin("nodematch.group fixed at -Inf", a) for a in approximations(fm))
        @test occursin("nodematch.group fixed at -Inf", sprint(show, fm))
        # drop=false refuses, naming the term, R's sentence and the ways out
        e = try ergm(net, [Edges(), NodeMatch(:group)]; method=:mcmle, n_samples=20,
                     burnin=10, interval=1, drop=false); nothing catch err; err end
        @test e isa ArgumentError
        @test occursin("nodematch.group", e.msg) && occursin("smallest attainable", e.msg)
        @test occursin("drop=true", e.msg) && occursin("Offset(term, -Inf)", e.msg)
        # ... and so does the MPLE under drop=false
        e = try ergm(net, [Edges(), NodeMatch(:group)]; drop=false); nothing catch err; err end
        @test e isa ArgumentError && occursin("mple:", e.msg)
        # ... and an interior statistic is untouched (the fixture guards the
        # numbers; here only that no warning fires and nothing is dropped)
        flo = florentine_marriage()
        ok = @test_logs ergm(flo, [Edges(), NodeCov(:wealth)])
        @test all(isfinite, coef(ok))
        @test isempty(ERGM._boundary_columns_once(ERGM._mple_data(flo, ok.model.formula.terms, false)...))
    end

    @testset "Statistics with all-zero change statistics: R's attainable range" begin
        # A statistic whose change statistics are all ZERO on the observed
        # network gives the pseudo-likelihood a flat direction, not a
        # one-signed gradient, so the design's boundary test cannot see it;
        # Newton then stopped at its start on the singular information and
        # returned 0 for EVERY coefficient (17 of 139 random small fits), and
        # the default MCMLE, started there, could report a finite triangle
        # coefficient with `Converged: true` where R fixes it at -Inf. R
        # compares the observed statistic with the term's attainable range
        # (`ergm.checkextreme.model`); so does ERGM.jl now.
        g = network(10; directed=false)
        for k in 1:2:9; add_edge!(g, k, k + 1); end          # a perfect matching
        model = ERGMModel(ERGMFormula([Edges(), Triangle()]), g)
        X, nt, no = ERGM._mple_data(g, model.formula.terms, false)
        @test all(iszero, X[:, 2])                          # no dyad closes a two-path
        @test isempty(ERGM.Extension.boundary_columns(X, nt, no))   # invisible to the design
        @test ERGM.Extension.extreme_statistics(model.formula.terms, g) == [(2, :min)]
        f = @test_logs (:warn, r"triangle are at their smallest attainable values") match_mode=:any fit_ergm(
            g, [Edges(), Triangle()]; method=:mple)
        @test coef(f)[2] == -Inf && stderror(f)[2] == 0.0
        @test coef(f)[1] ≈ log(5 / 40) atol = 1e-8          # every dyad kept: logit(5/45)
        @test f.converged
        # Brute force: a saturated dyad-independent model with a singleton
        # level. Every finite coefficient reproduces its cell's observed tie
        # fraction (the score equations), a cell with no tie — or no dyad at
        # all, the singleton's own cell — is fixed at -Inf, a full one at
        # +Inf, and nothing stalls at 0
        rng = Random.Xoshiro(20261007)
        checked = 0
        for _ in 1:80
            n = rand(rng, 5:9)
            net = network(n; directed=false)
            labs = ["p", "q", "r"]
            a = [labs[rand(rng) < 0.12 ? 3 : rand(rng, 1:2)] for _ in 1:n]
            count(==("r"), a) == 1 || continue                 # a singleton level
            length(unique(a)) == 3 || continue
            dens = 0.2 + 0.5 * rand(rng)
            for i in 1:n, j in (i + 1):n
                rand(rng) < dens && add_edge!(net, i, j)
            end
            set_vertex_attribute!(net, :a, a)
            cell(i, j) = Tuple(sort([a[i], a[j]]))
            dy = Dict{Tuple{String,String},Int}(); ti = Dict{Tuple{String,String},Int}()
            for i in 1:n, j in (i + 1):n
                c = cell(i, j)
                dy[c] = get(dy, c, 0) + 1
                ti[c] = get(ti, c, 0) + (has_edge(net, i, j) ? 1 : 0)
            end
            base = ("p", "p")                                  # the dropped first cell
            (0 < get(ti, base, 0) < get(dy, base, 0)) || continue   # an interior reference
            fit = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                fit_ergm(net, [Edges(), NodeMix(:a)])
            end
            nms = fit.model.formula.terms.names
            θ = coef(fit)
            @test θ[1] ≈ log(ti[base] / (dy[base] - ti[base])) atol = 1e-6
            for k in 2:length(θ)
                l1, l2 = split(nms[k], ".")[3:4]
                c = Tuple(sort([String(l1), String(l2)]))
                d, t = get(dy, c, 0), get(ti, c, 0)
                if t == 0
                    @test θ[k] == -Inf
                elseif t == d
                    @test θ[k] == Inf
                else
                    @test 1 / (1 + exp(-(θ[1] + θ[k]))) ≈ t / d atol = 1e-6
                end
            end
            @test fit.converged && !any(isnan, θ)
            checked += 1
        end
        @test checked >= 10
    end

    @testset "Golden fixture: boundary statistics, R's drop and nodematch(diff=TRUE) (provenanced)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "boundary_ergm.toml"))
        # -Inf compared exactly; NA (nan) position by position, the rest at
        # the fixture's tolerance
        function agree(key, x)
            want = Float64.(g.values[key])
            length(want) == length(x) || return false
            isnan.(want) == isnan.(x) || return false
            ok = .!isnan.(want)
            tol = golden_tolerance(g, key)
            return all(isapprox(w, v; atol=tol) for (w, v) in zip(want[ok], x[ok]))
        end
        quiet(f) = Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())

        # (a) the 10-node matching ~ edges + triangle: the MPLE, and the
        # DEFAULT fit (an MCMLE with triangle fixed at -Inf, R's drop)
        ma = network(10; directed=false)
        for k in 1:2:9; add_edge!(ma, k, k + 1); end
        fa = quiet(() -> fit_ergm(ma, [Edges(), Triangle()]; method=:mple))
        @test fa.model.formula.terms.names == g.values["a_terms"]
        @test agree("a_mple_coefficients", coef(fa)) || error(golden_report(g, "a_mple_coefficients", coef(fa)))
        @test agree("a_mple_std_errors", stderror(fa))
        la = @test_logs (:warn, r"^mcmle: observed statistic\(s\) triangle are at their smallest attainable values") match_mode=:any fit_ergm(
            ma, [Edges(), Triangle()]; rng=Random.Xoshiro(11))
        @test la.method === :mcmle && la.converged
        @test coef(la)[2] == -Inf && stderror(la)[2] == 0.0
        @test abs(coef(la)[1] - g.values["a_mle_edges_mean"]) <= g.values["a_mle_tolerance"]
        @test isnan(loglikelihood(la))                      # a dyad-dependent constraint
        @test dof(la) == 1
        @test any(occursin("triangle fixed at -Inf", a) for a in approximations(la))
        @test occursin("triangle fixed at -Inf", sprint(show, la))
        # the sampler honours the bound: no simulated network has a triangle
        sims = simulate_ergm(la; n_sim=5, rng=Random.Xoshiro(2))
        @test all(compute(Triangle(), s) == 0 for s in sims)

        # (b) a singleton level ~ edges + nodemix: R's labels, R's -Inf cells
        mb = network(10; directed=false)
        for (i, j) in zip(g.values["b_tails"], g.values["b_heads"]); add_edge!(mb, i, j); end
        set_vertex_attribute!(mb, :a, String.(g.values["b_attribute"]))
        fb = quiet(() -> fit_ergm(mb, [Edges(), NodeMix(:a)]))
        @test fb.method === :mple && fb.converged
        @test fb.model.formula.terms.names == g.values["b_terms"]
        @test agree("b_mple_coefficients", coef(fb)) || error(golden_report(g, "b_mple_coefficients", coef(fb)))
        @test agree("b_mple_std_errors", stderror(fb))

        # (c) every degree 3 ~ edges + degree(1): all-zero degree1 column
        mc = network(8; directed=false)
        for (i, j) in zip(g.values["c_tails"], g.values["c_heads"]); add_edge!(mc, i, j); end
        fc = quiet(() -> fit_ergm(mc, [Edges(), Degree(1)]; method=:mple))
        @test fc.model.formula.terms.names == g.values["c_terms"]
        @test agree("c_mple_coefficients", coef(fc)) || error(golden_report(g, "c_mple_coefficients", coef(fc)))
        @test agree("c_mple_std_errors", stderror(fc))

        # (d) a statistic that never varies: NaN where R reports NA, with R's
        # "not varying" warning; the rest is R's fit
        flo = florentine_marriage()
        set_vertex_attribute!(flo, :z, zeros(16))
        fd = @test_logs (:warn, r"nodecov.z do not vary on the dyads fitted.*not varying") match_mode=:any fit_ergm(
            flo, [Edges(), NodeCov(:z)])
        @test fd.model.formula.terms.names == g.values["d_terms"]
        @test agree("d_mple_coefficients", coef(fd)) || error(golden_report(g, "d_mple_coefficients", coef(fd)))
        @test agree("d_mple_std_errors", stderror(fd))
        @test fd.converged && !is_exact(fd) && dof(fd) == 1
        @test any(occursin("nodecov.z not identifiable", a) for a in approximations(fd))
        # simulable: a NaN coefficient of a dyad-independent statistic is
        # any value, so the model is simulated with it at 0
        @test length(simulate_ergm(fd; n_sim=2, rng=Random.Xoshiro(1))) == 2
        e = try fit_ergm(flo, [Edges(), NodeCov(:z)]; se=:bootstrap, n_boot=3); nothing catch err; err end
        @test e isa ArgumentError && occursin("not identifiable", e.msg)

        # (e) the statnet tutorial's Goodreau model TYPED AS IN R:
        # nodematch(diff=TRUE) expands over every level, and the two levels
        # with no within-level tie (Black, Other) are fixed at -Inf by the
        # default fit, with R's message, instead of being refused
        fmh = load_dataset(:faux_mesa_high)
        G = [Edges(), NodeFactor(:Grade), NodeMatch(:Grade; diff=true), NodeFactor(:Race),
             NodeMatch(:Race; diff=true), NodeFactor(:Sex), NodeMatch(:Sex), GWESP(0.25),
             GWDegree(0.5)]
        fe = quiet(() -> fit_ergm(fmh, G; method=:mple))
        @test fe.model.formula.terms.names == g.values["e_terms"]
        @test agree("e_mple_coefficients", coef(fe)) || error(golden_report(g, "e_mple_coefficients", coef(fe)))
        @test agree("e_mple_std_errors_exact", stderror(fe)) ||
              error(golden_report(g, "e_mple_std_errors_exact", stderror(fe)))
        @test agree("e_mple_std_errors_as_shipped", stderror(fe))
        le = @test_logs (:warn, r"^mcmle: observed statistic\(s\) nodematch.Race.Black, nodematch.Race.Other are at their smallest attainable values") match_mode=:any fit_ergm(
            fmh, G; rng=Random.Xoshiro(7))
        @test le.method === :mcmle && le.converged
        want = Float64.(g.values["e_mle_coef_mean"])
        tol = Float64.(g.values["e_mle_tolerance"])
        @test isinf.(want) == isinf.(coef(le))
        @test all(isinf(w) ? c == w : abs(c - w) <= t for (c, w, t) in zip(coef(le), want, tol)) ||
              error("Goodreau MCMLE outside R's seed spread: " *
                    string([(n, round(c; digits=4), round(w; digits=4), round(t; digits=4))
                            for (n, c, w, t) in zip(coefnames(le), coef(le), want, tol)
                            if !(isinf(w) ? c == w : abs(c - w) <= t)]))
        @test isfinite(loglikelihood(le))           # dyad-independent drops keep the bridge
        @test dof(le) == length(coef(le)) - 2
        # drop=false: the strict mode refuses, naming both levels
        e = try fit_ergm(fmh, G; drop=false, rng=Random.Xoshiro(7)); nothing catch err; err end
        @test e isa ArgumentError && occursin("nodematch.Race.Black, nodematch.Race.Other", e.msg)
    end

    @testset "Refused, not mis-fit: two-mode networks and constraints" begin
        # Two-mode: within-mode dyads are structurally impossible, and used to
        # be counted as observations (nobs = 30 on 8 cross-mode dyads) and
        # toggled by the sampler.
        two_mode = network(6; bipartite=2)
        add_edge!(two_mode, 1, 3)
        add_edge!(two_mode, 2, 5)
        e = try ERGMModel(ERGMFormula([Edges(), Triangle()]), two_mode); nothing catch err; err end
        @test e isa ArgumentError
        @test occursin("bipartite", e.msg)
        @test occursin("one-mode", e.msg)
        @test_throws ArgumentError fit_ergm(two_mode, [Edges()])
        bn = BipartiteNetwork(2, 4)
        e2 = try ERGMModel(ERGMFormula([Edges()]), bn); nothing catch err; err end
        @test e2 isa ArgumentError
        @test occursin("bipartite", e2.msg)
        @test_throws ArgumentError fit_ergm(bn, [Edges()])

        # Constraints: accepted and silently ignored before; refused now
        e3 = try ERGMFormula([Edges()]; constraints=[FakeConstraint()]); nothing catch err; err end
        @test e3 isa ArgumentError
        @test occursin("constraints are not implemented", e3.msg)
        @test isempty(ERGMFormula([Edges()]; constraints=ERGM.ConstraintTerm[]).constraints)
    end


    # ------------------------------------------------------------------
    # MCMLE honesty and performance
    # ------------------------------------------------------------------
    @testset "MH kernel: mh_toggle! is the sampler, bit-identical to the pre-refactor loop" begin
        # (a) The binary-network adapter over the kernel reproduces the
        # hand-written loop it replaced EXACTLY. The literals below were
        # computed on the pre-refactor `_mh_run!` (on top of commit 72c2b86 "added bib citation", 2026-09-09) with
        # the same seeds; the kernel draws from the rng in the same order
        # (proposal, then the acceptance uniform), so the chains coincide.
        net = set_test_attrs!(fixture_undirected())
        model = ERGMModel(ERGMFormula([Edges(), NodeMatch(:group)]), net)
        θ = [-1.0, 0.5]
        # `proposal=:random` is that loop's proposal (a uniformly random
        # dyad); the default is now TNT, pinned separately below
        out = mh_sample(model, θ; n_samples=5, burnin=100, interval=10,
                        rng=Random.Xoshiro(1), proposal=:random)
        @test out.stats == [6.0 3.0; 5.0 4.0; 5.0 4.0; 3.0 2.0; 7.0 3.0]

        netd = network(6; directed=true)
        for (i, j) in [(1, 2), (2, 1), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3),
                       (1, 5), (5, 6), (6, 2)]
            add_edge!(netd, i, j)
        end
        md = ERGMModel(ERGMFormula([Edges(), Mutual(), Triangle()]), netd)
        outd = mh_sample(md, [-1.0, 0.8, 0.2]; n_samples=5, burnin=100,
                         interval=10, rng=Random.Xoshiro(1), proposal=:random)
        @test outd.stats == [17.0 7.0 33.0; 16.0 6.0 26.0; 13.0 5.0 14.0;
                             11.0 3.0 6.0; 7.0 1.0 1.0]

        # ... and through the multi-chain wrapper (per-chain seeds drawn from
        # the caller rng, chains concatenated in order)
        sims = sample_networks(model, θ; n_sim=6, burnin=200, interval=20,
                               rng=Random.Xoshiro(99), n_chains=3, proposal=:random)
        @test [ne(s) for s in sims] == [4, 7, 5, 6, 4, 7]

        # (b) The kernel is exported, documented, and usable on a state that
        # is not a network at all: a two-state toy chain with g(y) = y at
        # θ = log 3 has P(y = 1) = 3/4.
        @test Base.isexported(ERGM, :mh_toggle!)
        @test occursin("TERGM", string(@doc mh_toggle!))
        state = Ref(false)
        draws = Float64[]
        n_rec = mh_toggle!(Random.Xoshiro(1), [log(3.0)], [0.0],
                           rng -> 1,
                           (delta, move) -> (delta[1] = 1.0; state[]),
                           (move, removal) -> (state[] = !removal),
                           k -> push!(draws, state[]);
                           burnin=100, interval=1, n_samples=20_000)
        @test n_rec == 20_000 == length(draws)
        @test abs(mean(draws) - 0.75) < 0.02

        # (c) A variant-shaped adoption: a move is (layer, i, j) over two
        # independent edges-only layers with different densities — the
        # ERGMMulti sketch from the docstring. Each layer's edge count must
        # sit at its own logistic density.
        layers = [network(8; directed=false), network(8; directed=false)]
        θl = [-1.0, 1.0]                     # per-layer edges coefficients
        counts = zeros(2)
        n_rec2 = 0
        delta2 = zeros(2)
        propose = rng -> (rand(rng, 1:2), rand(rng, 1:7), rand(rng, 2:8))
        change! = (d, mv) -> begin
            l, i, j = mv
            fill!(d, 0.0)
            d[l] = 1.0
            has_edge(layers[l], i, j)
        end
        apply! = (mv, removal) -> begin
            l, i, j = mv
            removal ? rem_edge!(layers[l], i, j) : add_edge!(layers[l], i, j)
            nothing
        end
        on_sample = k -> (counts .+= ne.(layers); nothing)
        # propose can draw i == j; make such a move a no-op through change!
        # returning a zero delta and apply! being skipped by has_edge
        change2! = (d, mv) -> begin
            l, i, j = mv
            i == j ? (fill!(d, 0.0); true) : change!(d, mv)
        end
        apply2! = (mv, removal) -> begin
            l, i, j = mv
            i == j || apply!(mv, removal)
            nothing
        end
        n_rec2 = mh_toggle!(Random.Xoshiro(2), θl, delta2, propose, change2!, apply2!,
                            on_sample; burnin=5000, interval=20, n_samples=2000)
        expected = 28 .* (1 ./ (1 .+ exp.(-θl)))
        @test n_rec2 == 2000
        @test all(abs.(counts ./ 2000 .- expected) .< 1.5)

        # (d) Contract errors
        @test_throws ArgumentError mh_toggle!(Random.Xoshiro(1), [0.0], [0.0, 0.0],
                                              rng -> 1, (d, m) -> false, (m, r) -> nothing,
                                              k -> nothing; burnin=0, interval=1, n_samples=1)
        @test_throws ArgumentError mh_toggle!(Random.Xoshiro(1), [0.0], [0.0],
                                              rng -> 1, (d, m) -> false, (m, r) -> nothing,
                                              k -> nothing; burnin=-1, interval=1, n_samples=1)
        @test_throws ArgumentError mh_toggle!(Random.Xoshiro(1), [0.0], [0.0],
                                              rng -> 1, (d, m) -> false, (m, r) -> nothing,
                                              k -> nothing; burnin=0, interval=0, n_samples=1)
    end

    @testset "MH kernel is allocation-free per step" begin
        # The kernel itself adds 0 bytes per step: on a state whose callables
        # allocate nothing, the cost of a call does not grow with burn-in.
        state = Ref(false)
        kernel(b) = mh_toggle!(Random.Xoshiro(1), [log(3.0)], [0.0],
                               rng -> 1,
                               (delta, move) -> (delta[1] = 1.0; state[]),
                               (move, removal) -> (state[] = !removal),
                               k -> nothing;
                               burnin=b, interval=1, n_samples=1)
        kernel(100)
        @test (@allocated kernel(20_000)) == (@allocated kernel(10_000))

        # The binary-network adapter (`mh_sample`): everything ERGM.jl owns —
        # proposal, `change_stat_all!`, the closures, the sample write — adds
        # nothing per step. What remains, ~1 byte per step and decaying with
        # the number of toggles, is Graphs.jl growing the adjacency vectors
        # of a freshly copied network inside `add_edge!` (`insert!`); it is
        # identical under the pre-refactor loop and is not the kernel's.
        # Bounded here so a real per-step allocation (a boxed capture, a
        # tuple-allocating proposal, a per-term Vector) cannot hide in it.
        flo = florentine_marriage()
        model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo)
        θ = [-1.7, 0.1]
        mh_sample(model, θ; n_samples=1, burnin=100, interval=1, rng=Random.Xoshiro(1),
                  proposal=:random)
        a10 = @allocated mh_sample(model, θ; n_samples=1, burnin=10_000, interval=1,
                                   rng=Random.Xoshiro(1), proposal=:random)
        a20 = @allocated mh_sample(model, θ; n_samples=1, burnin=20_000, interval=1,
                                   rng=Random.Xoshiro(1), proposal=:random)
        @test a20 - a10 < 4 * 10_000
        # ... and the change statistics on the sampled states are exactly 0 B
        out = mh_sample(model, θ; n_samples=20, burnin=1000, interval=50,
                        rng=Random.Xoshiro(1), return_networks=true)
        delta = zeros(2)
        terms = model.formula.terms
        worst = 0
        for s in out.networks, i in 1:16, j in (i + 1):16
            ERGM.change_stat_all!(delta, terms, s, i, j)
            worst = max(worst, @allocated ERGM.change_stat_all!(delta, terms, s, i, j))
        end
        @test worst == 0

        # The same at 36 statistics. `Base.map` over the term tuple is unrolled
        # only below 32 elements; from 32 on it boxed every Float64 (~30 KB per
        # call — it was measured at 28,720 B per `change_stat_all!`
        # and per MH step at p = 32). The fills are generated per term count
        # now, so a NodeMix on an 8-level attribute (35 cells + edges = 36) is
        # as allocation-free as `edges + gwesp`.
        n = 60
        rng = Random.Xoshiro(36)
        big = network(n; directed=false)
        for i in 1:n, j in (i + 1):n
            rand(rng) < 0.08 && add_edge!(big, i, j)
        end
        set_vertex_attribute!(big, :g, Dict(v => "L$(mod1(v, 8))" for v in 1:n))
        wide = ERGMModel(ERGMFormula([Edges(), NodeMix(:g)]), big)
        wterms = wide.formula.terms
        @test length(wterms) == 36
        wdelta = zeros(36)
        ERGM.change_stat_all!(wdelta, wterms, big, 1, 2)
        @test (@allocated ERGM.change_stat_all!(wdelta, wterms, big, 1, 2)) == 0
        ERGM._change_stat_tuple(wterms, big, 1, 2)
        @test (@allocated ERGM._change_stat_tuple(wterms, big, 1, 2)) == 0
        @test change_stat_all(wterms, big, 1, 2) == wdelta
        @test compute_all(wterms, big) == [compute(t, big) for t in wterms.terms]
        θw = zeros(36)
        mh_sample(wide, θw; n_samples=1, burnin=100, interval=1, rng=Random.Xoshiro(1),
                  proposal=:random)
        w10 = @allocated mh_sample(wide, θw; n_samples=1, burnin=10_000, interval=1,
                                   rng=Random.Xoshiro(1), proposal=:random)
        w20 = @allocated mh_sample(wide, θw; n_samples=1, burnin=20_000, interval=1,
                                   rng=Random.Xoshiro(1), proposal=:random)
        @test w20 - w10 < 4 * 10_000       # the Graphs.jl growth only, as above
    end

    @testset "Hot paths are allocation-free" begin
        # The per-toggle change statistics — the innermost loop of both the
        # MPLE design and every MH proposal — on a 500-node sparse network,
        # as in benchmark/regression_tests.jl (which remains the standalone
        # runner the site's tools/run_benchmarks.jl consumes); here so that
        # `Pkg.test()` alone guards them.
        function er_network(rng, n, m; directed)
            g = network(n; directed=directed)
            while ne(g) < m
                i, j = rand(rng, 1:n), rand(rng, 1:n)
                i == j && continue
                add_edge!(g, i, j)
            end
            return g
        end
        n = 500
        net_u = er_network(Random.Xoshiro(1), n, 5n; directed=false)
        net_d = er_network(Random.Xoshiro(2), n, 10n; directed=true)
        rng = Random.Xoshiro(20260712)
        dyads = Tuple{Int,Int}[]
        while length(dyads) < 25
            i, j = rand(rng, 1:n), rand(rng, 1:n)
            i == j || push!(dyads, (i, j))
        end
        function worst_alloc(term, g)
            worst = 0
            for (i, j) in dyads
                change_stat(term, g, i, j)
                worst = max(worst, @allocated change_stat(term, g, i, j))
            end
            return worst
        end
        for term in [Edges(), Triangle(), GWESP(0.5), GWESP(0.0), GWDSP(0.5),
                     Kstar(2), TwoPath(), GWDegree(0.5), Degree(2),
                     GWNSP(0.5), Concurrent(), DegRange(1, 3), MeanDeg(), Density(),
                     ERGM.TriadCensus(2), ESP(0), ESP(2)]
            @test worst_alloc(term, net_u) == 0
        end
        for term in [Edges(), Mutual(), Triangle(), OStar(2), IStar(2),
                     GWESP(0.5), GWESP(0.5; type=:ITP), GWESP(0.5; type=:OSP),
                     GWDSP(0.5), GWDSP(0.5; type=:ISP),
                     GWIDegree(0.5), GWODegree(0.5), IDegree(2), ODegree(2),
                     GWNSP(0.5; type=:ITP), IDegRange(2), ODegRange(1, 3),
                     Sender(3), Receiver(4), ERGM.TriadCensus(9),
                     TransitiveTies(), CyclicalTies(), ESP(1), ESP(1; type=:ITP),
                     ESP(2; type=:OSP), ESP(0; type=:ISP)]
            @test worst_alloc(term, net_d) == 0
        end
    end

    @testset "MCMLE non-convergence is loud" begin
        flo = florentine_marriage()
        model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo)
        # Started far from the MLE (θ = 0: density 1/2 against 20/120
        # observed), one iteration can only take a partial Hummel step, so
        # the fit CANNOT have converged — deterministically, at any seed.
        fit = @test_logs (:warn, r"did not converge in maxiter=1 iterations") match_mode=:any mcmle(
            model; maxiter=1, n_samples=50, init=[0.0, 0.0], bridge_rungs=0,
            rng=Random.Xoshiro(1))
        @test !fit.converged
        md = fit_metadata(fit)
        @test any(occursin("did not converge", a) for a in md.approximations)
        # The verdict is the stopping rule's: under the default confidence
        # rule the caveat quotes its p-value and the step length, never the
        # classical t-ratio / Hotelling diagnostics (which are not the rule,
        # and on a converged fit describe the pre-step sample)
        caveat = only(a for a in md.approximations if occursin("did not converge", a))
        @test occursin("equivalence test p", caveat) && occursin("step length γ", caveat)
        @test !occursin("t-ratio", caveat) && !occursin("Hotelling", caveat)

        # The classical diagnostics are recomputed on the final sample at the
        # returned coefficients (an unconverged fit draws one there)
        c = fit.mcmc_convergence
        @test c isa ERGM.MCMLEConvergence
        @test c.iterations == 1
        @test 0 < c.step_length < 1          # never reached full step length
        @test length(c.t_ratios) == 2 && all(c.t_ratios .>= 0)
        @test 0 <= c.hotelling_p <= 1
        @test c.n_eff >= 2

        # `show` prints the caveat right under the verdict
        printed = sprint(show, fit)
        @test occursin("Converged: false\n  MCMLE did not converge (99% equivalence test p", printed)
        @test !occursin("t-ratio", printed)
        @test occursin("init=coef(fit)", printed)
        # Under the legacy rule the max t-ratio IS part of the verdict
        hfit = @test_logs (:warn, r"Hotelling p .*max t-ratio") match_mode=:any mcmle(
            model; maxiter=1, n_samples=50, init=[0.0, 0.0], bridge_rungs=0,
            termination=:hotelling, rng=Random.Xoshiro(1))
        @test occursin("max t-ratio", sprint(show, hfit))

        # ... and the advice is actionable: continue from where it stopped
        more = mcmle(model; maxiter=5, n_samples=400, init=coef(fit), bridge_rungs=0,
                     rng=Random.Xoshiro(2))
        @test more.mcmc_convergence.iterations >= 1
        @test norm(coef(more) .- coef(fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mple))) <
              norm(coef(fit) .- coef(fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mple)))
        @test_throws ArgumentError mcmle(model; init=[0.0], n_samples=10, maxiter=1)

        # MPLE fits carry no report and no MC term
        mp = fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mple)
        @test mp.mcmc_convergence === nothing
        @test mcmc_se(mp) == zeros(2)
        @test mp.vcov_fisher == vcov(mp)

        # The 5-seed fixture fits (asserted converged in the provenanced
        # testset) do not warn: a converged fit is silent
        quiet = @test_logs mcmle(model; n_samples=1024, rng=Random.Xoshiro(101),
                                 bridge_rungs=0)
        @test quiet.converged
        @test quiet.mcmc_convergence.step_length == 1.0
        # ... and its printed verdict is the termination test alone
        qp = sprint(show, quiet)
        @test occursin("Converged: true\n  Termination: 99% equivalence test p", qp)
        @test !occursin("t-ratio", qp) && !occursin("Hotelling", qp)
    end

    @testset "bridge_rungs=0 skips the log-likelihood, and only that" begin
        flo = florentine_marriage()
        model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo)
        f16 = mcmle(model; n_samples=200, maxiter=2, rng=Random.Xoshiro(5))
        f0 = mcmle(model; n_samples=200, maxiter=2, rng=Random.Xoshiro(5), bridge_rungs=0)
        # The bridge consumes randomness only AFTER the final sample, so
        # everything but the log-likelihood is bit-identical
        @test coef(f0) == coef(f16)
        @test stderror(f0) == stderror(f16)
        @test vcov(f0) == vcov(f16)
        @test f0.mcmc_samples == f16.mcmc_samples
        @test f0.converged == f16.converged
        @test isfinite(loglikelihood(f16)) && isfinite(aic(f16)) && isfinite(bic(f16))
        @test isnan(loglikelihood(f0)) && isnan(aic(f0)) && isnan(bic(f0))
        @test any(occursin("bridge_rungs=0", a) for a in fit_metadata(f0).approximations)
        @test !any(occursin("bridge_rungs=0", a) for a in fit_metadata(f16).approximations)
        printed = sprint(show, f0)
        @test occursin("Log-likelihood: not estimated (bridge_rungs=0)", printed)
        @test occursin("AIC: not estimated", printed)
        @test !occursin("NaN", printed)
        # Still a complete StatsAPI surface (NaN is a value, not a missing method)
        @test all(values(NetworkCore.check_statsapi(f0; strict=true)))
        @test_throws ArgumentError mcmle(model; bridge_rungs=-1, n_samples=10, maxiter=1)
        # The default is unchanged
        @test Base.kwarg_decl(first(methods(mcmle))) ⊇ [:bridge_rungs]
        @test isfinite(loglikelihood(mcmle(model; n_samples=100, maxiter=1,
                                           rng=Random.Xoshiro(1), bridge_samples=20)))
    end

    @testset "MCMLE n_chains: split chains agree and are thread-count independent" begin
        flo = florentine_marriage()
        model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo)
        # The `n_samples` split over `n_chains` is the fixed-size sampler's
        # (TNT, `effective_size=nothing`); the ESS-adaptive default sizes its
        # chains by its own rule and is pinned in "ESS-adaptive MCMLE ..."
        fixed = (proposal=:tnt, effective_size=nothing)
        one = mcmle(model; n_samples=2000, rng=Random.Xoshiro(7), bridge_rungs=0, fixed...)
        two = mcmle(model; n_samples=2000, rng=Random.Xoshiro(7), bridge_rungs=0, n_chains=2,
                    fixed...)
        @test one.converged && two.converged
        @test size(two.mcmc_samples) == (2000, 2)
        # Same estimating equation, different Monte-Carlo sample: agreement
        # within the fixture's MC width (0.03 on this model)
        @test maximum(abs.(coef(two) .- coef(one))) < 0.03
        @test maximum(abs.(stderror(two) .- stderror(one))) < 0.03
        # n_chains=1 is exactly the single-chain sampler
        @test coef(mcmle(model; n_samples=200, maxiter=2, rng=Random.Xoshiro(5),
                         bridge_rungs=0, n_chains=1)) ==
              coef(mcmle(model; n_samples=200, maxiter=2, rng=Random.Xoshiro(5),
                         bridge_rungs=0))
        # The chain-aware ESS: per-chain Geyer ESS summed, never above n
        @test 2 <= two.mcmc_convergence.n_eff <= 2000
        s, lengths = ERGM._mcmc_sample(model, coef(one), 10, 100, 5;
                                       rng=Random.Xoshiro(1), n_chains=3)
        @test lengths == [4, 3, 3] && size(s) == (10, 2)
        @test_throws ArgumentError mcmle(model; n_chains=0, n_samples=10, maxiter=1)

        # Thread-count independence, for real: the same n_chains=3 fit in a
        # fresh process with a DIFFERENT thread count is bit-identical
        # (chains are seeded from the caller's rng and concatenated in order;
        # `n_chains` never defaults to Threads.nthreads()).
        three = mcmle(model; n_samples=300, rng=Random.Xoshiro(3), bridge_rungs=0,
                      n_chains=3, maxiter=2)
        other_threads = Threads.nthreads() == 1 ? 4 : 1
        script = """
            using ERGM, Random
            flo = load_dataset(:florentine_marriage)
            m = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo)
            f = mcmle(m; n_samples=300, rng=Xoshiro(3), bridge_rungs=0, n_chains=3, maxiter=2)
            println(repr(coef(f))); println(repr(stderror(f)))
            """
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$other_threads --project=$(dirname(@__DIR__)) -e $script`
        lines = split(strip(read(pipeline(cmd; stderr=devnull), String)), '\n')
        @test length(lines) == 2
        @test lines[1] == repr(coef(three))
        @test lines[2] == repr(stderror(three))
    end

    @testset "MCMC-error component of the MCMLE standard errors" begin
        flo = florentine_marriage()
        model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo)
        # Fixed-size TNT sampling (`effective_size=nothing`): the Monte-Carlo
        # share is compared across SAMPLE SIZES, which the ESS-adaptive
        # default would choose itself
        fixed = (proposal=:tnt, effective_size=nothing)
        big = mcmle(model; n_samples=4096, rng=Random.Xoshiro(101), bridge_rungs=0, fixed...)
        # (the sample size is held fixed: the confidence rule would boost it)
        small = mcmle(model; n_samples=40, max_n_samples=40, rng=Random.Xoshiro(101),
                      bridge_rungs=0, fixed...)
        share(f) = 100 .* (mcmc_se(f) ./ stderror(f)) .^ 2
        # At the fixture budget the MC variance share is well under 1 % (R
        # reports 0 % on the same model; the variance share is 0.3–0.4 %), and
        # it grows as the sample shrinks
        @test all(share(big) .< 1.0)
        @test all(share(small) .> share(big))
        # R's "MCMC %" is NOT the variance share: `ergm:::summary.ergm`
        # computes `round(100 * (tot.se - mod.se) / tot.se)`, the share of the
        # standard error itself ([1,1] R-style vs [2,1] for
        # the variance share on this model at n_samples=150)
        r_pct(f) = round.(Int, 100 .* (stderror(f) .- sqrt.(diag(f.vcov_fisher))) ./ stderror(f))
        @test ERGM._mcmc_percent(big) == r_pct(big) == [0, 0]
        @test ERGM._mcmc_percent(small) == r_pct(small)
        @test all(ERGM._mcmc_percent(small) .<= round.(Int, share(small)))
        mid = mcmle(model; n_samples=150, rng=Random.Xoshiro(101), bridge_rungs=0, fixed...)
        @test ERGM._mcmc_percent(mid) == r_pct(mid)
        @test all(x -> x isa Int, ERGM._mcmc_percent(big))
        # The two definitions are different numbers: se = 1 with se_fisher =
        # 0.6 (mcmc_se = 0.8) is 40 % of the standard error (R) and 64 % of
        # its variance
        synthetic = ERGMResult(big.model, big.coefficients, [1.0, 1.0], big.z_values,
                               big.p_values, Matrix(1.0I, 2, 2), NaN, NaN, NaN, :mcmle,
                               true, big.mcmc_samples, :mcmc, :none,
                               Matrix(0.36I, 2, 2), [0.8, 0.8], big.mcmc_convergence,
                               big.chain_lengths, nothing, big.termination)
        @test ERGM._mcmc_percent(synthetic) == [40, 40]
        @test round.(Int, share(synthetic)) == [64, 64]
        @test occursin("MCMC % of the standard error (100·(se − se_fisher)/se): edges 40, gwesp.fixed.0.5 40",
                       sprint(show, synthetic))
        nanres = ERGMResult(big.model, big.coefficients, [NaN, 1.0], big.z_values,
                            big.p_values, Matrix(1.0I, 2, 2), NaN, NaN, NaN, :mcmle,
                            true, big.mcmc_samples, :mcmc, :none,
                            Matrix(0.36I, 2, 2), [0.8, 0.8], big.mcmc_convergence,
                            big.chain_lengths, nothing, big.termination)
        @test isnan(ERGM._mcmc_percent(nanres)[1]) && ERGM._mcmc_percent(nanres)[2] == 40
        # vcov = V_fisher + V_fisher Σ_mc V_fisher, symmetric positive definite,
        # and the Fisher part is what `vcov` used to be
        @test issymmetric(vcov(big)) && isposdef(vcov(big))
        @test issymmetric(big.vcov_fisher) && isposdef(big.vcov_fisher)
        @test vcov(big) ≈ big.vcov_fisher .+ (vcov(big) .- big.vcov_fisher)
        @test all(diag(vcov(big)) .>= diag(big.vcov_fisher))
        @test stderror(big) ≈ sqrt.(diag(big.vcov_fisher) .+ mcmc_se(big) .^ 2)
        @test Base.isexported(ERGM, :mcmc_se)
        @test se_method(big) == :fisher
        printed = sprint(show, big)
        @test occursin("MCMC % of the standard error (100·(se − se_fisher)/se): edges 0, gwesp.fixed.0.5 0", printed)
        @test occursin(r"MCMC % of the standard error \(100·\(se − se_fisher\)/se\): edges [1-9]", sprint(show, small))
        # The Monte-Carlo covariance of the mean is the Geyer variance per
        # statistic over n on an iid sample, and adds across chains
        rng = Random.Xoshiro(1)
        iid = randn(rng, 4000, 2) .* [1.0 2.0]
        Σ = ERGM._mc_cov_of_mean(iid)
        @test Σ[1, 1] ≈ 1.0 / 4000 rtol = 0.15
        @test Σ[2, 2] ≈ 4.0 / 4000 rtol = 0.15
        @test abs(Σ[1, 2]) < 0.2 * sqrt(Σ[1, 1] * Σ[2, 2])
        Σ2 = ERGM._mc_cov_of_mean(iid, [2000, 2000])
        @test Σ2 ≈ Σ rtol = 0.2
        # An AR(1) chain has asymptotic variance (1+ρ)/(1−ρ) times the iid one
        ρ = 0.8
        ar = zeros(20_000)
        for t in 2:length(ar)
            ar[t] = ρ * ar[t - 1] + randn(rng)
        end
        Σar = ERGM._mc_cov_of_mean(reshape(ar, :, 1))
        @test Σar[1, 1] * length(ar) ≈ (1 + ρ) / (1 - ρ) * var(ar) rtol = 0.25
    end

    @testset "mcmc_convergence on synthetic chains" begin
        @test Base.ispublic(ERGM, :mcmc_convergence)
        rng = Random.Xoshiro(1)
        draws = randn(rng, 2000, 2) .+ [1.0 5.0]
        ok = ERGM.mcmc_convergence(draws, [1.0, 5.0])
        @test ok.converged
        @test all(ok.t_ratios .< 0.1)
        @test 0.05 < ok.hotelling_p <= 1
        @test 1000 < ok.n_eff <= 2000
        off = ERGM.mcmc_convergence(draws, [3.0, 5.0])
        @test !off.converged
        @test off.t_ratios[1] > 1.5
        @test off.hotelling_p < 0.05
        # Both gates are separate: a target 0.15 sd off passes a loosened
        # t-ratio threshold but not the Hotelling test (with n = 2000 the mean
        # is known to ~0.02 sd), and the true target passes Hotelling but not
        # an absurdly strict t-ratio threshold
        near = ERGM.mcmc_convergence(draws, [1.0, 5.15]; conv_threshold=0.25)
        @test !near.converged && maximum(near.t_ratios) < 0.25 && near.hotelling_p < 0.05
        strict = ERGM.mcmc_convergence(draws, [1.0, 5.0]; conv_threshold=0.01)
        @test !strict.converged && strict.hotelling_p > 0.05
        # Chains add information: ESS over two independent halves ≥ one chain
        two = ERGM.mcmc_convergence(draws, [1.0, 5.0]; chain_lengths=[1000, 1000])
        @test two.converged
        @test two.n_eff > 1000
        # An autocorrelated chain has a smaller ESS and a less confident test
        ar = zeros(2000, 1)
        for t in 2:2000
            ar[t, 1] = 0.9 * ar[t - 1, 1] + randn(rng)
        end
        arc = ERGM.mcmc_convergence(ar, [0.0])
        @test arc.n_eff < 400
        # Degenerate statistic: t-ratio Inf, never converged
        deg = ERGM.mcmc_convergence(hcat(draws[:, 1], fill(2.0, 2000)), [1.0, 2.0])
        @test !deg.converged && deg.t_ratios[2] == Inf
        @test_throws ArgumentError ERGM.mcmc_convergence(draws, [1.0])
        @test_throws ArgumentError ERGM.mcmc_convergence(draws, [1.0, 5.0]; chain_lengths=[1000])
        # `mcmle` itself records the same tests on its final sample
        flo = florentine_marriage()
        fit = mcmle(ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo);
                    n_samples=500, rng=Random.Xoshiro(3), bridge_rungs=0)
        again = ERGM.mcmc_convergence(fit.mcmc_samples,
                                      compute_all(fit.model.formula.terms, flo))
        @test again.t_ratios == fit.mcmc_convergence.t_ratios
        @test again.hotelling_p == fit.mcmc_convergence.hotelling_p
        @test again.n_eff == fit.mcmc_convergence.n_eff
    end

    @testset "Dyad-scaled sampler defaults everywhere" begin
        flo = florentine_marriage()
        model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo)
        # One rule
        @test ERGM.Extension.mcmc_defaults(model) == (burnin = 2400, interval = 100)
        @test ERGM.Extension.mcmc_defaults(124750) == (burnin = 2495000, interval = 12475)
        @test Base.isexported(ERGM.Extension, :mcmc_defaults)   # the extension API, not an `ERGM` name
        # Every sampler still takes burnin/interval by name (the vocabulary is
        # unchanged; only the default moved from a literal to `nothing`)
        for f in (mh_sample, simulate_ergm, sample_networks)
            kw = Base.kwarg_decl(first(methods(f)))
            @test :burnin in kw && :interval in kw
        end
        gof_kw = Base.kwarg_decl(only(m for m in methods(gof) if m.module === ERGM))
        @test :burnin in gof_kw && :interval in gof_kw
        # The default resolves to the rule: identical to passing it explicitly
        θ = [-1.7, 0.1]
        @test mh_sample(model, θ; n_samples=5, rng=Random.Xoshiro(1)).stats ==
              mh_sample(model, θ; n_samples=5, burnin=2400, interval=100,
                        rng=Random.Xoshiro(1)).stats
        @test [as_matrix(s) for s in sample_networks(model, θ; n_sim=3, rng=Random.Xoshiro(2))] ==
              [as_matrix(s) for s in sample_networks(model, θ; n_sim=3, burnin=2400,
                                                     interval=100, rng=Random.Xoshiro(2))]
        fit = fit_ergm(flo, [Edges()])
        g = gof(fit; n_sim=12, stats=[:degree], rng=Random.Xoshiro(3))
        g_explicit = gof(fit; n_sim=12, stats=[:degree], burnin=2400, interval=100,
                         rng=Random.Xoshiro(3))
        @test n_simulations(g) == 12
        @test g.statistics[1].simulated == g_explicit.statistics[1].simulated
        # ... and an explicit value is honoured (a different budget, a different draw)
        g_other = gof(fit; n_sim=12, stats=[:degree], burnin=300, interval=30,
                      rng=Random.Xoshiro(3))
        @test g_other.statistics[1].simulated != g.statistics[1].simulated
        # The MPLE bootstrap uses the same rule
        boot = fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mple, se=:bootstrap, n_boot=4,
                        rng=Random.Xoshiro(4))
        boot_explicit = fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mple, se=:bootstrap, n_boot=4,
                                 boot_burnin=2400, boot_interval=100, rng=Random.Xoshiro(4))
        @test stderror(boot) == stderror(boot_explicit)
    end

    @testset "MPLE design build allocates O(unique rows)" begin
        # `_mple_data` used to allocate a fresh Vector{Float64} per dyad as the
        # Dict key; it now keys on an NTuple built on
        # the stack, so the per-dyad sweep allocates NOTHING and the whole build
        # allocates only the row table (its Dict growth) and the returned
        # arrays. Measured on a 200-node ER network: edges+gwesp+nodecov
        # (19900 unique rows) 4.57 MB → 4.38 MB (≈220 B per unique row, all of
        # it the table and the returned X/n_tot/n_one); edges+nodematch
        # (2 unique rows over 19900 dyads) 1.59 MB → 1.07 KB. The
        # compressed-row semantics are unchanged (the provenanced fixture
        # asserts the numbers).
        rng = Random.Xoshiro(2026)
        n = 100
        net = network(n; directed=false)
        for i in 1:n, j in (i+1):n
            rand(rng) < 0.06 && add_edge!(net, i, j)
        end
        set_vertex_attribute!(net, :x, Dict(v => randn(rng) for v in 1:n))
        set_vertex_attribute!(net, :g, Dict(v => (isodd(v) ? "a" : "b") for v in 1:n))
        m = ERGMModel(ERGMFormula([Edges(), GWESP(0.5), NodeCov(:x)]), net)
        ts = m.formula.terms
        X, n_tot, n_one = ERGM._mple_data(net, ts, false)     # warm up
        bytes = @allocated ERGM._mple_data(net, ts, false)
        n_rows = size(X, 1)
        @test sum(n_tot) == n * (n - 1) ÷ 2
        @test sum(n_one) == ne(net)
        @test bytes < n_rows * 256            # the table + returned arrays only

        # The sweep itself is allocation-free: with the row table pre-sized,
        # not one byte per dyad
        rows = Dict{NTuple{3, Float64}, Tuple{Float64, Float64}}()
        sizehint!(rows, n_rows)
        ERGM._mple_rows!(rows, net, ts, false)
        empty!(rows); sizehint!(rows, n_rows)
        @test (@allocated ERGM._mple_rows!(rows, net, ts, false)) == 0

        # Heavily compressed design: O(unique rows), not O(dyads)
        ts2 = ERGMModel(ERGMFormula([Edges(), NodeMatch(:g)]), net).formula.terms
        X2, = ERGM._mple_data(net, ts2, false)
        @test size(X2, 1) == 2
        @test (@allocated ERGM._mple_data(net, ts2, false)) < 4096

        # 36 statistics (edges + a NodeMix on an 8-level attribute): the key
        # tuple is still built on the stack — `Base.map` over ≥ 32 terms
        # boxed every value (it measured 201 MB for a 120-node sweep
        # at p = 32) — so the sweep allocates nothing and the build is
        # O(unique rows × p)
        set_vertex_attribute!(net, :g8, Dict(v => "L$(mod1(v, 8))" for v in 1:n))
        ts36 = ERGMModel(ERGMFormula([Edges(), NodeMix(:g8)]), net).formula.terms
        @test length(ts36) == 36
        X36, n_tot36, n_one36 = ERGM._mple_data(net, ts36, false)
        @test size(X36, 1) == 36                       # one row per mixing cell
        @test sum(n_tot36) == n * (n - 1) ÷ 2
        b36 = @allocated ERGM._mple_data(net, ts36, false)
        @test b36 < size(X36, 1) * 48 * 36
        rows36 = Dict{NTuple{36, Float64}, Tuple{Float64, Float64}}()
        sizehint!(rows36, 64)
        ERGM._mple_rows!(rows36, net, ts36, false)
        empty!(rows36); sizehint!(rows36, 64)
        @test (@allocated ERGM._mple_rows!(rows36, net, ts36, false)) == 0

        # Compressed rows: a row-per-dyad design gives the same fit
        Xd = Vector{Vector{Float64}}(); yd = Float64[]
        for i in 1:n, j in (i+1):n
            push!(Xd, change_stat_all(ts, net, i, j))
            push!(yd, has_edge(net, i, j) ? 1.0 : 0.0)
        end
        ref = newton_fit(logistic_derivatives(permutedims(reduce(hcat, Xd)),
                                              Bool.(yd)), zeros(3))
        @test coef(mple(m)) ≈ ref.θ atol = 1e-8
    end


    # ------------------------------------------------------------------
    # Missing-data maximum likelihood
    # ------------------------------------------------------------------
    @testset "Constrained sampler: toggleable dyad sets" begin
        flo = florentine_marriage()
        masked = copy(flo)
        for (i, j) in ((3, 4), (1, 9), (7, 16), (2, 11))
            set_missing_dyad!(masked, i, j)
        end
        model = ERGMModel(ERGMFormula([Edges(), Triangle()]), masked)
        θ = [-1.5, 0.3]
        observed = [(i, j) for i in 1:16 for j in (i+1):16
                    if !is_missing_dyad(masked, i, j)]

        # :masked — every observed dyad keeps its observed value in every
        # sampled network; the masked dyads do move (both states are visited)
        cons = mh_sample(model, θ; n_samples=200, toggleable=:masked,
                         rng=Random.Xoshiro(3), return_networks=true)
        @test size(cons.stats) == (200, 2)
        @test all(has_edge(s, i, j) == has_edge(masked, i, j)
                  for s in cons.networks for (i, j) in observed)
        for (i, j) in ((3, 4), (1, 9), (7, 16), (2, 11))
            states = [has_edge(s, i, j) for s in cons.networks]
            @test any(states) && !all(states)
            @test all(is_missing_dyad(s, i, j) for s in cons.networks)  # mask kept
        end
        # Under dyad independence the constrained mean is the observed ties
        # plus the model expectation over the masked dyads
        em = ERGMModel(ERGMFormula([Edges()]), masked)
        c1 = mh_sample(em, [-1.0]; n_samples=4000, toggleable=:masked,
                       burnin=200, interval=5, rng=Random.Xoshiro(4))
        @test mean(c1.stats) ≈ 18 + 4 / (1 + exp(1.0)) atol = 0.1

        # :all — the masked dyads are toggled too (the unconditional model);
        # under θ = -6 an edges-only chain empties the masked ties as well
        allc = mh_sample(em, [-6.0]; n_samples=50, toggleable=:all,
                         burnin=3000, interval=20, rng=Random.Xoshiro(5),
                         return_networks=true)
        @test any(!has_edge(s, 1, 9) for s in allc.networks)
        @test any(!has_edge(s, 7, 16) for s in allc.networks)
        # ... whereas :free (with the opt-in) freezes them
        freec = mh_sample(em, [-6.0]; n_samples=50, toggleable=:free,
                          burnin=3000, interval=20, rng=Random.Xoshiro(5),
                          return_networks=true, missing=:condition_on_face)
        @test all(has_edge(s, 1, 9) && has_edge(s, 7, 16) for s in freec.networks)

        # The dyad-set sizes and the dyad-scaled defaults per chain
        @test ERGM._n_toggleable(masked, :free) == 116
        @test ERGM._n_toggleable(masked, :all) == 120
        @test ERGM._n_toggleable(masked, :masked) == 4
        @test ERGM._resolve_mcmc_controls(model, nothing, nothing; toggleable=:masked) ==
              (80, 100)
        @test ERGM._resolve_mcmc_controls(model, nothing, nothing) ==
              (20 * 116, max(100, 116 ÷ 10))

        # Refusals: :masked needs a mask, the set must be one of the three,
        # and the missing-data chains take no `missing=` opt-in
        clean = ERGMModel(ERGMFormula([Edges()]), flo)
        e = try mh_sample(clean, [0.0]; n_samples=2, toggleable=:masked) catch err; err end
        @test e isa ArgumentError && occursin("toggleable=:masked", e.msg)
        @test_throws ArgumentError mh_sample(clean, [0.0]; n_samples=2, toggleable=:some)
        e = try mh_sample(em, [0.0]; n_samples=2, toggleable=:all,
                          missing=:condition_on_face) catch err; err end
        @test e isa ArgumentError && occursin("only meaningful for toggleable=:free", e.msg)
        @test_throws ArgumentError ERGM._mh_run!(Random.Xoshiro(1), copy(masked),
                                                 model.formula.terms, θ,
                                                 1, 0, 1, false, :bogus)

        # `_random_network`: masked dyads keep their face value by default
        # and are randomized on request (the mask itself survives both)
        rng = Random.Xoshiro(6)
        kept = [(has_edge(ERGM._random_network(masked; density=0.05, rng=rng), 1, 9),
                 has_edge(ERGM._random_network(masked; density=0.05, rng=rng), 3, 4))
                for _ in 1:20]
        @test all(k == (true, false) for k in kept)
        randomized = [ERGM._random_network(masked; density=0.5, rng=rng,
                                           randomize_masked=true) for _ in 1:40]
        @test any(!has_edge(s, 1, 9) for s in randomized)
        @test any(has_edge(s, 3, 4) for s in randomized)
        @test all(n_missing_dyads(s) == 4 for s in randomized)
    end

    @testset "Dyad-independent log-normalizers over the three dyad sets" begin
        # Exhaustive enumeration on a 5-vertex network (10 dyads, 2 masked)
        # for a dyad-independent model: log Z over all dyads, over the free
        # dyads with the masked ones fixed at face value, and over the masked
        # dyads with the observed ones fixed — the reference normalizers of
        # the three bridges.
        net = network(5; directed=false)
        for (i, j) in ((1, 2), (2, 3), (3, 4), (1, 5))
            add_edge!(net, i, j)
        end
        set_vertex_attribute!(net, :g, Dict(1 => "a", 2 => "a", 3 => "b", 4 => "b", 5 => "a"))
        set_missing_dyad!(net, 1, 2)      # masked, face value: tie
        set_missing_dyad!(net, 4, 5)      # masked, face value: no tie
        model = ERGMModel(ERGMFormula([Edges(), NodeMatch(:g)]), net)
        θ = [-0.7, 0.9]
        dyads = [(i, j) for i in 1:5 for j in (i+1):5]
        masked = [(1, 2), (4, 5)]
        free = setdiff(dyads, masked)
        function logZ_enum(varying, fixed_from)
            total = -Inf
            for bits in 0:(2^length(varying) - 1)
                y = copy(fixed_from)
                for (k, (i, j)) in enumerate(varying)
                    has_edge(y, i, j) && rem_edge!(y, i, j)
                    (bits >> (k - 1)) & 1 == 1 && add_edge!(y, i, j)
                end
                s = dot(θ, compute_all(model.formula.terms, y))
                total = max(total, s) + log1p(exp(-abs(total - s)))
            end
            total
        end
        @test ERGM._dyad_independent_logZ(model, θ; toggleable=:all) ≈
              logZ_enum(dyads, net) atol = 1e-10
        @test ERGM._dyad_independent_logZ(model, θ; toggleable=:free) ≈
              logZ_enum(free, net) atol = 1e-10
        @test ERGM._dyad_independent_logZ(model, θ; toggleable=:masked) ≈
              logZ_enum(masked, net) atol = 1e-10
        @test ERGM._dyad_independent_logZ(model, θ) ==
              ERGM._dyad_independent_logZ(model, θ; toggleable=:free)
        # Without a mask the three coincide
        clean = ERGMModel(ERGMFormula([Edges(), NodeMatch(:g)]), florentine_marriage() |>
                          n -> (set_vertex_attribute!(n, :g, Dict(v => (isodd(v) ? "a" : "b") for v in 1:16)); n))
        za = ERGM._dyad_independent_logZ(clean, θ; toggleable=:all)
        @test za == ERGM._dyad_independent_logZ(clean, θ; toggleable=:free)
        @test ERGM._dyad_independent_logZ(clean, θ; toggleable=:masked) ≈
              dot(θ, compute_all(clean.formula.terms, clean.network)) atol = 1e-12
        # The bridge of a dyad-independent model is exact, and the missing-data
        # log-likelihood log Z_obs − log Z is the observed-dyad logistic one
        ll = ERGM._bridge_logZ(model, θ; toggleable=:masked, nrungs=2, n_samples=10,
                               burnin=1, interval=1) -
             ERGM._bridge_logZ(model, θ; toggleable=:all, nrungs=2, n_samples=10,
                               burnin=1, interval=1)
        X, n_tot, n_one = ERGM._mple_data(net, model.formula.terms, false)
        η = X * θ
        @test ll ≈ sum(n_one .* η .- n_tot .* log1p.(exp.(η))) atol = 1e-10
    end

    @testset "mcmc_convergence against a Monte-Carlo target" begin
        rng = Random.Xoshiro(7)
        free = randn(rng, 3000, 2) .* [2.0 1.0] .+ [1.0 5.0]
        cons = randn(rng, 3000, 2) .* [0.5 0.5] .+ [1.0 5.0]
        tgt = vec(mean(cons, dims=1))
        ok = ERGM.mcmc_convergence(free, tgt; target_samples=cons)
        @test ok.converged
        # The per-draw sd behind the t-ratios includes both chains
        @test ok.t_ratios ≈ abs.(tgt .- vec(mean(free, dims=1))) ./
                             sqrt.(vec(var(free, dims=1)) .+ vec(var(cons, dims=1)))
        far = cons .+ [1.5 0.0]
        bad = ERGM.mcmc_convergence(free, vec(mean(far, dims=1)); target_samples=far)
        @test !bad.converged && bad.t_ratios[1] > 0.5
        # Two chains on the target side
        two = ERGM.mcmc_convergence(free, tgt; target_samples=cons,
                                    target_chain_lengths=[1500, 1500])
        @test two.converged
        @test_throws ArgumentError ERGM.mcmc_convergence(free, tgt; target_samples=cons[:, 1:1])
        @test_throws ArgumentError ERGM.mcmc_convergence(free, tgt; target_samples=cons,
                                                         target_chain_lengths=[1000, 1000])
        # The Fisher-information helper refuses a non-PD Σ_free − Σ_obs
        nan = @test_logs (:warn, r"not positive definite") ERGM.Extension.mcmle_covariance(
            cons, [3000], free, [3000], 2)
        @test all(isnan, nan[3])
        fine = ERGM.Extension.mcmle_covariance(free, [3000], cons, [3000], 2)
        @test all(isfinite, fine[3])
        @test fine[1] ≈ inv(cov(free) .- cov(cons)) atol = 1e-10
    end

    @testset "Missing-data MCMLE (missing=:mle)" begin
        flo = florentine_marriage()
        masked = copy(flo)
        for (i, j) in ((3, 4), (1, 9), (7, 16), (2, 11))
            set_missing_dyad!(masked, i, j)
        end

        # (1) Dyad independence: the missing-data MLE IS the logistic MLE of
        # the observed dyads — the available-case MPLE — so the constrained
        # chain's mean (observed ties + model expectation over the masked
        # dyads) equals the free chain's mean at the MPLE to within MC error,
        # the convergence tests pass there and the estimate is the MPLE. An
        # R-free internal consistency check.
        model_di = ERGMModel(ERGMFormula([Edges(), NodeCov(:wealth)]), masked)
        ac = mple(model_di)
        @test ac.missing_method === :available_case
        f_di = mcmle(model_di; missing=:mle, n_samples=1000, rng=Random.Xoshiro(1))
        @test f_di.converged
        @test f_di.missing_method === :mle
        @test f_di.coefficients ≈ ac.coefficients atol = 1e-6
        # The Fisher information Σ_free − Σ_obs is the observed-dyad logistic
        # information, so the SEs agree with the MPLE's up to MC error
        @test all(abs.(f_di.std_errors ./ ac.std_errors .- 1) .< 0.1)
        # ... and the two bridges are exact under dyad independence: the
        # log-likelihood is the observed-dyad logistic log-likelihood
        @test f_di.loglik ≈ ac.loglik atol = 1e-8
        @test nobs(f_di) == 116

        # (4) No mask: `missing=:mle` is a no-op, bit-identical to `:error`
        clean = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo)
        a = mcmle(clean; n_samples=300, maxiter=3, bridge_rungs=2, bridge_samples=50,
                  rng=Random.Xoshiro(2))
        b = mcmle(clean; n_samples=300, maxiter=3, bridge_rungs=2, bridge_samples=50,
                  rng=Random.Xoshiro(2), missing=:mle)
        @test a.coefficients == b.coefficients
        @test a.std_errors == b.std_errors
        @test a.loglik == b.loglik
        @test b.missing_method === :none

        # (5)/(6) A dyad-dependent fit on the masked network: metadata, show,
        # approximations, n_chains, the constrained-chain controls
        model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), masked)
        fit = mcmle(model; missing=:mle, n_samples=1000, rng=Random.Xoshiro(3),
                    bridge_rungs=4, bridge_samples=200)
        @test fit.converged
        @test fit_metadata(fit).missing_method == :mle
        @test missing_method(fit) === :mle
        @test nobs(fit) == 116
        @test isfinite(fit.loglik)
        @test all(fit.mcmc_se .> 0)
        @test all(fit.std_errors .> fit.mcmc_se)
        @test length(fit.mcmc_convergence.t_ratios) == 2
        s = sprint(show, fit)
        @test occursin("Missing dyads: 4 masked; missing-data maximum likelihood " *
                       "(masked dyads integrated out by a constrained chain)", s)
        @test any(occursin("constrained chain", a) for a in approximations(fit))
        @test !any(occursin("held fixed", a) for a in approximations(fit))
        fit2 = mcmle(model; missing=:mle, n_samples=1000, rng=Random.Xoshiro(3),
                     bridge_rungs=0, n_chains=2, obs_burnin=50, obs_interval=10)
        @test fit2.converged
        @test isnan(fit2.loglik)
        @test abs(fit2.coefficients[1] - fit.coefficients[1]) < 0.1
        @test supports_missing(mcmle)
        @test missing_policies(mcmle) == (:error, :condition_on_face, :mle)
        @test_throws ArgumentError mcmle(model; missing=:face, n_samples=50)

        # (5) The samplers refuse `missing=:mle` with an explanation: a
        # simulated network has a value at every dyad
        for refuse in (() -> simulate_ergm(fit; n_sim=1, missing=:mle),
                       () -> gof(fit; n_sim=1, missing=:mle),
                       () -> sample_networks(model, fit.coefficients; n_sim=1, missing=:mle),
                       () -> mh_sample(model, fit.coefficients; n_samples=1, missing=:mle))
            e = try refuse(); nothing catch err; err end
            @test e isa ArgumentError
            @test occursin("simulation under missing data is not defined", e.msg)
            @test occursin("mcmle(model; missing=:mle)", e.msg)
        end
        # ... but the ordinary opt-ins still work on an :mle fit
        sims = @test_logs (:warn, r"conditions on them at their face value") match_mode=:any simulate_ergm(
            fit; n_sim=2, burnin=100, interval=10, rng=Random.Xoshiro(4),
            missing=:condition_on_face)
        @test length(sims) == 2
    end

    # ------------------------------------------------------------------
    # Golden fixture: statnet ergm on a MASKED flomarriage, with provenance
    # (test/fixtures/flomarriage_missing_ergm.toml, generated by
    # test/fixtures/r/flomarriage_missing_ergm.R). Four dyads set to NA: two
    # ties, two non-ties. It pins all three of ERGM.jl's treatments against
    # what R actually does with NA ties:
    #   (a) summary statistics — R treats every NA dyad as ABSENT;
    #   (b) the available-case MPLE — R's dyad-independent fit with NA
    #       dyads IS the logistic regression on the observed dyads (1e-6);
    #   (c) `missing=:mle` — R's missing-data MCMLE, compared against R's
    #       own measured seed-to-seed spread.
    # ------------------------------------------------------------------
    @testset "Golden fixture: statnet ergm on masked flomarriage (provenanced)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "flomarriage_missing_ergm.toml"))
        @test g.provenance["ergm_version"] == "4.12.0"
        @test g.provenance["masked_dyads"] == "(3,4), (1,9), (7,16), (2,11)"
        flo = florentine_marriage()
        masked = copy(flo)
        for (i, j) in ((3, 4), (1, 9), (7, 16), (2, 11))
            set_missing_dyad!(masked, i, j)
        end
        @test n_missing_dyads(masked) == 4

        # --- (a) summary statistics: R's NA-as-absent convention ---------
        # statnet's summary() evaluates the statistics with every NA dyad
        # treated as absent (18 edges, not 20). ERGM.jl's `compute` reads
        # the stored face value — a masked tie is still a tie in the graph —
        # so R's numbers are reproduced on a copy with the masked ties
        # removed. Both facts are asserted so neither convention is silent.
        @test compute(Edges(), masked) == 20.0               # face value
        absent = copy(masked)
        for (i, j) in missing_dyads(absent)
            has_edge(absent, i, j) && rem_edge!(absent, i, j)
        end
        @test g.values["summary_statistic_names"] ==
              ["edges", "nodecov.wealth", "gwesp.fixed.0.5"]
        stats = [compute(Edges(), absent), compute(NodeCov(:wealth), absent),
                 compute(GWESP(0.5), absent)]
        @test check_golden(g, "summary_statistics", stats) ||
              error(golden_report(g, "summary_statistics", stats))
        @test g.values["n_observed_dyads"] == 116
        # `summary_stats` is the statnet `summary()` counterpart and honours
        # the missing-data contract: it refuses the masked network, and under
        # the explicit `missing=:face` opt-in reproduces R's numbers on the
        # absent-copy (and the face values on the masked network itself)
        sum_terms = [Edges(), NodeCov(:wealth), GWESP(0.5)]
        @test_throws ArgumentError summary_stats(masked, sum_terms)
        ss = summary_stats(absent, sum_terms; missing=:face)
        @test String.(collect(keys(ss))) == g.values["summary_statistic_names"]
        @test check_golden(g, "summary_statistics", collect(values(ss)))
        @test summary_stats(masked, [Edges()]; missing=:face).edges == 20.0

        # --- (b) available-case MPLE == R's fit with NA dyads -------------
        # The first R pin of `supports_missing(mple)`: with NA dyads R's
        # dyad-independent MLE is the logistic regression on the 116
        # observed dyads, which is exactly what `mple` computes. 1e-6.
        di = fit_ergm(masked, [Edges(), NodeCov(:wealth)])
        @test di.missing_method === :available_case
        @test nobs(di) == 116
        @test g.values["di_terms"] == ["edges", "nodecov.wealth"]
        @test check_golden(g, "di_coefficients", di.coefficients) ||
              error(golden_report(g, "di_coefficients", di.coefficients))
        @test check_golden(g, "di_std_errors", di.std_errors) ||
              error(golden_report(g, "di_std_errors", di.std_errors))
        @test di.loglik ≈ g.values["di_loglik"] atol = 1e-6
        @test di.aic ≈ g.values["di_aic"] atol = 1e-6

        # --- (c) missing-data MCMLE == R's missing-data MCMLE -------------
        # Both sides are Monte Carlo: the mean of five seeded `missing=:mle`
        # fits against the frozen R fit, at the tolerance the fixture
        # justifies from R's own seed-to-seed spread. Observed gap of the
        # five-fit mean to R's frozen fit: 0.0063 (edges), 0.0075 (gwesp);
        # to R's own five-seed mean: 0.002 / 0.0002.
        @test g.values["dd_terms"] == ["edges", "gwesp.fixed.0.5"]
        dd_fits = [fit_ergm(masked, [Edges(), GWESP(0.5)]; method=:mcmle,
                            missing=:mle, n_samples=4096, rng=Random.Xoshiro(s))
                   for s in (101, 202, 303, 404, 505)]
        @test all(f.converged for f in dd_fits)
        @test all(f.missing_method === :mle for f in dd_fits)
        dd_coef = mean(f.coefficients for f in dd_fits)
        dd_se = mean(f.std_errors for f in dd_fits)
        @test check_golden(g, "dd_coefficients", dd_coef) ||
              error(golden_report(g, "dd_coefficients", dd_coef))
        @test check_golden(g, "dd_std_errors", dd_se) ||
              error(golden_report(g, "dd_std_errors", dd_se))
        r_sd = Float64.(g.values["mcmle_seed_sd"])
        gap = abs.(dd_coef .- Float64.(g.values["dd_coefficients"]))
        @test all(gap .< 3 .* r_sd)
        # R's "MCMC %" column rounds to 0 on this budget; so does ours
        @test all(ERGM._mcmc_percent(f) == Int.(g.values["dd_mcmc_percent"]) for f in dd_fits)
        # Unlike the unmasked fixture, the MCMLE here does take Newton steps:
        # the available-case MPLE is not the missing-data MLE
        ac = fit_ergm(masked, [Edges(), GWESP(0.5)]; method=:mple)
        @test abs(dd_coef[2] - ac.coefficients[2]) > 3 * r_sd[2]

        # --- (d) :condition_on_face and :mle are different estimands -----
        # Conditioning on the face values (two masked ties counted as ties,
        # two masked non-ties as non-ties) lands far from the missing-data
        # MLE — more than 3 of R's seed sds on both coefficients (observed
        # gap 0.11 on both) — which is why it is never the default.
        cf = @test_logs (:warn, r"conditions on them at their face value") match_mode=:any fit_ergm(
            masked, [Edges(), GWESP(0.5)]; method=:mcmle, missing=:condition_on_face,
            n_samples=4096, rng=Random.Xoshiro(101))
        @test all(abs.(cf.coefficients .- dd_coef) .> 3 .* r_sd)
    end


    # ------------------------------------------------------------------
    # Sampler, MCMLE, inference defaults,
    # terms and API hygiene
    # ------------------------------------------------------------------
    @testset "TNT proposal: exact against enumeration, with its Hastings ratio" begin
        # Exact distribution of the sufficient statistics by enumerating every
        # graph on the toggleable dyads (`fixed` dyads held at a value)
        function exact_stats_dist(n, directed, terms, θ; fixed=Dict{Tuple{Int,Int},Bool}())
            dy = directed ? [(i, j) for i in 1:n for j in 1:n if i != j] :
                            [(i, j) for i in 1:n for j in (i + 1):n]
            free = [d for d in dy if !haskey(fixed, d)]
            ts = TermSet(terms)
            d = Dict{Vector{Float64},Float64}()
            for m in 0:(2^length(free) - 1)
                g = network(n; directed=directed)
                for (k, (i, j)) in enumerate(free)
                    (m >> (k - 1)) & 1 == 1 && add_edge!(g, i, j)
                end
                for ((i, j), v) in fixed
                    v && add_edge!(g, i, j)
                end
                st = compute_all(ts, g)
                d[st] = get(d, st, 0.0) + exp(dot(θ, st))
            end
            Z = sum(values(d))
            return Dict(k => v / Z for (k, v) in d)
        end
        function empirical(stats)
            d = Dict{Vector{Float64},Float64}()
            N = size(stats, 1)
            for r in eachrow(stats)
                k = collect(r)
                d[k] = get(d, k, 0.0) + 1 / N
            end
            return d
        end
        tv(a, b) = 0.5 * sum(abs(get(a, k, 0.0) - get(b, k, 0.0)) for k in union(keys(a), keys(b)))

        # (a) Both proposals sample exp(θ'g)/Z exactly: total-variation
        # distance of 300 000 draws to the enumerated distribution (measured
        # 0.002-0.006; the uncorrected TNT below is at 0.25-0.58)
        cases = ((4, false, [Edges(), Triangle(), Kstar(2)], [-0.3, 0.6, -0.2]),
                 (4, false, [Edges(), Triangle()], [-2.5, 1.0]),   # sparse: E = 0 ↔ 1 often
                 (3, true, [Edges(), Mutual()], [-1.0, 1.5]),
                 (3, true, [Edges(), Mutual()], [-3.0, 0.5]))
        for (n, directed, terms, θ) in cases
            exact = exact_stats_dist(n, directed, terms, θ)
            model = ERGMModel(ERGMFormula(terms), network(n; directed=directed))
            for prop in (:tnt, :random)
                out = mh_sample(model, θ; n_samples=300_000, burnin=1000, interval=1,
                                rng=Random.Xoshiro(1), proposal=prop)
                @test tv(exact, empirical(out.stats)) < 0.015
            end
        end

        # (b) The test has power: the same tie/no-tie moves WITHOUT the
        # Hastings correction converge to a visibly different distribution
        function tnt_without_hastings(model, θ; N=300_000)
            net = copy(model.network); terms = model.formula.terms
            s = ERGM._TNTState(net, :free); n = nv(net); D = is_directed(net)
            propose = function (rng)
                if rand(rng) < 0.5 && s.n_edges > 0
                    v, k = ERGM._fw_find(s.tree, s.topbit, rand(rng, 1:s.slots))
                    u = Int(outneighbors(net, v)[k])
                    return D ? (v, u) : (min(v, u), max(v, u))
                end
                i = rand(rng, 1:n); j = rand(rng, 1:n)
                while i == j || (!D && j < i)
                    i = rand(rng, 1:n); j = rand(rng, 1:n)
                end
                return (i, j)
            end
            cur = compute_all(terms, net)
            rows = zeros(N, length(terms))
            mh_toggle!(Random.Xoshiro(1), θ, zeros(length(terms)), propose,
                       (dl, mv) -> (ERGM.change_stat_all!(dl, terms, net, mv...); has_edge(net, mv...)),
                       (mv, rem) -> begin
                           delta = ERGM.change_stat_all(terms, net, mv...)
                           rem ? rem_edge!(net, mv...) : add_edge!(net, mv...)
                           cur .+= (rem ? -1 : 1) .* delta
                           ERGM._tnt_toggled!(s, mv[1], mv[2], rem, D)
                           nothing
                       end,
                       k -> (rows[k, :] .= cur; nothing);
                       burnin=1000, interval=1, n_samples=N)
            return rows
        end
        for (n, directed, terms, θ) in cases[[2, 3]]
            model = ERGMModel(ERGMFormula(terms), network(n; directed=directed))
            @test tv(exact_stats_dist(n, directed, terms, θ),
                     empirical(tnt_without_hastings(model, θ))) > 0.1
        end

        # (c) Detailed balance, exactly: the TNT transition kernel on the 64
        # graphs of 4 vertices, with the proposal probabilities written out
        # (a tie: ½/E + ½/N; a non-tie: ½/N; no tie at all: every dyad 1/N)
        # and the package's `_tnt_log_hastings` in the acceptance ratio
        n = 4
        dy = [(i, j) for i in 1:n for j in (i + 1):n]
        N = length(dy)
        ts = TermSet([Edges(), Triangle()])
        θ = [-2.5, 1.0]
        gstat = Dict{Int,Vector{Float64}}()
        for m in 0:(2^N - 1)
            g = network(n; directed=false)
            for (k, (i, j)) in enumerate(dy)
                (m >> (k - 1)) & 1 == 1 && add_edge!(g, i, j)
            end
            gstat[m] = compute_all(ts, g)
        end
        π = [exp(dot(θ, gstat[m])) for m in 0:(2^N - 1)]
        π ./= sum(π)
        P = zeros(2^N, 2^N)
        for m in 0:(2^N - 1)
            E = count_ones(m)
            for k in 1:N
                tie = (m >> (k - 1)) & 1 == 1
                q = E > 0 ? (tie ? 0.5 / E : 0.0) + 0.5 / N : 1.0 / N
                m2 = m ⊻ (1 << (k - 1))
                logr = dot(θ, gstat[m2] .- gstat[m]) + ERGM._tnt_log_hastings(E, N, tie)
                P[m + 1, m2 + 1] += q * min(1.0, exp(logr))
            end
            P[m + 1, m + 1] = 1 - sum(P[m + 1, :])
        end
        @test maximum(abs.(π .* P .- (π .* P)')) < 1e-15        # π_a P_ab == π_b P_ba
        @test maximum(abs.(P' * π .- π)) < 1e-14

        # (d) Masked dyads: under `:free` (`missing=:condition_on_face`) the
        # frozen tie is never proposed and the chain samples the conditional
        # model; under `:all` it samples the unconditional one
        θm = [-1.0, 0.8]
        masked = network(4; directed=false)
        add_edge!(masked, 1, 2); add_edge!(masked, 2, 3)
        set_missing_dyad!(masked, 1, 2)
        mm = ERGMModel(ERGMFormula([Edges(), Triangle()]), masked)
        cond = exact_stats_dist(4, false, [Edges(), Triangle()], θm;
                                fixed=Dict((1, 2) => true))
        uncond = exact_stats_dist(4, false, [Edges(), Triangle()], θm)
        out_free = mh_sample(mm, θm; n_samples=200_000, burnin=1000, interval=1,
                             rng=Random.Xoshiro(2), missing=:condition_on_face,
                             return_networks=false)
        @test tv(cond, empirical(out_free.stats)) < 0.015
        out_all = mh_sample(mm, θm; n_samples=200_000, burnin=1000, interval=1,
                            rng=Random.Xoshiro(3), toggleable=:all)
        @test tv(uncond, empirical(out_all.stats)) < 0.015

        # (e) The default and the vocabulary
        flo = florentine_marriage()
        model = ERGMModel(ERGMFormula([Edges(), Triangle()]), flo)
        @test mh_sample(model, [-1.5, 0.2]; n_samples=20, rng=Random.Xoshiro(4)).stats ==
              mh_sample(model, [-1.5, 0.2]; n_samples=20, rng=Random.Xoshiro(4),
                        proposal=:tnt).stats
        for f in (mh_sample, sample_networks)
            @test_throws ArgumentError f(model, [-1.5, 0.2]; proposal=:bogus)
        end
        @test_throws ArgumentError mcmle(model; proposal=:bogus)
        for f in (mh_sample, sample_networks, mcmle, mple)
            @test :proposal in Base.kwarg_decl(only(methods(f)))
        end
        for f in (simulate_ergm, gof)
            @test :proposal in Base.kwarg_decl(only(methods(f, (ERGMResult,))))
        end
    end

    @testset "SPDyad proposal: exact against enumeration, detailed balance, state" begin
        # (a) The shared-partner-focused mixture samples exp(θ'g)/Z exactly
        function exact_dist(n, directed, terms, θ)
            dy = directed ? [(i, j) for i in 1:n for j in 1:n if i != j] :
                            [(i, j) for i in 1:n for j in (i + 1):n]
            ts = TermSet(terms)
            d = Dict{Vector{Float64},Float64}()
            for m in 0:(2^length(dy) - 1)
                g = network(n; directed=directed)
                for (k, (i, j)) in enumerate(dy)
                    (m >> (k - 1)) & 1 == 1 && add_edge!(g, i, j)
                end
                st = round.(compute_all(ts, g); digits=6) .+ 0.0
                d[st] = get(d, st, 0.0) + exp(dot(θ, st))
            end
            Z = sum(values(d))
            return Dict(k => v / Z for (k, v) in d)
        end
        function empirical(stats)
            d = Dict{Vector{Float64},Float64}()
            for r in eachrow(stats)
                k = round.(collect(r); digits=6) .+ 0.0
                d[k] = get(d, k, 0.0) + 1 / size(stats, 1)
            end
            return d
        end
        tv(a, b) = 0.5 * sum(abs(get(a, k, 0.0) - get(b, k, 0.0)) for k in union(keys(a), keys(b)))
        for (n, directed, terms, θ) in ((4, false, [Edges(), Triangle(), Kstar(2)], [-0.3, 0.6, -0.2]),
                                        (5, false, [Edges(), GWESP(0.5)], [-1.5, 0.8]),
                                        (4, false, [Edges(), Triangle()], [-2.5, 1.0]),
                                        (4, true, [Edges(), GWESP(0.5)], [-1.5, 0.7]),
                                        (3, true, [Edges(), Mutual()], [-3.0, 0.5]))
            model = ERGMModel(ERGMFormula(terms), network(n; directed=directed))
            out = mh_sample(model, θ; n_samples=400_000, burnin=1000, interval=1,
                            rng=Random.Xoshiro(1), proposal=:spdyad)
            @test tv(exact_dist(n, directed, terms, θ), empirical(out.stats)) < 0.015
        end

        # (b) The Hastings ratio the package computes is log q(d|y*)/q(d|y) of
        # the mixture written out independently — L(y) found by brute force —
        # and the chain it defines satisfies detailed balance exactly, for
        # every graph on 4 vertices and every dyad, undirected and directed
        for directed in (false, true)
            n = 4
            dy = directed ? [(i, j) for i in 1:n for j in 1:n if i != j] :
                            [(i, j) for i in 1:n for j in (i + 1):n]
            N = length(dy)
            ts = TermSet([Edges(), GWESP(0.5)])
            θ = [-1.2, 0.6]
            graph(m) = (g = network(n; directed=directed);
                        for (k, (i, j)) in enumerate(dy)
                            (m >> (k - 1)) & 1 == 1 && add_edge!(g, i, j)
                        end; g)
            sp(g, i, j) = directed ?
                count(k -> k != i && k != j && has_edge(g, i, k) && has_edge(g, k, j), 1:n) :
                count(k -> k != i && k != j && has_edge(g, i, k) && has_edge(g, j, k), 1:n)
            Lset(g) = [d for d in dy if sp(g, d...) > 0]
            function q(g, d)                 # the mixture proposal probability
                E = Int(ne(g)); L = Lset(g); tie = has_edge(g, d...)
                tnt = E == 0 ? 1 / N : (tie ? 0.5 / E + 0.5 / N : 0.5 / N)
                isempty(L) && return tnt
                return 0.25 * (d in L ? 1 / length(L) : 0.0) + 0.75 * tnt
            end
            nstate = 2^N
            π = [exp(dot(θ, compute_all(ts, graph(m)))) for m in 0:(nstate - 1)]
            π ./= sum(π)
            worst_h = 0.0
            worst_db = 0.0
            for m in 0:(nstate - 1)
                g = graph(m)
                st = ERGM._SPDyadState(g, :free)
                @test sort(st.keys) == sort(Lset(g))
                for (k, d) in enumerate(dy)
                    m2 = m ⊻ (1 << (k - 1))
                    g2 = graph(m2)
                    removal = has_edge(g, d...)
                    h = ERGM._proposal_log_hastings(st, g, d[1], d[2], removal)
                    worst_h = max(worst_h, abs(h - log(q(g2, d) / q(g, d))))
                    a12 = min(1.0, π[m2 + 1] / π[m + 1] * exp(h))
                    h21 = log(q(g, d) / q(g2, d))
                    a21 = min(1.0, π[m + 1] / π[m2 + 1] * exp(h21))
                    worst_db = max(worst_db, abs(π[m + 1] * q(g, d) * a12 -
                                                 π[m2 + 1] * q(g2, d) * a21))
                end
            end
            @test worst_h < 1e-12
            @test worst_db < 1e-15
        end

        # (c) The incrementally updated state equals a freshly built one after
        # thousands of toggles, undirected and directed
        for directed in (false, true)
            rng = Random.Xoshiro(3)
            g = network(12; directed=directed)
            st = ERGM._SPDyadState(g, :free)
            for _ in 1:3000
                i, j = rand(rng, 1:12), rand(rng, 1:12)
                (i == j || (!directed && j < i)) && continue
                removal = has_edge(g, i, j)
                removal ? rem_edge!(g, i, j) : add_edge!(g, i, j)
                ERGM._proposal_toggled!(st, g, i, j, removal)
            end
            fresh = ERGM._SPDyadState(g, :free)
            @test Dict(zip(st.keys, st.cnt)) == Dict(zip(fresh.keys, fresh.cnt))
            @test all(st.pos[k] == idx for (idx, k) in enumerate(st.keys))
            @test st.tnt.n_edges == ne(g) && st.tnt.slots == fresh.tnt.slots
        end

        # (d) Efficiency, the reason for it: on faux.mesa.high's gwesp model
        # the effective sample size per draw is several times TNT's
        fmh = load_dataset(:faux_mesa_high)
        model = ERGMModel(ERGMFormula([Edges(), NodeMatch(:Grade), NodeMatch(:Race),
                                       GWESP(0.25)]), fmh)
        θ = [-6.41, 1.99, 0.285, 1.449]
        ess(prop) = ERGM._effective_sample_size(
            mh_sample(model, θ; n_samples=600, rng=Random.Xoshiro(1), proposal=prop).stats)
        @test ess(:spdyad) > 2.5 * ess(:tnt)
        # With frozen masked dyads the proposal falls back to TNT
        masked = copy(florentine_marriage()); set_missing_dyad!(masked, 1, 9)
        mm = ERGMModel(ERGMFormula([Edges(), Triangle()]), masked)
        kw = (n_samples=20, rng=Random.Xoshiro(4), missing=:condition_on_face)
        @test mh_sample(mm, [-1.5, 0.2]; kw..., proposal=:spdyad).stats ==
              mh_sample(mm, [-1.5, 0.2]; n_samples=20, rng=Random.Xoshiro(4),
                        missing=:condition_on_face, proposal=:tnt).stats
    end

    @testset "ESS-adaptive MCMLE (effective_size=) and the Hummel step" begin
        # `_hummel_step` is the public building block: full step when the
        # cloud covers the target, a doubled partial step otherwise
        rng = Random.Xoshiro(1)
        draws = randn(rng, 2000, 2) .+ [1.0 5.0]
        near = ERGM._hummel_step(draws, [1.1, 5.0])
        @test near.step_length == 1.0 && !near.singular
        @test near.delta ≈ cov(draws) \ ([1.1, 5.0] .- vec(mean(draws, dims=1)))
        far = ERGM._hummel_step(draws, [9.0, 5.0]; step_length=0.1)
        @test far.step_length == 0.2
        capped = ERGM._hummel_step(draws, [1e3, 5.0]; step_length=1.0, max_step_norm=2.0)
        @test norm(capped.delta) ≈ 2.0
        @test ERGM._hummel_step(hcat(draws[:, 1], draws[:, 1]), [1.0, 1.0]).singular
        @test_throws ArgumentError ERGM._hummel_step(draws, [1.0])

        # The public drivers: `mcmle_solve` on a toy family g ~ N(θ, I) ...
        trng = Random.Xoshiro(1)
        sol = ERGM.Extension.mcmle_solve((θ, n) -> randn(trng, n, 2) .+ θ', [0.0, 0.0];
                                labels=["a", "b"], n_samples=2000, target=[1.0, -2.0])
        @test sol.converged && sol.step_length == 1.0
        @test sol.coef ≈ [1.0, -2.0] atol = 0.1
        @test sol.se ≈ [1.0, 1.0] atol = 0.1
        @test sol.termination_p < 0.01 && size(sol.final.samples) == (2000, 2)
        # ... which never steps with take_step=false, and says when it cannot
        calls = Ref(0)
        nostep = ERGM.Extension.mcmle_solve((θ, n) -> (calls[] += 1; randn(trng, n, 1) .+ θ'), [0.3];
                                   labels=["a"], n_samples=500, target=[0.3],
                                   take_step=false)
        @test nostep.coef == [0.3] && nostep.converged
        hot = ERGM.Extension.mcmle_solve((θ, n) -> randn(trng, n, 1) .+ θ', [0.0]; labels=["a"],
                                n_samples=4000, target=[0.5], termination=:hotelling,
                                maxiter=40)
        @test abs(hot.coef[1] - 0.5) < 0.1
        @test_throws ArgumentError ERGM.Extension.mcmle_solve((θ, n) -> zeros(n, 1), [0.0];
                                                     labels=["a"], n_samples=10,
                                                     termination=:nope)
        # `bridge_integrate` / `_bridge_quadrature`: exact on a known family
        exact = log1p(exp(2.0)) - log(2.0)
        est = ERGM.Extension.bridge_integrate((θu, k) -> [1 / (1 + exp(-θu[1]))], [0.0], [2.0]; rungs=8)
        @test est ≈ exact atol = 1e-5
        @test ERGM.Extension.bridge_integrate((θu, k) -> [1 / (1 + exp(-θu[1]))], [0.0], [2.0];
                                     rungs=8, threaded=true) == est
        @test ERGM.Extension.bridge_integrate((θu, k) -> [0.0, 1.0], [-Inf, 1.0], [-Inf, 1.0]) == 0.0
        @test ERGM._bridge_quadrature([u^3 for u in 0:0.25:1]) ≈ 0.25
        # `ess_sample`: the interval doubles until the target ESS is met
        arng = Random.Xoshiro(1); st = Ref(0.0)
        function ar_extend(c, n, interval)
            out = zeros(n, 1)
            for k in 1:n
                for _ in 1:interval
                    st[] = 0.95 * st[] + randn(arng)
                end
                out[k, 1] = st[]
            end
            return out
        end
        S1, c1, iv1 = ERGM.Extension.ess_sample(ar_extend, 150.0, 256; interval=1)
        @test size(S1) == (256, 1) && c1 == [256] && iv1 >= 8
        @test ERGM._effective_sample_size(S1, c1) >= 150

        # Adaptive sampling reaches the target ESS and R's estimates
        flo = florentine_marriage()
        model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo)
        fits = [mcmle(model; effective_size=64, bridge_rungs=0, rng=Random.Xoshiro(s))
                for s in (101, 202, 303)]
        @test all(f -> f.converged, fits)
        @test all(f -> f.mcmc_convergence.n_eff >= 64, fits)
        g = load_golden(joinpath(@__DIR__, "fixtures", "flomarriage_ergm.toml"))
        @test check_golden(g, "dd_coefficients", mean(coef.(fits)))
        # Reproducible, and thread-count independent by construction (one
        # seed per chain from rng)
        a = mcmle(model; effective_size=64, n_chains=2, bridge_rungs=0, rng=Random.Xoshiro(7))
        b = mcmle(model; effective_size=64, n_chains=2, bridge_rungs=0, rng=Random.Xoshiro(7))
        @test coef(a) == coef(b) && a.chain_lengths == b.chain_lengths
        @test length(a.chain_lengths) == 2
        @test_throws ArgumentError mcmle(model; effective_size=2)
        # R ergm 4's design is the default: SPDyad (`MCMC.prop = ~sparse +
        # .triadic` selects it for every one-mode network) and
        # MCMLE.effectiveSize = 64 — a default fit IS the explicit one, draw
        # for draw; fixed-size TNT sampling is one keyword pair away
        d = mcmle(model; n_chains=2, bridge_rungs=0, rng=Random.Xoshiro(7))
        e = mcmle(model; proposal=:spdyad, effective_size=64, n_chains=2, bridge_rungs=0,
                  rng=Random.Xoshiro(7))
        @test coef(d) == coef(e) && d.mcmc_samples == e.mcmc_samples
        @test coef(d) == coef(a)            # `a` above: effective_size=64 given, proposal default
        @test coef(d) != coef(mcmle(model; proposal=:tnt, n_chains=2, bridge_rungs=0,
                                    rng=Random.Xoshiro(7)))
        t = mcmle(model; proposal=:tnt, effective_size=nothing, n_samples=500,
                  bridge_rungs=0, rng=Random.Xoshiro(7))
        @test size(t.mcmc_samples, 1) >= 500 && t.chain_lengths == [size(t.mcmc_samples, 1)]
        @test occursin("proposal::Symbol=:spdyad", string(@doc mcmle))
        @test occursin("effective_size=64", string(@doc mcmle))
        # The sampler itself: the returned sample meets the target
        ch = ERGM._AdaptiveChains([copy(flo)], 2, false)
        S, counts = ERGM._mcmc_sample_ess!(ch, model, [-1.7, 0.1], 100.0, 256, 500;
                                           rng=Random.Xoshiro(2))
        @test size(S) == (256, 2) && counts == [256]
        @test ERGM._effective_sample_size(S, counts) >= 100
        @test ch.interval > 2 && ch.burned

        # `mcmle_sampler`, the extension API's packaging of that sampler (the
        # one `mcmle` and the curved MCMLE run on, and ERGMEgo's moment
        # matching): the same draws as the internal sampler, draw for draw
        s1 = ERGM.Extension.mcmle_sampler(model; rng=Random.Xoshiro(2), burnin=500,
                                          interval=16)
        @test s1.n_samples == 256                      # max(256, 32·2, 4·64)
        d1 = s1.draw([-1.7, 0.1], 256)
        ch2 = ERGM._AdaptiveChains([copy(flo)], 2, false)
        S2, c2 = ERGM._mcmc_sample_ess!(ch2, model, [-1.7, 0.1], 64.0, 256, 500;
                                        rng=Random.Xoshiro(2), proposal=:spdyad)
        @test d1.samples == S2 && d1.chain_lengths == c2
        # the chain is continued: a second call starts where the first ended
        d2 = s1.draw([-1.7, 0.1], 256)
        @test d2.samples != d1.samples && ERGM._effective_sample_size(d2.samples, d2.chain_lengths) >= 64
        # the boost raises the target ESS (capped at max_n_samples / 4) and
        # returns the number of draws to store
        s2 = ERGM.Extension.mcmle_sampler(model; n_samples=100, rng=Random.Xoshiro(1))
        @test s2.resize(256, 2.0) == 512            # target 128 → 4·128
        @test s2.resize(512, 2.0) == 1024           # target 256
        @test s2.resize(1024, 2.0) == 1600          # target capped at 1600/4 = 400 → n_max
        @test s2.resize(1600, 2.0) == 1600
        # fixed-size: mh_sample's draws exactly; the boost multiplies the count
        f1 = ERGM.Extension.mcmle_sampler(model; effective_size=nothing, n_samples=300,
                                          proposal=:tnt, rng=Random.Xoshiro(3))
        @test f1.n_samples == 300
        df = f1.draw([-1.7, 0.1], 300)
        @test df.samples == mh_sample(model, [-1.7, 0.1]; n_samples=300,
                                      rng=Random.Xoshiro(3), proposal=:tnt).stats
        @test df.chain_lengths == [300]
        @test f1.resize(300, 1.5) == 450 && f1.resize(4500, 2.0) == 4800 &&
              f1.resize(4800, 2.0) == 4800
        # another model on the same vertex set (a curved family's working
        # statistics) is sampled by the same chains
        m3 = ERGMModel(ERGMFormula([Edges()]), flo)
        @test size(s1.draw([-1.7], 256; model=m3).samples) == (256, 1)
        # `mcmle` IS mcmle_solve on this sampler: the same fit, coefficient
        # for coefficient
        rs = Random.Xoshiro(9)
        sm = ERGM.Extension.mcmle_sampler(model; n_samples=1000, rng=rs)
        target = Vector{Float64}(compute_all(model.formula.terms, flo))
        st = mple(model)
        rs2 = Random.Xoshiro(9)
        fm = mcmle(model; init=coef(st), bridge_rungs=0, rng=rs2)
        sol = ERGM.Extension.mcmle_solve((θ, n) -> (d = sm.draw(θ, n);
                                         (samples=d.samples, chain_lengths=d.chain_lengths,
                                          target=target)),
                                         coef(st); labels=["edges", "gwesp"],
                                         n_samples=sm.n_samples, resize=sm.resize)
        @test sol.coef == coef(fm)
        # refusals, before any draw
        @test_throws ArgumentError ERGM.Extension.mcmle_sampler(model; effective_size=4)
        @test_throws ArgumentError ERGM.Extension.mcmle_sampler(model; proposal=:gibbs)
        @test_throws ArgumentError ERGM.Extension.mcmle_sampler(model; n_samples=1)
        @test_throws ArgumentError ERGM.Extension.mcmle_sampler(model; n_samples=100,
                                                                max_n_samples=50)
        masked = copy(flo); set_missing_dyad!(masked, 1, 9)
        @test_throws ArgumentError ERGM.Extension.mcmle_sampler(
            ERGMModel(ERGMFormula([Edges()]), masked))
    end

    @testset "TNT sampler is allocation-free per step" begin
        # The kernel with a Hastings callable adds 0 bytes per step
        state = Ref(false)
        kernel(b) = mh_toggle!(Random.Xoshiro(1), [log(3.0)], [0.0],
                               rng -> 1,
                               (delta, move) -> (delta[1] = 1.0; state[]),
                               (move, removal) -> (state[] = !removal),
                               k -> nothing;
                               burnin=b, interval=1, n_samples=1,
                               hastings=(move, removal) -> removal ? -0.1 : 0.1)
        kernel(100)
        @test (@allocated kernel(20_000)) == (@allocated kernel(10_000))
        # ... and so does a full network MH step, both proposals, measured
        # EXACTLY: on a saturated network (θ_edges = 50, every dyad a tie, so
        # every proposal — TNT's tie branch and its dyad branch alike — is a
        # rejected removal) nothing toggles, Graphs.jl never regrows an
        # adjacency vector (that cost, ~100 B per *accepted* insertion, is
        # NetworkCore/Graphs' `add_edge!`, not the sampler's), and 20 000 extra
        # steps allocate nothing (the same bound as benchmark/regression_tests.jl)
        sat = network(60; directed=false)
        satm = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), sat)
        θs = [50.0, 0.0]
        for prop in (:tnt, :random, :spdyad)
            g = copy(sat)
            ERGM._mh_run!(Random.Xoshiro(1), g, satm.formula.terms, θs, 1, 200_000, 1,
                          false; proposal=prop)
            @test ne(g) == 60 * 59 ÷ 2
            run(b) = ERGM._mh_run!(Random.Xoshiro(2), g, satm.formula.terms, θs, 1, b, 1,
                                   false; proposal=prop)
            run(1000)
            # Zero allocation per step: 20,000 extra steps add nothing. The
            # slack absorbs the few dozen bytes of platform noise (Windows);
            # even 1 byte per step would exceed it 80-fold.
            @test abs((@allocated run(40_000)) - (@allocated run(20_000))) <= 256
        end
        flo = florentine_marriage()
        # The Fenwick tree draws and updates without allocating
        s = ERGM._TNTState(flo, :free)
        ERGM._fw_find(s.tree, s.topbit, 3)
        @test (@allocated ERGM._fw_find(s.tree, s.topbit, 3)) == 0
        @test (@allocated ERGM._tnt_toggled!(s, 1, 2, false, false)) == 0
        # ... and a uniform slot is a uniform tie: every tie of flomarriage is
        # reached exactly twice over the 2E slots
        s = ERGM._TNTState(flo, :free)
        @test s.slots == 2 * ne(flo) && s.n_edges == ne(flo)
        hits = Dict{Tuple{Int,Int},Int}()
        for r in 1:s.slots
            v, k = ERGM._fw_find(s.tree, s.topbit, r)
            u = Int(outneighbors(flo, v)[k])
            e = (min(v, u), max(v, u))
            hits[e] = get(hits, e, 0) + 1
        end
        @test length(hits) == ne(flo) && all(==(2), values(hits))
    end

    @testset "MCMLE always takes a Monte-Carlo step (flomarriage edges + triangle)" begin
        # Before 0.2 the convergence test ran BEFORE the first Newton update,
        # and 12 of 30 seeds returned the MPLE unchanged. statnet's order:
        # step at every iteration, then test.
        #
        # The deterministic pin: a sampler whose draws sit 0.02 sd from the
        # target at the starting point passes both stopping rules THERE, so
        # the old order returned the start verbatim; the iteration must
        # instead return the start plus the full Newton step Σ⁻¹(target − ḡ)
        # (draws that do not depend on θ make that step exact).
        srng = Random.Xoshiro(7)
        S0 = randn(srng, 4000, 2)
        S0 .-= mean(S0, dims=1)
        S0 = S0 * inv(cholesky(Symmetric(cov(S0))).U)     # mean 0, covariance I
        shift = [0.02, -0.02]
        fixed_draw(θ, n) = S0[1:n, :] .+ shift'           # independent of θ
        θ0 = [0.3, -0.1]
        for rule in (:confidence, :hotelling)
            sol = ERGM.Extension.mcmle_solve(fixed_draw, θ0; labels=["a", "b"], n_samples=4000,
                                    termination=rule, maxiter=3)
            @test sol.coef != θ0                          # a step was taken
            @test sol.coef ≈ θ0 .+ cov(S0) \ (zeros(2) .- vec(mean(fixed_draw(θ0, 4000), dims=1))) atol = 1e-8
            @test sol.converged && sol.iterations == 1
        end
        # ... and at the start the old (test-first) rule would indeed have
        # stopped: max t 0.02 < 0.1 and a Hotelling p far above 0.05
        legacy = ERGM.mcmc_convergence(fixed_draw(θ0, 4000), zeros(2))
        @test legacy.converged

        # The same on real fits: no default MCMLE fit returns the MPLE
        g = load_golden(joinpath(@__DIR__, "fixtures", "mcmle_ergm.toml"))
        flo = florentine_marriage()
        mp = coef(fit_ergm(flo, [Edges(), Triangle()]; method=:mple))
        @test check_golden(g, "tri_mple", mp) || error(golden_report(g, "tri_mple", mp))
        K = 10
        fits = [fit_ergm(flo, [Edges(), Triangle()]; method=:mcmle, bridge_rungs=0,
                         rng=Random.Xoshiro(s)) for s in 1:K]
        @test all(f -> f.converged, fits)
        @test !any(f -> coef(f) == mp, fits)
        # A consistency check, not the pin above: the mean of ten fits is
        # within 4 standard errors of R's own default estimator (its ten-seed
        # sd over √10) of the high-precision MLE. It cannot see the old defect
        # (its mean, 0.179 on triangle, is inside that width of 0.156); the
        # deterministic `mcmle_solve` check and the verbatim-MPLE check can.
        C = reduce(hcat, coef.(fits))'
        tol = golden_tolerance(g, "tri_mean_sd_multiple") .*
              Float64.(g.values["tri_seed_sd"]) ./ sqrt(10)
        gap = abs.(vec(mean(C, dims=1)) .- Float64.(g.values["tri_mle_precise"]))
        @test all(gap .< tol)
        # The verdict and the standard errors come from ONE sample: the one
        # the passing termination test was computed on
        for f in fits
            @test f.termination.rule === :confidence
            @test f.termination.p_value < 1 - f.termination.confidence
            @test f.termination.n_samples == size(f.mcmc_samples, 1)
        end
        printed = sprint(show, fits[1])
        @test occursin("Converged: true\n  Termination: 99% equivalence test p", printed)
    end

    @testset "MCMLE stopping rule: R's confidence test, attainable under MC noise" begin
        # The equivalence test: converged only when the Monte-Carlo
        # confidence region of the estimating equation fits inside the
        # tolerance region (0.1 of the statistics' variance)
        rng = Random.Xoshiro(11)
        S = randn(rng, 4000, 2)
        on = ERGM.Extension.confidence_test(S, [4000], vec(mean(S, dims=1)), zeros(2))
        @test on.converged && on.p_value < 0.01
        # 0.5 sd off: outside the tolerance region (√0.1 ≈ 0.32 sd)
        off = ERGM.Extension.confidence_test(S, [4000], vec(mean(S, dims=1)) .+ [0.5, 0.0], zeros(2))
        @test !off.converged
        # On target but too few effective draws to be 99 % sure: not
        # converged, and the next sample is to be enlarged
        few = ERGM.Extension.confidence_test(S[1:40, :], [40], vec(mean(S[1:40, :], dims=1)), zeros(2))
        @test !few.converged && few.boost > 1
        # The ellipsoid distance (R's `.ellipsoid_mahalanobis`): the smallest
        # W-Mahalanobis distance from y to the boundary of the tolerance
        # ellipsoid {x : x'U⁻¹x ≤ 1}. Closed forms: with W = w·I and U = I it
        # is (1 − ‖y‖)²/w; from the centre it is 1/(W's largest eigenvalue)
        # when U = I (the nearest boundary point lies along that axis)
        y = [0.5, 0.0]
        @test ERGM._ellipsoid_mahalanobis(y, 0.01 .* Matrix(I, 2, 2), Matrix(1.0I, 2, 2)) ≈
              (1 - norm(y))^2 / 0.01 atol = 1e-6
        Wd = [0.01 0.0; 0.0 0.04]
        @test ERGM._ellipsoid_mahalanobis([0.0, 0.0], Wd, Matrix(1.0I, 2, 2)) ≈
              1 / maximum(eigvals(Wd)) atol = 1e-6
        # ... and against a brute-force search over the boundary
        erng = Random.Xoshiro(1)
        for _ in 1:3
            A = randn(erng, 2, 2); W = A * A' + 0.1I
            B = randn(erng, 2, 2); U = B * B' + 0.5I
            L = cholesky(U).L
            y = 0.3 .* (L * normalize(randn(erng, 2)))
            brute = minimum(dot(y - L * [cos(a), sin(a)], W \ (y - L * [cos(a), sin(a)]))
                            for a in range(0, 2π; length=20_000))
            @test ERGM._ellipsoid_mahalanobis(y, W, U) ≈ brute rtol = 1e-4
        end
        # The legacy rule stays available, and the vocabulary is checked
        flo = florentine_marriage()
        model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), flo)
        legacy = mcmle(model; termination=:hotelling, n_samples=1000, bridge_rungs=0,
                       rng=Random.Xoshiro(3))
        @test legacy.termination.rule === :hotelling
        @test_throws ArgumentError mcmle(model; termination=:precision)
        @test_throws ArgumentError mcmle(model; conv_confidence=1.0)
        @test_throws ArgumentError mcmle(model; max_n_samples=10, n_samples=100)
    end

    @testset "Golden fixture: statnet ergm MCMLE on faux.mesa.high (provenanced)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "mcmle_ergm.toml"))
        @test g.provenance["ergm_version"] == "4.12.0"
        fmh = load_dataset(:faux_mesa_high)
        ta = [Edges(), NodeMatch(:Grade), NodeMatch(:Race), GWESP(0.25)]
        tb = [ta; GWDegree(0.5)]
        # Same data, same statistics, R's labels
        sa = summary_stats(fmh, ta)
        sb = summary_stats(fmh, tb)
        @test String.(collect(keys(sa))) == g.values["a_terms"]
        @test String.(collect(keys(sb))) == g.values["b_terms"]
        @test check_golden(g, "a_summary", collect(values(sa)))
        @test check_golden(g, "b_summary", collect(values(sb)))

        # (a) The textbook gwesp model AT THE DEFAULTS: converged, and every
        # coefficient and standard error within the fixture's tolerance of
        # R's eleven-seed mean: 4 sd of the difference, floored at a tenth of
        # R's standard error (see [tolerance]). The floor binds for gwesp and
        # for every standard error, so those are compared at a practical-
        # equivalence margin of 10 % of a standard error, not at 4 sd: two
        # MCMC designs carry different O(1/ESS) finite-sample biases, larger
        # than R's own seed sd on those coordinates (0.0015 on gwesp).
        fa = mcmle(ERGMModel(ERGMFormula(ta), fmh); bridge_rungs=0,
                   rng=Random.Xoshiro(20261002))
        @test fa.converged
        @test all(abs.(coef(fa) .- Float64.(g.values["a_coef_mean"])) .<
                  Float64.(g.values["a_coef_tolerance"]))
        @test all(abs.(stderror(fa) .- Float64.(g.values["a_se_mean"])) .<
                  Float64.(g.values["a_se_tolerance"]))

        # (b) ... and with gwdegree, which diverged before 0.2 (gwdeg −27
        # against R's 0.29); four chains, so the threaded CI cell is quick
        fb = mcmle(ERGMModel(ERGMFormula(tb), fmh); bridge_rungs=0, n_chains=4,
                   rng=Random.Xoshiro(20261003))
        @test fb.converged
        @test all(abs.(coef(fb) .- Float64.(g.values["b_coef_mean"])) .<
                  Float64.(g.values["b_coef_tolerance"]))
        @test all(abs.(stderror(fb) .- Float64.(g.values["b_se_mean"])) .<
                  Float64.(g.values["b_se_tolerance"]))

        # (a') R's default design for a triadic model (the shared-partner
        # proposal with ESS-adaptive sampling), spelled out, at a second seed:
        # the same tolerances as (a)
        fs = mcmle(ERGMModel(ERGMFormula(ta), fmh); bridge_rungs=0, proposal=:spdyad,
                   effective_size=64, rng=Random.Xoshiro(20261004))
        @test fs.converged
        @test all(abs.(coef(fs) .- Float64.(g.values["a_coef_mean"])) .<
                  Float64.(g.values["a_coef_tolerance"]))
        @test all(abs.(stderror(fs) .- Float64.(g.values["a_se_mean"])) .<
                  Float64.(g.values["a_se_tolerance"]))
        # ... and its GOF draws are not autocorrelated: the interval is chosen
        # from the measured autocorrelation time (lag-1 0.66 before)
        gf = gof(fs; n_sim=100, n_chains=1, stats=[:degree], rng=Random.Xoshiro(2))
        sims = gf.statistics[1].simulated
        ac1(x) = cor(x[1:(end - 1)], x[2:end])
        @test all(abs(ac1(Float64.(sims[:, k]))) < 0.35 for k in 1:3)

        # (d) The bridge log-likelihood at R's coefficients against R's
        # 64-step bridge, within 4 of R's 16-step seed sds
        ma = ERGMModel(ERGMFormula(ta), fmh)
        b, iv = ERGM._resolve_mcmc_controls(ma, nothing, nothing)
        θR = Float64.(g.values["a_coefficients"])
        ll = ERGM._bridge_loglik(ma, θR, compute_all(ma.formula.terms, fmh); nrungs=16,
                                 n_samples=1000, burnin=b, interval=iv,
                                 rng=Random.Xoshiro(5))
        @test abs(ll - g.values["bridge64_mean"]) <
              golden_tolerance(g, "bridge_sd_multiple") * g.values["bridge16_seed_sd"]
    end

    @testset "Bridge quadrature: Simpson's rule, not the biased trapezoid" begin
        # With the rung expectations computed EXACTLY (enumeration of the
        # 2^15 graphs on 6 vertices), the quadrature error alone is visible:
        # at the default 16 rungs the bridge's Simpson rule is 60× closer to
        # the exact value than the pre-0.2 trapezoid rule
        n = 6
        pairs = [(i, j) for i in 1:n for j in (i + 1):n]
        G = zeros(2^length(pairs), 2)
        for m in 0:(2^length(pairs) - 1)
            adj = falses(n, n)
            for (k, (i, j)) in enumerate(pairs)
                (m >> (k - 1)) & 1 == 1 && (adj[i, j] = adj[j, i] = true)
            end
            tri = count(adj[a, b] && adj[b, c] && adj[a, c]
                        for a in 1:n for b in (a + 1):n for c in (b + 1):n)
            G[m + 1, :] = [count_ones(m), tri]
        end
        logZ(θ) = (v = G * θ; mx = maximum(v); mx + log(sum(exp.(v .- mx))))
        meanstat(θ) = (v = G * θ; w = exp.(v .- maximum(v)); vec(w' * G) ./ sum(w))
        θ = [-1.5, 1.2]
        θ0 = [-1.5, 0.0]
        Δ = θ .- θ0
        exact = logZ(θ) - logZ(θ0)
        m = 16
        f = [dot(Δ, meanstat(θ0 .+ u .* Δ)) for u in range(0, 1; length=m + 1)]
        simpson = ERGM._bridge_quadrature(f)
        trapezoid = (sum(f) - (f[1] + f[end]) / 2) / m
        @test abs(simpson - exact) < 0.05 * abs(trapezoid - exact)   # measured 0.0004 vs 0.022
        @test abs(simpson - exact) < 1e-3
        @test_throws ArgumentError ERGM._bridge_quadrature(f[1:4])     # odd segments
        # An odd rung count is raised to the next even one, so Simpson applies
        flo = florentine_marriage()
        model = ERGMModel(ERGMFormula([Edges(), Triangle()]), flo)
        @test isfinite(ERGM._bridge_logZ(model, [-1.6, 0.2]; nrungs=3, n_samples=50,
                                         burnin=500, interval=20, rng=Random.Xoshiro(1)))
    end

    @testset "Exact log-normaliser includes θ'g(∅) (user term with g(∅) ≠ 0)" begin
        # A user term whose value on the EMPTY network is not 0: the number of
        # non-ties. Every built-in dyad-independent term has g(∅) = 0, which
        # is why the omission went unnoticed: the log-likelihood was off by
        # exactly θ'g(∅) (+259.84 instead of −51.55 on flomarriage).
        flo = florentine_marriage()
        terms = [GEmptyTerm(), NodeCov(:wealth)]
        mp = fit_ergm(flo, terms)
        mc = fit_ergm(flo, terms; method=:mcmle, n_samples=500, rng=Random.Xoshiro(1))
        @test loglikelihood(mc) ≈ loglikelihood(mp) atol = 1e-8
        # non-ties + nodecov is a reparametrisation of edges + nodecov, so the
        # exact MLE log-likelihood is R's `ergm(flomarriage ~ edges +
        # nodecov("wealth"))` (provenanced fixture), whose MPLE is its MLE
        gflo = load_golden(joinpath(@__DIR__, "fixtures", "flomarriage_ergm.toml"))
        @test loglikelihood(mp) ≈ gflo.values["di_loglik"] atol = 1e-6
        @test loglikelihood(mc) ≈ gflo.values["di_loglik"] atol = 1e-6
        model = ERGMModel(ERGMFormula(terms), flo)
        θ = [0.3, 0.01]
        g0 = compute_all(model.formula.terms, ERGM._empty_copy(flo))
        @test g0[1] == 120.0
        # log Z = θ'g(∅) + Σ log(1 + exp(θ'δ)) over the 120 dyads
        manual = dot(θ, g0) + sum(log1p(exp(dot(θ, change_stat_all(model.formula.terms, flo, i, j))))
                                  for i in 1:16 for j in (i + 1):16)
        @test ERGM._dyad_independent_logZ(model, θ) ≈ manual atol = 1e-9
        # ... and through the bridge of a dyad-dependent model, against
        # enumeration on 5 vertices (2^10 graphs)
        n = 5
        net = network(n; directed=false)
        for (i, j) in [(1, 2), (2, 3), (1, 3), (4, 5)]
            add_edge!(net, i, j)
        end
        dm = ERGMModel(ERGMFormula([GEmptyTerm(), Triangle()]), net)
        θd = [0.6, 0.5]
        dy = [(i, j) for i in 1:n for j in (i + 1):n]
        vals = Float64[]
        for m in 0:(2^length(dy) - 1)
            g = network(n; directed=false)
            for (k, (i, j)) in enumerate(dy)
                (m >> (k - 1)) & 1 == 1 && add_edge!(g, i, j)
            end
            push!(vals, dot(θd, compute_all(dm.formula.terms, g)))
        end
        exact = maximum(vals) + log(sum(exp.(vals .- maximum(vals))))
        est = ERGM._bridge_logZ(dm, θd; nrungs=8, n_samples=2000, burnin=500,
                                interval=10, rng=Random.Xoshiro(2))
        @test est ≈ exact atol = 0.1
    end

    @testset "summary_stats validates and expands the formula as a model does" begin
        flo = florentine_marriage()
        sl = samplike()
        # Undefined on the network: refused, never a silent 0 or another statistic
        @test_throws ArgumentError summary_stats(flo, [Mutual()])
        @test_throws ArgumentError summary_stats(flo, [IStar(2)])
        @test_throws ArgumentError summary_stats(sl, [Degree(1)])
        @test_throws ArgumentError summary_stats(sl, [Kstar(2)])
        @test_throws ArgumentError summary_stats(flo, [NodeCov(:welth)])     # misspelled
        e = try summary_stats(flo, [NodeCov(:welth)]) catch err; err end
        @test occursin("wealth", e.msg)                                        # lists what exists
        # Multi-level terms expand to the model's statistics under R's labels
        fmh = load_dataset(:faux_mesa_high)
        ss = summary_stats(fmh, [Edges(), NodeFactor(:Race)])
        model = ERGMModel(ERGMFormula([Edges(), NodeFactor(:Race)]), fmh)
        @test String.(collect(keys(ss))) == model.formula.terms.names
        @test collect(values(ss)) == compute_all(model.formula.terms, fmh)
        # A self-loop is refused, as the model refuses it: on a looped
        # triangle R's summary(~edges + kstar(2) + triangle) is (4, 7, 1),
        # and ERGM.jl's off-diagonal terms would have returned (4, 5, 1)
        lt = network(3; directed=false, loops=true)
        for (i, j) in ((1, 2), (2, 3), (1, 3), (1, 1))
            add_edge!(lt, i, j)
        end
        e = try summary_stats(lt, [Edges(), Kstar(2), Triangle()]); nothing catch err; err end
        @test e isa ArgumentError && occursin("self-loop", e.msg) && occursin("vertex 1", e.msg)
        rem_edge!(lt, 1, 1)                          # loop-free: computed as before
        @test summary_stats(lt, [Edges(), Kstar(2), Triangle()]) ==
              (edges = 3.0, kstar2 = 3.0, triangle = 1.0)
    end

    @testset "Terms that read an edge attribute live are refused by the samplers" begin
        # The samplers' rem_edge! deletes a tie's attributes, so a term that
        # reads one live decays to its default under MCMC: here the weighted
        # edge count would become the edge count. Refused at every sampler
        # entry point; the MPLE (no toggles) and a snapshotting term are fine.
        flo = florentine_marriage()
        for (k, e) in enumerate(collect(edges(flo)))
            set_edge_attribute!(flo, :weight, src(e), dst(e), 1.0 + k / 4)
        end
        live = ERGMModel(ERGMFormula([Edges(), LiveWeightTerm()]), flo)
        θ = [-1.0, -0.2]
        refused(f) = (e = try f(); nothing catch err; err end;
                      e isa ArgumentError && occursin("reads an edge attribute", e.msg) &&
                      occursin("liveweight", e.msg) && occursin("materialize", e.msg))
        @test refused(() -> sample_networks(live, θ; n_sim=2, rng=Random.Xoshiro(1)))
        @test refused(() -> mh_sample(live, θ; n_samples=2, rng=Random.Xoshiro(1)))
        @test refused(() -> mcmle(live; n_samples=50, rng=Random.Xoshiro(1)))
        @test refused(() -> fit_ergm(flo, [Edges(), LiveWeightTerm()]; method=:mcmle))
        # (alone: with Edges() the live weights, > 1 only on ties, separate the
        # pseudo-likelihood, and a separated MPLE refuses the bootstrap first)
        @test refused(() -> fit_ergm(flo, [LiveWeightTerm()]; method=:mple,
                                     se=:bootstrap, n_boot=4))
        mp = fit_ergm(flo, [LiveWeightTerm()]; method=:mple)   # no toggles: fine
        @test mp.converged
        @test refused(() -> simulate_ergm(mp; n_sim=2, rng=Random.Xoshiro(1)))
        @test refused(() -> gof(mp; n_sim=2, rng=Random.Xoshiro(1)))
        # Without the edge attributes there is nothing to lose: not refused
        plain = florentine_marriage()
        @test length(sample_networks(ERGMModel(ERGMFormula([Edges(), LiveWeightTerm()]), plain),
                                     θ; n_sim=2, burnin=10, interval=1,
                                     rng=Random.Xoshiro(1))) == 2
        # A term that snapshots the weights at model construction samples,
        # and its chain is the chain of the equivalent EdgeCov(W)
        snap = ERGMModel(ERGMFormula([Edges(), SnappedWeightTerm()]), flo)
        @test snap.formula.terms[2] isa SnapWeightTerm
        W = snap.formula.terms[2].W
        a = mh_sample(snap, θ; n_samples=20, burnin=50, interval=5, rng=Random.Xoshiro(2))
        b = mh_sample(ERGMModel(ERGMFormula([Edges(), EdgeCov(W)]), flo), θ;
                      n_samples=20, burnin=50, interval=5, rng=Random.Xoshiro(2))
        @test a.stats == b.stats
        # The probe runs only for terms from outside ERGM: built-in terms on
        # a network with edge attributes are never probed
        @test length(sample_networks(ERGMModel(ERGMFormula([Edges(), Triangle()]), flo),
                                     [-1.0, 0.1]; n_sim=2, burnin=10, interval=1,
                                     rng=Random.Xoshiro(3))) == 2
    end

    @testset "An unannotated user term is fitted end to end (no ambiguity)" begin
        # The ERGMUserterms README pattern: `change_stat(::T, net, i, j)` with
        # untyped dyad arguments. Before 0.2 the error fallback was typed
        # `i::Int, j::Int`, so this method was AMBIGUOUS with it inside
        # fit_ergm (MethodError).
        flo = florentine_marriage()
        @test hasmethod(change_stat, Tuple{ExampleTerm, Network{Int,false}, Int, Int})
        fit = fit_ergm(flo, [Edges(), ExampleTerm()])
        @test fit.converged && length(coef(fit)) == 2
        mc = fit_ergm(flo, [Edges(), ExampleTerm()]; method=:mcmle, n_samples=300,
                      rng=Random.Xoshiro(1))
        @test isfinite(coef(mc)[2])
        @test length(simulate_ergm(mc; n_sim=2, rng=Random.Xoshiro(2))) == 2
        @test isempty(Test.detect_ambiguities(ERGM))
        # The fallback names the missing method in words
        e = try change_stat(NoChangeStat(), flo, 1, 2) catch err; err end
        @test e isa ArgumentError && occursin("change_stat(::NoChangeStat, net, i, j)", e.msg)
        e = try compute(NoChangeStat(), flo) catch err; err end
        @test e isa ArgumentError && occursin("compute(::NoChangeStat, net)", e.msg)
    end

    @testset "MPLE of a dyad-dependent formula withholds naive inference by default" begin
        flo = florentine_marriage()
        fit = fit_ergm(flo, [Edges(), Triangle()]; method=:mple)
        @test all(isnan, fit.z_values) && all(isnan, fit.p_values)
        @test all(isfinite, stderror(fit))                 # the naive SEs are shown
        printed = sprint(show, fit)
        @test occursin("z values and p-values are not reported (NaN)", printed)
        @test occursin("se=:bootstrap", printed)
        e = try confint(fit) catch err; err end
        @test e isa ArgumentError && occursin("se=:bootstrap", e.msg)
        @test any(occursin("withheld", a) for a in approximations(fit))
        @test all(values(NetworkCore.check_statsapi(fit; strict=true)))
        # The written opt-in restores R's naive Wald table, with the caveat
        naive = fit_ergm(flo, [Edges(), Triangle()]; method=:mple, se=:hessian)
        @test coef(naive) == coef(fit) && stderror(naive) == stderror(fit)
        @test all(isfinite, naive.p_values)
        @test size(confint(naive)) == (2, 2)
        @test occursin("the p-values should not be trusted", sprint(show, naive))
        # The parametric bootstrap is calibrated: z, p and intervals reported
        boot = fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mple, se=:bootstrap, n_boot=30,
                        rng=Random.Xoshiro(3))
        @test all(isfinite, boot.p_values) && size(confint(boot)) == (2, 2)
        # Dyad-independent formulas are exact and untouched
        di = fit_ergm(flo, [Edges(), NodeCov(:wealth)])
        @test all(isfinite, di.p_values) && size(confint(di)) == (2, 2)
        @test !any(occursin("withheld", a) for a in approximations(di))
    end

    @testset "Threaded samplers rethrow the task's own exception" begin
        flo = florentine_marriage()
        model = ERGMModel(ERGMFormula([Edges(), ThrowingTerm()]), flo)
        for f in (() -> sample_networks(model, [0.0, 0.0]; n_sim=4, n_chains=2),
                  () -> ERGM._mcmc_sample(model, [0.0, 0.0], 10, 10, 1; n_chains=2))
            e = try f() catch err; err end
            @test e isa ArgumentError
            @test e.msg == "boom from a chain"
        end
        t = Threads.@spawn throw(ArgumentError("inner"))
        wrapped = try wait(t); nothing catch err; err end
        @test wrapped isa TaskFailedException
        inner = NetworkCore.unwrap_task_exception(CompositeException([wrapped]))
        @test inner isa ArgumentError && inner.msg == "inner"
        # the threaded loops run on NetworkCore's one helper, not a copy
        @test ERGM.spawn_all === NetworkCore.spawn_all
        @test !isdefined(ERGM, :_spawn_all) && !isdefined(ERGM, :_unwrap_task_exception)
    end

    @testset "Terms compare structurally (==, hash)" begin
        @test NodeFactor(:g) == NodeFactor(:g)
        @test hash(NodeFactor(:g)) == hash(NodeFactor(:g))
        @test NodeFactor(:g) != NodeFactor(:h)
        @test NodeMix(:g) == NodeMix(:g)
        W = [0.0 1.0; 1.0 0.0]
        @test EdgeCov(W) == EdgeCov(copy(W)) && hash(EdgeCov(W)) == hash(EdgeCov(copy(W)))
        @test Degree(0:2) == Degree([0, 1, 2]) && hash(Degree(0:2)) == hash(Degree([0, 1, 2]))
        @test GWESP(0.5) == GWESP(0.5) && GWESP(0.5) != GWESP(0.5; type=:ITP)
        @test Sender() == Sender() && Sender(2) != Receiver(2)
        @test Edges() != Triangle()
        @test length(Set([NodeFactor(:g), NodeFactor(:g), NodeMatch(:g)])) == 2
        # Materialized twins too (a dense attribute snapshot inside)
        flo = florentine_marriage()
        m1 = ERGMModel(ERGMFormula([NodeCov(:wealth)]), flo)
        m2 = ERGMModel(ERGMFormula([NodeCov(:wealth)]), flo)
        @test m1.formula.terms.terms == m2.formula.terms.terms
    end

    @testset "Golden fixture: statnet terms added in 0.2 (concurrent, gwnsp, degrange, meandeg, density, triadcensus, sender/receiver)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "ergm_terms.toml"))
        flo = florentine_marriage()
        sl = samplike()
        TC = ERGM.TriadCensus
        flo_terms = [Concurrent(), GWNSP(0.5), GWNSP(0.0), DegRange(2), DegRange(1, 3),
                     MeanDeg(), Density(), TC(), TransitiveTies(), CyclicalTies()]
        sf = summary_stats(flo, flo_terms)
        @test String.(collect(keys(sf))) == g.values["new_terms_flo_names"]
        @test check_golden(g, "new_terms_flo_summary", collect(values(sf))) ||
              error(golden_report(g, "new_terms_flo_summary", collect(values(sf))))
        sl_terms = [GWNSP(0.5), GWNSP(0.5; type=:ITP), GWNSP(0.5; type=:OSP),
                    GWNSP(0.5; type=:ISP), IDegRange(2), ODegRange(1, 3), MeanDeg(),
                    Density(), TC(0:15), Sender(), Receiver(), TransitiveTies(),
                    CyclicalTies()]
        ss = summary_stats(sl, sl_terms)
        @test String.(collect(keys(ss))) == g.values["new_terms_samplike_names"]
        @test check_golden(g, "new_terms_samplike_summary", collect(values(ss))) ||
              error(golden_report(g, "new_terms_samplike_summary", collect(values(ss))))

        # Change statistics are exact against brute force on random graphs
        rng = Random.Xoshiro(20261002)
        for directed in (false, true), rep in 1:3
            n = 8
            net = network(n; directed=directed)
            for i in 1:n, j in 1:n
                i != j && rand(rng) < 0.3 && add_edge!(net, i, j)
            end
            ts = directed ?
                Any[GWNSP(0.5), GWNSP(0.3; type=:ITP), GWNSP(0.3; type=:OSP),
                    GWNSP(0.3; type=:ISP), MeanDeg(), Density(), IDegRange(1, 3),
                    ODegRange(2), Sender(3), Receiver(4), TransitiveTies(), CyclicalTies(),
                    [TC(l) for l in 0:15]...] :
                Any[Concurrent(), GWNSP(0.5), GWNSP(0.0), DegRange(2), DegRange(1, 3),
                    MeanDeg(), Density(), TransitiveTies(), CyclicalTies(),
                    [TC(l) for l in 0:3]...]
            for t in ts
                @test check_change_stats(t, net) === nothing
            end
        end

        # R refuses the same networks, and so does ERGM.jl
        @test occursin("directed==TRUE", g.values["r_error_concurrent_directed"])
        @test occursin("directed==FALSE", g.values["r_error_sender_undirected"])
        @test_throws ArgumentError summary_stats(sl, [Concurrent()])
        @test_throws ArgumentError summary_stats(sl, [DegRange(2)])
        @test_throws ArgumentError summary_stats(flo, [Sender()])
        @test_throws ArgumentError summary_stats(flo, [IDegRange(1)])
        @test_throws ArgumentError summary_stats(flo, [TC(7)])           # no type 7 undirected
        @test_throws ArgumentError compute(TC(), flo)                    # a specification
        @test_throws ArgumentError compute(Sender(), sl)
        @test_throws ArgumentError DegRange(3, 2)
        # Traits
        @test !is_dyad_dependent(MeanDeg()) && !is_dyad_dependent(Sender())
        @test is_dyad_dependent(Concurrent()) && is_dyad_dependent(GWNSP(0.5))
        @test requires_undirected(Concurrent()) && requires_directed(Receiver())
        # A p1-style fit: edges + mutual + sender + receiver on samplike
        fit = fit_ergm(sl, [Edges(), Mutual(), Sender(), Receiver()]; method=:mple, se=:hessian)
        @test length(coef(fit)) == 2 + 17 + 17
        @test fit.model.formula.terms.names[3] == "sender2"
    end

    @testset "Golden fixture: esp(d) / desp(d, type=) (edgewise shared partners)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "ergm_terms.toml"))
        flo = florentine_marriage()
        sl = samplike()
        # R's summary() and labels: esp<d> undirected, esp.<type><d> directed
        sf = summary_stats(flo, [ESP(0:3)])
        @test String.(collect(keys(sf))) == g.values["esp_flo_names"]
        @test check_golden(g, "esp_flo_summary", collect(values(sf))) ||
              error(golden_report(g, "esp_flo_summary", collect(values(sf))))
        ss = summary_stats(sl, [ESP(0:2), ESP(1; type=:ITP), ESP([0, 2]; type=:OSP),
                                ESP(1; type=:ISP)])
        @test String.(collect(keys(ss))) == g.values["esp_samplike_names"]
        @test check_golden(g, "esp_samplike_summary", collect(values(ss))) ||
              error(golden_report(g, "esp_samplike_summary", collect(values(ss))))
        @test name(ESP(1)) == "esp1" && name(ESP(1; type=:ITP)) == "esp.ITP1"
        @test name(ESP(2), sl) == "esp.OTP2" && name(ESP(2; type=:ISP), flo) == "esp2"
        @test name(ESP(0:2)) == "esp(0,1,2)"

        # MPLE against MPLE: the expansion of esp(0:1) into two design columns
        fit = fit_ergm(flo, [Edges(), ESP(0:1)]; method=:mple)
        @test fit.model.formula.terms.names == g.values["esp_mple_terms"]
        @test check_golden(g, "esp_mple_coefficients", fit.coefficients) ||
              error(golden_report(g, "esp_mple_coefficients", fit.coefficients))
        @test check_golden(g, "esp_mple_std_errors", fit.std_errors) ||
              error(golden_report(g, "esp_mple_std_errors", fit.std_errors))

        # Change statistics are exact against brute force for every count and
        # every type, on random graphs of several densities
        rng = Random.Xoshiro(20261007)
        for directed in (false, true), p in (0.2, 0.45)
            n = 9
            net = network(n; directed=directed)
            for i in 1:n, j in 1:n
                i != j && (directed || i < j) && rand(rng) < p && add_edge!(net, i, j)
            end
            for k in 0:4, ty in (directed ? (:OTP, :ITP, :OSP, :ISP) : (:OTP,))
                @test check_change_stats(ESP(k; type=ty), net) === nothing
            end
        end

        # The geometric weighting of the shared-partner distribution is GWESP
        w(α, k) = exp(α) * (1 - (1 - exp(-α))^k)
        for α in (0.0, 0.7)
            @test compute(GWESP(α), flo) ≈ sum(w(α, k) * compute(ESP(k), flo) for k in 0:14)
            for ty in (:OTP, :ITP, :OSP, :ISP)
                @test compute(GWESP(α; type=ty), sl) ≈
                      sum(w(α, k) * compute(ESP(k; type=ty), sl) for k in 0:16)
            end
        end

        # R's attainable range (minval 0, no maxval): esp3 = 0 on flomarriage
        # is at its bound, and R's drop fixes it at -Inf
        @test ERGM.Extension.attainable_range(ESP(1), flo) == (0.0, Inf)
        @test ERGM.Extension.extreme_statistics([Edges(), ESP(3)], flo) == [(2, :min)]
        f3 = @test_logs (:warn, r"esp3 are at their smallest attainable") match_mode=:any fit_ergm(
            flo, [Edges(), ESP(3)]; method=:mple)
        @test coef(f3)[2] == -Inf

        # Refusals: R's RTP type, negative or repeated counts, an empty list,
        # and a multi-count specification evaluated as one statistic
        @test_throws ArgumentError ESP(1; type=:RTP)
        @test_throws ArgumentError ESP(1; type=:union)
        @test_throws ArgumentError ESP(-1)
        @test_throws ArgumentError ESP([1, 1])
        @test_throws ArgumentError ESP(Int[])
        @test_throws ArgumentError compute(ESP(0:1), flo)
        @test_throws ArgumentError change_stat(ESP(0:1), flo, 1, 2)
        # Traits: dyad-dependent, no attribute, no direction requirement
        @test is_dyad_dependent(ESP(1)) && !requires_directed(ESP(1)) &&
              !requires_undirected(ESP(1))
        @test ESP(0:2) == ESP([0, 1, 2]) && ESP(1) != ESP(1; type=:ITP)
    end

    @testset "Golden fixture: offset() terms (fixed coefficients; provenanced)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "offset_ergm.toml"))
        fmh = load_dataset(:faux_mesa_high)
        flo = florentine_marriage()
        # (a) Structural zeros: grade 7 has no tie to grades 8 and 9, and the
        # model says it cannot. Dyad-independent: the exact MLE, R's numbers
        ta = [Edges(), NodeMatch(:Grade), Offset(NodeMix(:Grade, 7, 8), -Inf),
              Offset(NodeMix(:Grade, 7, 9), -Inf)]
        fa = fit_ergm(fmh, ta)
        @test fa.model.formula.terms.names == g.values["a_terms"]
        for (k, v) in (("a_coefficients", coef(fa)), ("a_std_errors", stderror(fa)),
                       ("a_loglik", loglikelihood(fa)), ("a_aic", aic(fa)), ("a_bic", bic(fa)))
            @test check_golden(g, k, v) || error(golden_report(g, k, v))
        end
        @test dof(fa) == g.values["a_df"] && nobs(fa) == g.values["a_nobs"]
        @test is_exact(fa) && fa.converged
        @test occursin("fixed by offset and not estimated", sprint(show, fa))
        @test confint(fa)[3, :] == [-Inf, -Inf]
        # The samplers never add a forbidden tie
        forbidden(s) = count(e -> Set((get_vertex_attribute(fmh, :Grade, src(e)),
                                       get_vertex_attribute(fmh, :Grade, dst(e)))) in
                                  (Set((7, 8)), Set((7, 9))), edges(s))
        @test all(==(0), forbidden.(simulate_ergm(fa; n_sim=6, rng=Random.Xoshiro(1))))
        # (b) A finite offset
        fb = fit_ergm(fmh, [Offset(Edges(), -5.0), NodeMatch(:Grade)])
        @test fb.model.formula.terms.names == g.values["b_terms"]
        for (k, v) in (("b_coefficients", coef(fb)), ("b_std_errors", stderror(fb)),
                       ("b_loglik", loglikelihood(fb)), ("b_aic", aic(fb)), ("b_bic", bic(fb)))
            @test check_golden(g, k, v) || error(golden_report(g, k, v))
        end
        @test dof(fb) == g.values["b_df"]
        @test check_golden(g, "b_exact_coefficient", coef(fb)[2])
        @test check_golden(g, "b_exact_std_error", stderror(fb)[2])
        # (c) A dyad-dependent model by MCMLE, against R's eleven-seed mean
        fc = fit_ergm(flo, [Offset(Edges(), -1.7), GWESP(0.5)]; method=:mcmle,
                      rng=Random.Xoshiro(20261002))
        @test fc.converged && coef(fc)[1] == -1.7 && stderror(fc)[1] == 0
        @test all(abs.(coef(fc) .- Float64.(g.values["c_coef_mean"])) .<=
                  Float64.(g.values["c_coef_tolerance"]))
        @test all(abs.(stderror(fc) .- Float64.(g.values["c_se_mean"])) .<=
                  Float64.(g.values["c_se_tolerance"]))
        @test dof(fc) == 1 && ERGM._mcmc_percent(fc)[1] == 0
        @test isfinite(loglikelihood(fc))
        # ... and with structural zeros in an MCMLE (the bridge keeps the -Inf
        # offsets in its exact reference)
        fd = fit_ergm(fmh, [ta[1:2]; GWESP(0.25); ta[3:4]]; method=:mcmle, n_chains=4,
                      rng=Random.Xoshiro(2))
        @test fd.converged && isfinite(loglikelihood(fd))
        @test all(==(0), forbidden.(simulate_ergm(fd; n_sim=4, rng=Random.Xoshiro(3))))
        # The bootstrap holds an offset fixed (zero variance)
        bt = fit_ergm(flo, [Offset(Edges(), -1.7), GWESP(0.5)]; method=:mple, se=:bootstrap, n_boot=20,
                      rng=Random.Xoshiro(4))
        @test stderror(bt)[1] == 0 && isfinite(stderror(bt)[2]) && bt.p_values[2] > 0
        # A multi-statistic term takes one coefficient per statistic (or one)
        o = fit_ergm(fmh, [Edges(), Offset(NodeFactor(:Race), [-1.0, -2.0, -3.0, -4.0])])
        @test o.model.formula.terms.names[2] == "offset(nodefactor.Race.Hisp)"
        @test coef(o)[2:5] == [-1.0, -2.0, -3.0, -4.0]
        @test_throws ArgumentError fit_ergm(fmh, [Edges(), Offset(NodeFactor(:Race), [-1.0, -2.0])])
        # (d) +Inf: forced ties (group A is a complete clique). R's numbers
        dn = network(10; directed=false)
        for (i, j) in zip(g.values["d_tails"], g.values["d_heads"])
            add_edge!(dn, Int(i), Int(j))
        end
        set_vertex_attribute!(dn, :g, Dict(v => (v <= 4 ? "A" : "B") for v in 1:10))
        set_vertex_attribute!(dn, :x, Dict(v => Float64(v) for v in 1:10))
        forced = Offset(NodeMatch(:g; diff=true, level="A"), Inf)
        fdd = fit_ergm(dn, [Edges(), NodeCov(:x), forced])
        @test fdd.model.formula.terms.names == g.values["d_terms"]
        for (k, v) in (("d_coefficients", coef(fdd)), ("d_std_errors", stderror(fdd)),
                       ("d_loglik", loglikelihood(fdd)), ("d_aic", aic(fdd)), ("d_bic", bic(fdd)))
            @test check_golden(g, k, v) || error(golden_report(g, k, v))
        end
        @test dof(fdd) == g.values["d_df"]
        clique(s) = all(has_edge(s, i, j) for i in 1:4 for j in (i + 1):4)
        @test all(clique, simulate_ergm(fdd; n_sim=6, rng=Random.Xoshiro(5)))
        # ... and the exact conditional normaliser agrees with enumeration over
        # the 39 free dyads' worth of structure, checked on the MCMLE's exact
        # dyad-independent log-likelihood (= the MPLE's)
        mcd = fit_ergm(dn, [Edges(), NodeCov(:x), forced]; method=:mcmle, n_samples=300,
                       rng=Random.Xoshiro(6))
        @test loglikelihood(mcd) ≈ loglikelihood(fdd) atol = 1e-8

        # An infinite offset on a DYAD-DEPENDENT statistic is a state-dependent
        # constraint: a triangle-free network modelled as triangle-free
        tf = network(12; directed=false)
        for (i, j) in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
                       (9, 10), (10, 11), (11, 12), (1, 6), (3, 9)]
            add_edge!(tf, i, j)
        end
        ft = fit_ergm(tf, [Edges(), Kstar(2), Offset(Triangle(), -Inf)]; method=:mcmle,
                      rng=Random.Xoshiro(3))
        @test ft.converged && all(isfinite, coef(ft)[1:2])
        @test isnan(loglikelihood(ft))
        @test occursin("an infinite offset on a dyad-dependent statistic", sprint(show, ft))
        @test any(occursin("log-likelihood not estimated", a) for a in approximations(ft))
        @test all(s -> compute(Triangle(), s) == 0,
                  simulate_ergm(ft; n_sim=6, rng=Random.Xoshiro(4)))
        # Exactness of the constrained sampler: 4 vertices, triangle-free graphs
        # only, against enumeration of the edge-count distribution
        m4 = ERGMModel(ERGMFormula([Edges(), Offset(Triangle(), -Inf)]), network(4; directed=false))
        out = mh_sample(m4, [-0.3, -Inf]; n_samples=200_000, burnin=1000, interval=1,
                        rng=Random.Xoshiro(7))
        @test all(==(0), out.stats[:, 2])
        dy4 = [(i, j) for i in 1:4 for j in (i + 1):4]
        wts = zeros(7)
        for msk in 0:63
            g4 = network(4; directed=false)
            for (k, (i, j)) in enumerate(dy4)
                (msk >> (k - 1)) & 1 == 1 && add_edge!(g4, i, j)
            end
            compute(Triangle(), g4) == 0 && (wts[count_ones(msk) + 1] += exp(-0.3 * count_ones(msk)))
        end
        wts ./= sum(wts)
        emp = [mean(out.stats[:, 1] .== e) for e in 0:6]
        @test maximum(abs.(emp .- wts)) < 0.01

        # Offsets with missing-data maximum likelihood
        masked = copy(flo); set_missing_dyad!(masked, 1, 9); set_missing_dyad!(masked, 3, 4)
        fm = mcmle(ERGMModel(ERGMFormula([Offset(Edges(), -1.7), GWESP(0.5)]), masked);
                   missing=:mle, rng=Random.Xoshiro(5))
        @test fm.converged && coef(fm)[1] == -1.7 && stderror(fm)[1] == 0
        @test abs(coef(fm)[2] - coef(fc)[2]) < 0.1 && isfinite(loglikelihood(fm))

        # Refused, in words: a network the constraint gives probability 0
        @test_throws ArgumentError Offset(Edges(), NaN)
        e = try fit_ergm(fmh, [Edges(), Offset(NodeMatch(:Sex; diff=true, level="F"), -Inf)])
            catch err; err end
        @test e isa ArgumentError && occursin("probability 0", e.msg)          # observed F-F ties
        e = try fit_ergm(flo, [Edges(), Offset(Triangle(), -Inf)]; method=:mple) catch err; err end
        @test e isa ArgumentError && occursin("probability 0", e.msg)          # flo has triangles
        e = try fit_ergm(flo, [Offset(Edges(), Inf), NodeCov(:wealth)]) catch err; err end
        @test e isa ArgumentError && occursin("lacks a tie", e.msg)
        e = try fit_ergm(dn, [Edges(), forced, Offset(NodeCov(:x), -Inf)]) catch err; err end
        @test e isa ArgumentError
        @test Offset(Edges(), -1.0) == Offset(Edges(), -1.0)
        @test name(Offset(GWESP(0.5), 0.2), network(3; directed=true)) ==
              "offset(gwesp.OTP.fixed.0.5)"
    end

    @testset "Curved terms: the decay-derivative statistics" begin
        # D_α = ∂g_α/∂α against a central finite difference of the statistic,
        # and its change statistic against brute force — every GWESP type
        sl = samplike()
        flo = florentine_marriage()
        for (net, types) in ((flo, (:OTP,)), (sl, (:OTP, :ITP, :OSP, :ISP)))
            for ty in types, α in (0.0, 0.3, 1.2)
                d = ERGM._GWESPDecayScore(α, ty)
                h = 1e-5
                fd = α == 0 ?
                    (compute(GWESP(h; type=ty), net) - compute(GWESP(0.0; type=ty), net)) / h :
                    (compute(GWESP(α + h; type=ty), net) - compute(GWESP(α - h; type=ty), net)) / 2h
                @test compute(d, net) ≈ fd atol = (α == 0 ? 1e-2 : 1e-5) * max(1, abs(fd))
                @test check_change_stats(d, net) === nothing
            end
        end
        for α in (0.0, 0.5, 2.0)
            d = ERGM._GWDegreeDecayScore(α)
            h = 1e-5
            α > 0 && @test compute(d, flo) ≈
                (compute(GWDegree(α + h), flo) - compute(GWDegree(α - h), flo)) / 2h atol = 1e-5
            @test check_change_stats(d, flo) === nothing
        end
        # The kernel refactor left the fixed-decay statistic bit-identical
        @test compute(GWESP(0.5), flo) == 8.393469340287368

        # `fixed=false` is the curved term, R's labels, and not a statistic
        @test GWESP(0.25; fixed=false) isa CurvedGWESP
        @test GWESP(0.25; fixed=false) == CurvedGWESP(0.25)
        @test GWDegree(0.5; fixed=false) isa CurvedGWDegree
        @test GWESP(0.25; fixed=true) isa GWESP && GWDegree(0.5) isa GWDegree
        @test name(CurvedGWESP(0.25), flo) == "gwesp" && name(CurvedGWESP(0.25), sl) == "gwesp.OTP"
        @test name(CurvedGWESP(0.25; type=:ITP), sl) == "gwesp.ITP"
        @test name(CurvedGWDegree()) == "gwdegree"
        @test_throws ArgumentError CurvedGWESP(0.5; type=:union)
        @test_throws ArgumentError compute(CurvedGWESP(0.25), flo)
        @test_throws ArgumentError change_stat(CurvedGWDegree(0.5), flo, 1, 2)
        @test_throws ArgumentError summary_stats(flo, [GWESP(0.25; fixed=false)])
        e = try fit_ergm(flo, [Edges(), GWESP(0.25; fixed=false)]; method=:mple) catch err; err end
        @test e isa ArgumentError && occursin("method=:mcmle", e.msg)
        @test_throws ArgumentError fit_ergm(sl, [Edges(), GWDegree(0.5; fixed=false)];
                                            method=:mcmle)             # undirected only
        masked = copy(flo); set_missing_dyad!(masked, 1, 9)
        @test_throws ArgumentError fit_ergm(masked, [Edges(), GWESP(0.25; fixed=false)];
                                            method=:mcmle, missing=:mle)
        @test_throws ArgumentError fit_ergm(flo, [Offset(Edges(), -1.7), GWESP(0.25; fixed=false)];
                                            method=:mcmle)
    end

    @testset "Golden fixture: statnet ergm curved gwesp on faux.mesa.high (provenanced)" begin
        g = load_golden(joinpath(@__DIR__, "fixtures", "curved_ergm.toml"))
        fmh = load_dataset(:faux_mesa_high)
        terms = [Edges(), NodeMatch(:Grade), NodeMatch(:Race), GWESP(0.25; fixed=false)]
        fit = fit_ergm(fmh, terms; method=:mcmle, proposal=:spdyad, effective_size=64,
                       n_chains=4, rng=Random.Xoshiro(20261002))
        @test fit.converged
        @test fit.model.formula.terms.names == g.values["terms"]
        @test all(abs.(coef(fit) .- Float64.(g.values["coef_mean"])) .<
                  Float64.(g.values["coef_tolerance"]))
        @test all(abs.(stderror(fit) .- Float64.(g.values["se_mean"])) .<
                  Float64.(g.values["se_tolerance"]))
        @test abs(loglikelihood(fit) - g.values["loglik_mean"]) <
              golden_tolerance(g, "loglik_sd_multiple") * g.values["loglik_seed_sd"]
        # The decay moved from its start to R's estimate
        @test abs(coef(fit)[5] - 0.25) > 0.08
        # The result is an ordinary fit: StatsAPI, diagnostics, simulation, GOF
        @test dof(fit) == 5 && length(stderror(fit)) == 5
        @test all(values(NetworkCore.check_statsapi(fit; strict=true)))
        @test size(confint(fit)) == (5, 2)
        @test mcmc_diagnostics(fit).term_names == g.values["terms"]
        sims = simulate_ergm(fit; n_sim=4, proposal=:spdyad, rng=Random.Xoshiro(1))
        @test length(sims) == 4
        @test abs(mean(ne.(sims)) - ne(fmh)) < 60
        @test n_simulations(gof(fit; n_sim=12, stats=[:esp], proposal=:spdyad,
                                rng=Random.Xoshiro(2))) == 12
        # The simulated model IS the fixed-decay model at the fitted decay
        fixed = ERGMModel(ERGMFormula([terms[1:3]; GWESP(coef(fit)[5])]), fmh)
        a = mh_sample(fit.model, coef(fit); n_samples=5, rng=Random.Xoshiro(3))
        b = mh_sample(fixed, coef(fit)[1:4]; n_samples=5, rng=Random.Xoshiro(3))
        @test a.stats[:, 1:4] ≈ b.stats && all(==(0), a.stats[:, 5])
    end

    @testset "MPLE with a large finite offset reaches the finite maximum" begin
        # From a zero start a large offset saturates the fitted probabilities
        # and Newton ran off to a flat asymptote ("The MPLE does not exist",
        # triangle = −51) although the concave pseudo-likelihood has an
        # interior maximum. The start is now reached by continuation.
        rng = Random.Xoshiro(11)
        n = 60
        net = network(n; directed=false)
        for i in 1:n, j in (i + 1):n
            rand(rng) < 0.05 && add_edge!(net, i, j)
        end
        for c in 1:15                                  # some closed triads
            a, b, d = 3c, 3c + 1, 3c + 2
            add_edge!(net, a, b); add_edge!(net, b, d); add_edge!(net, a, d)
        end
        for c in (0.5, 1.5, 2.5)
            model = ERGMModel(ERGMFormula([Edges(), Triangle(), Offset(GWESP(0.0), c)]), net)
            fit = @test_logs mple(model; se=:hessian)       # silent: no asymptote warning
            @test fit.converged && coef(fit)[3] == c
            # At the maximum the pseudo-score of the free coefficients vanishes
            ts = model.formula.terms
            score = zeros(2)
            for i in 1:n, j in (i + 1):n
                d = change_stat_all(ts, net, i, j)
                pr = 1 / (1 + exp(-dot(coef(fit), d)))
                score .+= (has_edge(net, i, j) - pr) .* d[1:2]
            end
            @test maximum(abs.(score)) < 1e-5
            @test abs(coef(fit)[2]) < 20
        end
    end

    @testset "A mis-declared dyad-independent user term is refused at ERGMModel" begin
        sl = samplike()
        e = try ERGMModel(ERGMFormula([Edges(), MisdeclaredMutual()]), sl) catch err; err end
        @test e isa ArgumentError
        @test occursin("badmutual", e.msg) && occursin("MisdeclaredMutual", e.msg)
        @test occursin("is_dyad_dependent", e.msg)
        @test_throws ArgumentError fit_ergm(sl, [Edges(), MisdeclaredMutual()])
        @test_throws ArgumentError fit_ergm(sl, [Edges(), Offset(MisdeclaredMutual(), 0.3)])
        # Correctly declared user terms pass: independent ones are probed and
        # accepted, dependent ones are not probed at all
        flo = florentine_marriage()
        @test fit_ergm(flo, [Edges(), ExampleTerm()]).converged
        @test ERGM._needs_independence_probe(ExampleTerm())
        @test !ERGM._needs_independence_probe(Edges())               # built-in: never
        @test !ERGM._needs_independence_probe(NodeCov(:wealth))
        @test !ERGM._needs_independence_probe(ThrowingTerm())        # declared dependent
    end

    @testset "constraints= is refused in words on every entry point" begin
        flo = florentine_marriage()
        for c in ([:edges], :edges, [FakeConstraint()])
            e = try fit_ergm(flo, [Edges()]; constraints=c) catch err; err end
            @test e isa ArgumentError && occursin("constraints are not implemented", e.msg)
        end
        @test fit_ergm(flo, [Edges()]; constraints=Symbol[]).converged     # empty: harmless
        @test !Base.isexported(ERGM, :ConstraintTerm)
    end

    @testset "GOF thinning is ESS-aware" begin
        flo = florentine_marriage()
        fit = fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mcmle, n_samples=500,
                       bridge_rungs=0, rng=Random.Xoshiro(1))
        # An interval of 1 toggle makes successive draws near-copies: warned
        @test_logs (:warn, r"strongly autocorrelated") match_mode=:any gof(
            fit; n_sim=40, n_chains=1, interval=1, stats=[:degree], rng=Random.Xoshiro(2))
        # The default interval, adapted to the measured ESS, does not warn
        @test_logs gof(fit; n_sim=40, n_chains=1, stats=[:degree], rng=Random.Xoshiro(2))
        # Chains shorter than ten draws are not measured
        @test isnan(ERGM._gof_ess(fit.model, simulate_ergm(fit; n_sim=8, rng=Random.Xoshiro(3)),
                                  8, 4))
    end

    @testset "GOF distance panel on a Network{Int32} (unreachable-pair sentinel)" begin
        # `gdistances` on an Int32 network marks an unreachable vertex with
        # typemax(Int32); compared against typemax(Int) every unreachable pair
        # used to count as a distance of 2^31 - 1, and the panel's histogram had a
        # histogram of that length (tens of GB). The counts are checked first,
        # on their own, so a regression fails here without sizing anything.
        flo = florentine_marriage()
        for directed in (false, true)
            n = nv(flo)
            n32 = Network{Int32,directed}(n)
            n64 = network(n; directed=directed)
            for e in edges(flo)
                add_edge!(n32, Int32(src(e)), Int32(dst(e)))
                add_edge!(n64, src(e), dst(e))
            end
            c32, u32 = ERGM._distance_counts(n32)
            c64, u64 = ERGM._distance_counts(n64)
            @test u32 == u64
            directed || @test u32 == 15                 # Pucci: R's obs.dist Inf
            @test maximum(keys(c32)) < n
            @test c32 == c64
        end
        n32 = Network{Int32,false}(nv(flo))
        for e in edges(flo)
            add_edge!(n32, Int32(src(e)), Int32(dst(e)))
        end
        @test eltype(Graphs.gdistances(n32, 1)) == Int32
        # Only once the counts are right is a whole gof run: it matches the
        # Int64 network's observed panel exactly
        if maximum(keys(first(ERGM._distance_counts(n32)))) < nv(n32)
            fit32 = fit_ergm(n32, [Edges()])
            g32 = gof(fit32; n_sim=4, burnin=10, interval=1, stats=[:distance],
                      rng=Random.Xoshiro(5))
            g64 = gof(fit_ergm(flo, [Edges()]); n_sim=4, burnin=10, interval=1,
                      stats=[:distance], rng=Random.Xoshiro(5))
            d32 = only(g32.statistics); d64 = only(g64.statistics)
            @test d32.observed[end] == 15.0
            @test sum(d32.observed) == 120
            # flomarriage's largest finite distance is 5 (R's obs.dist)
            @test d32.observed[1:5] == d64.observed[1:5] == [20, 35, 32, 15, 3]
            @test d32.observed[end] == d64.observed[end]
            @test length(d32.labels) < 20
        end
    end

    @testset "Aqua" begin
        # Ambiguities are checked on ERGM alone above (Test.detect_ambiguities);
        # Aqua's version also walks the dependencies' methods
        Aqua.test_all(ERGM; ambiguities=false)
    end

    @testset "Every exported docstring carries a runnable example" begin
        # Every export has a docstring with a runnable
        # example. A docs build with checkdocs=:exports checks presence, not
        # content, so walk the docsystem: every ERGM-owned docstring of an
        # exported binding — including the ones ERGM attaches to the shared
        # NetworkCore/StatsAPI generics (`compute`, `gof`, `coef`, ...) — must
        # contain a fenced ```julia block, and every such block must RUN in a
        # fresh module that has done nothing but `using ERGM` (so an example
        # that needs `Random` or `Statistics` says so itself). Names ERGM
        # merely re-exports from NetworkCore.jl (`Network`, `add_edge!`, ...) are
        # documented there and only need to be documented somewhere. Sketches
        # that are deliberately not standalone programs use a ```jl fence.
        # Mirrors REM.jl's testset of the same name.
        # The `ERGM.Extension` API is held to the same rule: its generic
        # functions are owned by the submodule, their docstrings sit on
        # ERGM's methods (so they live in ERGM's docsystem meta) and the
        # module's own docstring in the submodule's.
        meta_ergm = Base.Docs.meta(ERGM)
        meta_ext = Base.Docs.meta(ERGM.Extension)
        documented_elsewhere(b) = any(haskey(Base.Docs.meta(m), b)
                                      for m in (NetworkCore, Graphs, StatsAPI))
        undocumented = String[]
        missing_example = String[]
        blocks = Tuple{String,String}[]
        bindings = vcat([(ERGM, nm) for nm in names(ERGM) if nm !== :ERGM],
                        [(ERGM.Extension, nm) for nm in names(ERGM.Extension)
                         if nm !== :Extension])
        for (mod, nm) in bindings
            b = Base.Docs.Binding(mod, nm)
            metas = [m for m in (meta_ergm, meta_ext) if haskey(m, b)]
            if isempty(metas)
                documented_elsewhere(b) || push!(undocumented, string(nm))
                continue
            end
            has_example = false
            for meta in metas, (_, ds) in meta[b].docs
                txt = ds.text isa AbstractString ? ds.text : join(string.(ds.text), "\n")
                for m in eachmatch(r"```julia\n(.*?)```"s, txt)
                    has_example = true
                    push!(blocks, (string(nm), String(m.captures[1])))
                end
                occursin("```jldoctest", txt) && (has_example = true)
            end
            has_example || push!(missing_example, string(nm))
        end
        @test isempty(undocumented)
        @test isempty(missing_example)
        # ERGM-owned docstrings sit on the foreign generics it extends
        for nm in (:coef, :stderror, :vcov, :confint, :coeftable, :gof, :compute, :name)
            @test haskey(meta_ergm, Base.Docs.Binding(ERGM, nm))
        end
        @test length(blocks) >= 40
        # Every block runs, statement by statement, and every `# value`
        # comment is compared with the value the statement returns: a
        # docstring cannot quote a number its own code does not produce
        # (NodeFactor/NodeMatch/NodeMix once stated 20/10/10 where the code
        # and R give 23/7/13)
        n_claims = 0
        wrong = String[]
        for (nm, code) in blocks
            m = Module(Symbol("DocExample_", nm))
            ok = try
                Core.eval(m, :(using ERGM))
                c, bad = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                    check_block(m, code; where="docstring of $nm")
                end
                n_claims += c
                append!(wrong, bad)
                true
            catch err
                println(stderr, "docstring example of $nm failed: ", sprint(showerror, err))
                false
            end
            @test ok
        end
        isempty(wrong) || println(stderr, join(wrong, "\n"))
        @test isempty(wrong)
        @test n_claims >= 100
    end

end
