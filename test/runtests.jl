using ERGM
using LinearAlgebra
using Networks
using Random
using Statistics
using StatsBase
using Test

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
                     GWESP(0.5), GWESP(1.2), GWDegree(0.5), GWDegree(1.2),
                     Degree(0), Degree(1), Degree(2), Degree(3),
                     GWDSP(0.5), GWDSP(1.2)]
            @test check_change_stats(term, net) === nothing
        end
    end

    @testset "Structural terms match brute force (directed)" begin
        net = fixture_directed()
        for term in [Edges(), Mutual(), Triangle(), Kstar(2), TwoPath(),
                     GWESP(0.5), GWESP(0.5; type=:ITP), GWESP(0.5; type=:OSP),
                     GWESP(0.5; type=:ISP), GWESP(0.5; type=:union),
                     GWESP(1.2; type=:ITP), GWDegree(0.5),
                     IDegree(0), IDegree(1), IDegree(2), ODegree(1), ODegree(2),
                     GWIDegree(0.5), GWIDegree(1.2), GWODegree(0.5),
                     GWDSP(0.5), GWDSP(0.5; type=:ITP), GWDSP(0.5; type=:OSP),
                     GWDSP(0.5; type=:ISP), GWDSP(0.5; type=:union),
                     GWDSP(1.2; type=:OSP)]
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

        # Names: OTP keeps statnet's directed default name; the union
        # variant is named so it cannot be confused with statnet's
        @test ERGM.name(GWESP(0.5)) == "gwesp.fixed.0.5"
        @test ERGM.name(GWESP(0.5; type=:OTP)) == "gwesp.fixed.0.5"
        @test ERGM.name(GWESP(0.5; type=:ITP)) == "gwesp.ITP.fixed.0.5"
        @test ERGM.name(GWESP(0.5; type=:union)) == "gwesp.union.fixed.0.5"
        @test_throws ArgumentError GWESP(0.5; type=:XYZ)

        # Undirected networks ignore the type: all variants agree
        unet = fixture_undirected()
        base = compute(GWESP(0.5), unet)
        for t in (:OTP, :ITP, :OSP, :ISP, :union)
            @test compute(GWESP(0.5; type=t), unet) ≈ base
        end
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

        # Vectors of degrees produce one term per degree, as in statnet
        @test Degree(0:2) == [Degree(0), Degree(1), Degree(2)]
        @test IDegree([1, 3]) == [IDegree(1), IDegree(3)]
        @test ODegree(1:2) == [ODegree(1), ODegree(2)]
        model = ERGMModel(ERGMFormula([Edges(); Degree(0:2)]), net)
        @test model.formula.terms.names == ["edges", "degree0", "degree1", "degree2"]
        @test compute_all(model.formula.terms, net) == [3.0, 1.0, 2.0, 2.0]

        @test_throws ArgumentError Degree(-1)

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
        @test_throws ArgumentError GWODegree(0.0)
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
        # 4→2, 1→2. By hand:
        #   OTP (ordered dyads): only (1,2) has shared partners, two of them
        #   ITP (ordered dyads): only (2,1) — the same two-paths read
        #     backwards, so the statistic equals OTP's (statnet: OTP and ITP
        #     are equivalent for DSP)
        #   OSP (unordered dyads): {1,3}, {1,4}, {3,4} share one out-partner
        #   ISP (unordered dyads): {2,3}, {2,4}, {3,4} share one in-partner
        #   union: {1,2} and {3,4} have two, the four tied dyads one each
        netA = network(5; directed=true)
        for (i, j) in [(1, 3), (3, 2), (1, 4), (4, 2), (1, 2)]
            add_edge!(netA, i, j)
        end
        for α in (0.5, 1.2)
            @test compute(GWDSP(α), netA) ≈ w2(α)                    # OTP default
            @test compute(GWDSP(α; type=:OTP), netA) ≈ w2(α)
            @test compute(GWDSP(α; type=:ITP), netA) ≈ w2(α)
            @test compute(GWDSP(α; type=:OSP), netA) ≈ 3.0
            @test compute(GWDSP(α; type=:ISP), netA) ≈ 3.0
            @test compute(GWDSP(α; type=:union), netA) ≈ 4.0 + 2.0 * w2(α)
        end

        # Names follow the GWESP convention
        @test ERGM.name(GWDSP(0.5)) == "gwdsp.fixed.0.5"
        @test ERGM.name(GWDSP(0.5; type=:OTP)) == "gwdsp.fixed.0.5"
        @test ERGM.name(GWDSP(0.5; type=:OSP)) == "gwdsp.OSP.fixed.0.5"
        @test ERGM.name(GWDSP(0.5; type=:union)) == "gwdsp.union.fixed.0.5"
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

        undirected_terms() = [Triangle(), GWESP(0.5), GWESP(1.2),
                              GWDegree(0.7), Kstar(2), TwoPath(),
                              Degree(0), Degree(2), Degree(3),
                              GWDSP(0.5), GWDSP(1.2)]
        directed_terms() = [Triangle(), GWESP(0.5), GWESP(0.5; type=:ITP),
                            GWESP(0.5; type=:OSP), GWESP(0.5; type=:ISP),
                            GWESP(0.5; type=:union), GWESP(1.2; type=:OSP),
                            GWDegree(0.7), TwoPath(),
                            IDegree(0), IDegree(2), ODegree(2),
                            GWIDegree(0.7), GWODegree(0.7),
                            GWDSP(0.5), GWDSP(0.5; type=:ITP),
                            GWDSP(0.5; type=:OSP), GWDSP(0.5; type=:ISP),
                            GWDSP(0.5; type=:union), GWDSP(1.2; type=:OSP)]

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

        # diff=true without a level, or a level without diff=true, is an error
        @test_throws ArgumentError NodeMatch(:group; diff=true)
        @test_throws ArgumentError NodeMatch(:group; level="A")
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

            terms = AbstractERGMTerm[Edges(), Triangle(), Kstar(2), Kstar(3),
                                     TwoPath(), GWESP(0.5), GWDegree(0.5),
                                     NodeFactor(:group),
                                     NodeFactor(:group; level="A"),
                                     NodeCov(:age), NodeMatch(:group),
                                     NodeMatch(:group; diff=true, level="A"),
                                     NodeMismatch(:group),
                                     AbsDiff(:age), EdgeCov(cov_matrix)]
            if is_directed(net)
                push!(terms, Mutual())
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

    @testset "Golden master vs R ergm (Florentine families)" begin
        flo = florentine_marriage()
        @test nv(flo) == 16
        @test ne(flo) == 20
        @test compute(Triangle(), flo) == 3.0  # summary(flomarriage ~ triangle)

        # R: ergm(flomarriage ~ edges)
        #    edges = -1.609438 (SE 0.244949)
        r1 = fit_ergm(flo, [Edges()])
        @test r1.coefficients[1] ≈ -1.609438 atol = 1e-3
        @test r1.std_errors[1] ≈ 0.244949 atol = 1e-2

        # R: ergm(flomarriage ~ edges + nodecov("wealth"))
        #    edges = -2.594929 (SE 0.536056), nodecov.wealth = 0.010546 (SE 0.004674)
        r2 = fit_ergm(flo, [Edges(), NodeCov(:wealth)])
        @test r2.coefficients[1] ≈ -2.594929 atol = 1e-3
        @test r2.coefficients[2] ≈ 0.010546 atol = 1e-4
        @test r2.std_errors[1] ≈ 0.536056 atol = 1e-2
        @test r2.std_errors[2] ≈ 0.004674 atol = 1e-3

        # vcov is the full covariance matrix, consistent with the SEs
        V = vcov(r2)
        @test size(V) == (2, 2)
        @test sqrt.([V[1, 1], V[2, 2]]) ≈ r2.std_errors atol = 1e-8
        @test V[1, 2] ≈ V[2, 1] atol = 1e-12
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

    @testset "Golden master: MCMLE matches R statnet (flomarriage ~ edges)" begin
        Random.seed!(42)
        flo = florentine_marriage()

        result = fit_ergm(flo, [Edges()]; method=:mcmle, n_samples=600)
        # MCMLE must recover the exact MLE logit(density):
        # log((20/120) / (100/120)) = log(20/100) = -1.609438,
        # matching R: ergm(flomarriage ~ edges), edges = -1.609438
        @test result.method == :mcmle
        @test result.converged
        @test result.coefficients[1] ≈ -1.609438 atol = 0.2
        @test !isnothing(result.mcmc_samples)
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

    @testset "MCMLE deprecated tol keyword still accepted" begin
        Random.seed!(5)
        net = network(6; directed=false)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        add_edge!(net, 3, 4)

        result = @test_logs (:warn, r"deprecated") match_mode = :any fit_ergm(
            net, [Edges()]; method=:mcmle, tol=1e-4, n_samples=100, max_iter=2)
        @test result.method == :mcmle
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

        # gof extends Networks.jl's shared generic and returns the shared
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

        # Backward compatibility: the pre-v0.5 private names are aliases of the
        # SAME generics, so downstream methods declared on them (TERGM.jl ships
        # `ERGM._requires_directed(::Delrecip) = true`) still drive validation
        @test ERGM._requires_directed === requires_directed
        @test ERGM._requires_undirected === requires_undirected
        @test ERGM._vertex_attribute(NodeCov(:age)) === :age
        @test ERGM._vertex_attribute(Edges()) === nothing

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
        mfit = fit_ergm(flo, [Edges()]; method=:mcmle, n_samples=200, max_iter=3)
        diag = mcmc_diagnostics(mfit)
        @test length(diag.autocorrelation) == 1
        @test diag.n_samples == 200

        # Geyer initial-sequence ESS and Geweke convergence fields
        @test length(diag.ess_geyer) == 1
        @test 2.0 <= diag.ess_geyer[1] <= 2 * diag.n_samples
        @test length(diag.geweke_z) == 1
        @test isfinite(diag.geweke_z[1])
        @test 0.0 < diag.geweke_p[1] <= 1.0
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
        @test ERGM._z_pvalues([ERGM._geweke_z(drift)])[1] < 1e-6

        # Constant chain degrades gracefully
        @test ERGM._geyer_ess(fill(3.0, 100)) == 100.0
        @test ERGM._geweke_z(fill(3.0, 100)) == 0.0
    end

    @testset "P-values use ccdf and never underflow to zero for finite z" begin
        # 2(1 − cdf) underflows to exactly 0 at |z| ≈ 8.3; ccdf is accurate
        @test ERGM._z_pvalues([10.0])[1] > 0.0
        @test ERGM._z_pvalues([10.0])[1] ≈ 1.5239706048320995e-23
        @test ERGM._z_pvalues([30.0])[1] > 0.0

        # Beyond |z| ≈ 38 even the ccdf tail underflows Float64, so finite
        # z-statistics are floored at floatmin — a finite estimate never
        # reports an exact-zero p-value
        @test ERGM._z_pvalues([40.0])[1] > 0.0
        @test ERGM._z_pvalues([40.0])[1] == floatmin(Float64)

        # Sanity at the center and edge cases
        @test ERGM._z_pvalues([0.0])[1] ≈ 1.0
        @test ERGM._z_pvalues([-2.0]) == ERGM._z_pvalues([2.0])
        @test isnan(ERGM._z_pvalues([NaN])[1])
        @test ERGM._z_pvalues([Inf])[1] == 0.0
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
        f1 = fit_ergm(flo, [Edges()]; method=:mcmle, n_samples=200, max_iter=3,
                      rng=Random.Xoshiro(1234), bridge_rungs=4, bridge_samples=100)
        f2 = fit_ergm(flo, [Edges()]; method=:mcmle, n_samples=200, max_iter=3,
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

        # Invalid se choice fails loudly
        @test_throws ArgumentError fit_ergm(flo, [Edges()]; se=:jackknife)
    end

    @testset "show() prints a pseudo-likelihood caveat only under dyad dependence" begin
        flo = florentine_marriage()

        # Dyad-dependent formula + MPLE => caveat
        dep_fit = fit_ergm(flo, [Edges(), Triangle()])
        dep_out = sprint(show, dep_fit)
        @test occursin("pseudolikelihood", dep_out)
        @test occursin("suspect", dep_out)

        # The coefficient table renders through the shared Networks.jl
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
        boot_fit = fit_ergm(flo, [Edges(), Triangle()]; se=:bootstrap, n_boot=10,
                            boot_burnin=500, boot_interval=50,
                            rng=Random.Xoshiro(8))
        boot_out = sprint(show, boot_fit)
        @test occursin("parametric-bootstrap", boot_out)
        @test !occursin("suspect", boot_out)
    end

    @testset "newton_fit" begin
        # Quadratic objective: exact one-step maximum, exact vcov
        A = [2.0 0.5; 0.5 1.0]
        target = [1.0, -0.5]
        quad(θ) = (-0.5 * dot(θ - target, A * (θ - target)), -A * (θ - target), -A)
        fit = newton_fit(quad, [10.0, -10.0])
        @test fit.converged
        @test fit.θ ≈ target atol = 1e-8
        @test fit.vcov ≈ inv(A) atol = 1e-8
        @test fit.se ≈ sqrt.([inv(A)[1, 1], inv(A)[2, 2]]) atol = 1e-8
        @test fit.loglik ≈ 0.0 atol = 1e-12

        # Poisson log-mean: ll(θ) = kθ − e^θ, maximum at log k, SE 1/√k.
        # Started far away, so step halving must engage
        k = 7.0
        pois(θ) = (k * θ[1] - exp(θ[1]), [k - exp(θ[1])], hcat(-exp(θ[1])))
        pfit = newton_fit(pois, [8.0])
        @test pfit.converged
        @test pfit.θ[1] ≈ log(k) atol = 1e-6
        @test pfit.se[1] ≈ 1 / sqrt(k) atol = 1e-6
        @test pfit.loglik ≈ k * log(k) - k atol = 1e-8
        @test pfit.iterations >= 1

        # θ0 is not mutated
        θ0 = [8.0]
        newton_fit(pois, θ0)
        @test θ0 == [8.0]
    end

    # ------------------------------------------------------------------
    # The shared logistic derivatives (review finding 15)
    #
    # ERGMMulti's MPLE, TERGM's CMPLE and ERGMRank's swap MPLE are all THIS
    # likelihood, and all three used to carry their own copy of the loop with a
    # per-row `x * x'` outer product inside it. One builder, one workspace, one
    # allocation bound — and the bound is what stops the outer product coming
    # back.
    # ------------------------------------------------------------------
    @testset "logistic_derivatives" begin
        rng = MersenneTwister(11)
        n, p = 400, 3
        X = randn(rng, n, p)
        βtrue = [0.7, -0.4, 0.2]
        y = [rand(rng) < 1 / (1 + exp(-dot(βtrue, X[r, :]))) for r in 1:n]

        d = logistic_derivatives(X, y)
        β = [0.1, 0.0, -0.1]
        ll, grad, hess = d(β)

        # Against the textbook formulas, written out independently
        η = X * β
        pr = 1 ./ (1 .+ exp.(-η))
        @test ll ≈ sum(y[r] ? log(pr[r]) : log1p(-pr[r]) for r in 1:n) atol = 1e-9
        @test grad ≈ X' * (Float64.(y) .- pr) atol = 1e-9
        @test hess ≈ -X' * ((pr .* (1 .- pr)) .* X) atol = 1e-9
        @test issymmetric(round.(hess; digits=10))

        # It IS a logistic regression: recovers the coefficients
        fit = newton_fit(d, zeros(p))
        @test fit.converged
        @test fit.θ ≈ βtrue atol = 0.35
        # ...and at the optimum the gradient vanishes
        @test norm(d(fit.θ)[2]) < 1e-6

        # The offset enters the linear predictor and nothing else
        off = randn(rng, n)
        doff = logistic_derivatives(X, y; offset=off)
        ll_o, grad_o, _ = doff(β)
        pr_o = 1 ./ (1 .+ exp.(-(η .+ off)))
        @test ll_o ≈ sum(y[r] ? log(pr_o[r]) : log1p(-pr_o[r]) for r in 1:n) atol = 1e-9
        @test grad_o ≈ X' * (Float64.(y) .- pr_o) atol = 1e-9
        # a zero offset is no offset
        @test logistic_derivatives(X, y; offset=zeros(n))(β)[1] ≈ ll atol = 1e-12

        @test_throws ArgumentError logistic_derivatives(X, y[1:10])
        @test_throws ArgumentError logistic_derivatives(X, y; offset=zeros(10))

        # ALLOCATION REGRESSION. The old loop allocated a p×p outer product and
        # two broadcast temporaries PER ROW PER EVALUATION — 470 KB on a 4200-row
        # CMPLE design. An evaluation now allocates only the gradient and Hessian
        # it returns: O(p²), independent of the number of rows.
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
        @test occursin("2 masked dyads", err.msg)   # shared Networks.jl message

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
            mcmle(model; n_samples=50, burnin=100, interval=5, max_iter=2,
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
                                         burnin=100, interval=5, max_iter=2,
                                         bridge_rungs=2, bridge_samples=50)

        # Explicit opt-in: fits, warns honestly, and records the method
        fit = @test_logs (:warn, r"conditions on them at their face value") match_mode=:any mcmle(
            model; missing=:condition_on_face, n_samples=200, burnin=1000,
            interval=20, max_iter=5, bridge_rungs=2, bridge_samples=100)
        @test length(fit.coefficients) == 1
        @test fit.missing_method === :condition_on_face
        @test nobs(fit) == 120 - 2
        # The treatment is visible in the printed output
        s = sprint(show, fit)
        @test occursin("Missing dyads: 2 masked", s)
        @test occursin("conditioned on face values", s)

        # An unmasked fit records :none and prints nothing about missingness
        clean_fit = mcmle(ERGMModel(ERGMFormula([Edges()]), florentine_marriage());
                          n_samples=200, burnin=1000, interval=20, max_iter=5,
                          bridge_rungs=2, bridge_samples=100)
        @test clean_fit.missing_method === :none
        @test !occursin("Missing dyads", sprint(show, clean_fit))
    end

    @testset "Missing dyads: MPLE declares support, records available-case" begin
        # MPLE's available-case pseudo-likelihood IS a principled treatment,
        # so it declares the ecosystem trait and takes no `missing` keyword.
        @test supports_missing(mple)
        @test !supports_missing(mcmle)
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
        dep = fit_ergm(flo, [Edges(), GWESP(0.5)])

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
                   n_samples=200, burnin=200, interval=5, max_iter=3,
                   rng=Random.Xoshiro(11))
        md_mc = fit_metadata(mc)
        @test md_mc.objective == :mc_likelihood
        @test !md_mc.is_exact
        @test md_mc.se_method == :fisher
        @test any(occursin("Monte-Carlo error", a) for a in md_mc.approximations)

        # Bootstrap standard errors are reported as such
        boot = fit_ergm(flo, [Edges(), GWESP(0.5)]; se=:bootstrap, n_boot=5,
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
    # The "Golden master vs R ergm" testset above carries R's numbers as bare
    # literals in comments. They are right, but they cannot be regenerated,
    # nobody can tell which ergm produced them, and the atols beside them were
    # chosen by hand. This testset loads the same comparison from a provenanced
    # TOML fixture (test/fixtures/flomarriage_ergm.toml), generated by
    # test/fixtures/r/flomarriage_ergm.R against a stated ergm version and seed,
    # with every tolerance justified in the fixture itself.
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
        @test g.values["summary_statistic_names"] ==
              ["edges", "nodecov.wealth", "gwesp.fixed.0.5"]
        stats = [compute(Edges(), flo), compute(NodeCov(:wealth), flo),
                 compute(GWESP(0.5), flo)]
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

        # DOCUMENTED BEHAVIOURAL DIFFERENCE (not a numerical one). ERGM.jl's
        # MCMLE checks convergence BEFORE applying its first Newton update, and
        # on this model the check passes at the MPLE (max t-ratio 0.006), so the
        # returned point estimate IS the MPLE and has zero seed-to-seed
        # variance. statnet always takes at least one MCMLE step. The estimate
        # is defensible — it satisfies E_θ[g] = g_obs to within Monte-Carlo
        # error, which is the MLE condition — and it lands inside R's own noise,
        # but the two numbers are not produced the same way and a reader
        # comparing them deserves to know. Pinned so the day it changes is
        # visible rather than silent.
        mple_dd = fit_ergm(flo, [Edges(), GWESP(0.5)]; method=:mple)
        @test dd_fits[1].coefficients ≈ mple_dd.coefficients atol = 1e-12
        @test std(f.coefficients[2] for f in dd_fits) < 1e-12
        # ...while the standard errors DO come from the MCMC sample, and vary.
        @test std(f.std_errors[2] for f in dd_fits) > 1e-6
    end
end
