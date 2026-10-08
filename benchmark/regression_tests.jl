#!/usr/bin/env julia
# benchmark/regression_tests.jl — allocation-regression assertions for the
# ERGM.jl hot loops. Standalone; run with
#     julia --project=benchmark benchmark/regression_tests.jl
#
# The per-toggle change statistics (the innermost loop of both MPLE design
# construction and every MH toggle proposal) are allocation-free as of the
# P0–P2 optimization sprints (measured 0 bytes for every term below). Any
# allocation appearing here is a performance regression — these tests assert
# the loops STAY allocation-free rather than tracking a noisy byte budget.

using ERGM
using NetworkCore
using Random
using Test

function er_network(rng::AbstractRNG, n::Int, m::Int; directed::Bool=false)
    net = network(n; directed=directed)
    while ne(net) < m
        i, j = rand(rng, 1:n), rand(rng, 1:n)
        i == j && continue
        add_edge!(net, i, j)
    end
    return net
end

"Bytes allocated by `change_stat` on a pre-warmed call, worst over `dyads`."
function max_allocs_change_stat(term, net, dyads)
    worst = 0
    for (i, j) in dyads
        change_stat(term, net, i, j)                     # warm up / compile
        worst = max(worst, @allocated change_stat(term, net, i, j))
    end
    return worst
end

@testset "ERGM allocation regressions" begin
    n = 500
    rng = Random.Xoshiro(20260712)
    net_u = er_network(Random.Xoshiro(1), n, 5 * n; directed=false)
    net_d = er_network(Random.Xoshiro(2), n, 10 * n; directed=true)
    dyads = Tuple{Int, Int}[]
    while length(dyads) < 25
        i, j = rand(rng, 1:n), rand(rng, 1:n)
        i == j || push!(dyads, (i, j))
    end

    @testset "change_stat is allocation-free (undirected)" begin
        for term in [Edges(), Triangle(), GWESP(0.5), GWESP(0.0), GWDSP(0.5),
                     Kstar(2), TwoPath(), GWDegree(0.5), Degree(2),
                     GWNSP(0.5), Concurrent(), DegRange(1, 3), MeanDeg(), Density(),
                     ERGM.TriadCensus(2), ESP(0), ESP(2)]
            @test max_allocs_change_stat(term, net_u, dyads) == 0
        end
    end

    @testset "change_stat is allocation-free (directed)" begin
        for term in [Edges(), Mutual(), Triangle(), OStar(2), IStar(2),
                     GWESP(0.5), GWESP(0.5; type=:ITP), GWESP(0.5; type=:OSP),
                     GWDSP(0.5), GWDSP(0.5; type=:ISP),
                     GWIDegree(0.5), GWODegree(0.5), IDegree(2), ODegree(2),
                     GWNSP(0.5; type=:ITP), IDegRange(2), ODegRange(1, 3),
                     Sender(3), Receiver(4), ERGM.TriadCensus(9),
                     TransitiveTies(), CyclicalTies(), ESP(1), ESP(1; type=:ITP),
                     ESP(2; type=:OSP), ESP(0; type=:ISP)]
            @test max_allocs_change_stat(term, net_d, dyads) == 0
        end
    end

    # A full network MH step, tie/no-tie (the default) and random, measured
    # exactly: on a saturated network every proposal is a rejected removal,
    # so nothing toggles (no Graphs.jl adjacency regrowth — that ~100 B per
    # accepted insertion is NetworkCore/Graphs' `add_edge!`, not the sampler's)
    # and 20 000 extra steps must add nothing. The bound is the one
    # `test/runtests.jl` uses for the same quantity: 256 bytes of slack
    # absorb the few dozen bytes of platform noise (Windows) in `@allocated`
    # itself, while even 1 byte per step would exceed it 80-fold. Both TNT
    # branches run: its tie draw (Fenwick tree) and its dyad draw.
    @testset "MH steps are allocation-free per step (TNT and random)" begin
        sat = network(60; directed=false)
        model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), sat)
        θ = [50.0, 0.0]
        for prop in (:tnt, :random, :spdyad)
            g = copy(sat)
            ERGM._mh_run!(Random.Xoshiro(1), g, model.formula.terms, θ, 1, 200_000, 1,
                          false; proposal=prop)
            run(b) = ERGM._mh_run!(Random.Xoshiro(2), g, model.formula.terms, θ, 1, b, 1,
                                   false; proposal=prop)
            run(1000)
            @test abs((@allocated run(40_000)) - (@allocated run(20_000))) <= 256
        end
        s = ERGM._TNTState(net_u, :free)
        ERGM._fw_find(s.tree, s.topbit, 7)
        @test (@allocated ERGM._fw_find(s.tree, s.topbit, 7)) == 0
    end

    # The whole-model fills at 36 statistics (edges + a NodeMix on an 8-level
    # attribute): `Base.map` over a term tuple is unrolled only below 32
    # elements and boxed every Float64 from there on (~30 KB per MH step /
    # per MPLE dyad, as measured before the change); the fills are
    # generated per term count now and must stay at 0 bytes at any width.
    @testset "change_stat_all! and the MPLE key are allocation-free at 36 statistics" begin
        set_vertex_attribute!(net_u, :g8, Dict(v => "L$(mod1(v, 8))" for v in 1:n))
        wide = ERGMModel(ERGMFormula([Edges(), NodeMix(:g8)]), net_u)
        wterms = wide.formula.terms
        @test length(wterms) == 36
        delta = zeros(36)
        worst = 0
        for (i, j) in dyads
            ERGM.change_stat_all!(delta, wterms, net_u, i, j)
            worst = max(worst, @allocated ERGM.change_stat_all!(delta, wterms, net_u, i, j))
            ERGM._change_stat_tuple(wterms, net_u, i, j)
            worst = max(worst, @allocated ERGM._change_stat_tuple(wterms, net_u, i, j))
        end
        @test worst == 0
        rows = Dict{NTuple{36, Float64}, Tuple{Float64, Float64}}()
        sizehint!(rows, 64)
        ERGM._mple_rows!(rows, net_u, wterms, false)
        empty!(rows); sizehint!(rows, 64)
        @test (@allocated ERGM._mple_rows!(rows, net_u, wterms, false)) == 0
    end
end
