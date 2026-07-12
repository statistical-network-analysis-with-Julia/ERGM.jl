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
using Network
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
        for term in [Edges(), Triangle(), GWESP(0.5), GWDSP(0.5),
                     Kstar(2), TwoPath(), GWDegree(0.5)]
            @test max_allocs_change_stat(term, net_u, dyads) == 0
        end
    end

    @testset "change_stat is allocation-free (directed)" begin
        for term in [Edges(), Mutual(), Triangle(),
                     GWESP(0.5), GWESP(0.5; type=:ITP), GWESP(0.5; type=:OSP),
                     GWDSP(0.5), GWDSP(0.5; type=:ISP),
                     GWIDegree(0.5), GWODegree(0.5)]
            @test max_allocs_change_stat(term, net_d, dyads) == 0
        end
    end
end
