using ERGM
using Network
using Random
using Statistics
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
                     GWESP(0.5), GWESP(1.2), GWDegree(0.5), GWDegree(1.2)]
            @test check_change_stats(term, net) === nothing
        end
    end

    @testset "Structural terms match brute force (directed)" begin
        net = fixture_directed()
        for term in [Edges(), Mutual(), Triangle(), Kstar(2), TwoPath(),
                     GWESP(0.5), GWDegree(0.5)]
            @test check_change_stats(term, net) === nothing
        end
    end

    @testset "Nodal and dyadic terms match brute force" begin
        for make_net in (fixture_undirected, fixture_directed)
            net = set_test_attrs!(make_net())
            for term in [NodeFactor(:group; level="A"), NodeCov(:age),
                         NodeMatch(:group), NodeMatch(:group; diff=true),
                         AbsDiff(:age)]
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
        @test compute(NodeMatch(:group; diff=true), net) == 1.0
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

    @testset "MCMLE recovers edges-only coefficient" begin
        Random.seed!(42)
        flo = florentine_marriage()

        result = fit_ergm(flo, [Edges()]; method=:mcmle,
                          n_samples=400, burnin=1000, interval=20, max_iter=10, tol=0.5)
        # For a dyad-independent model MCMLE should agree with the MLE
        @test result.method == :mcmle
        @test result.coefficients[1] ≈ -1.609438 atol = 0.25
        @test !isnothing(result.mcmc_samples)
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

        for stat in (:degree, :esp, :distance)
            @test haskey(gof_result.results, stat)
            stat_gof = gof_result.results[stat]
            @test length(stat_gof.observed) > 0
            # Two-sided Monte Carlo p-values live in [0, 1]
            @test all(0.0 .<= stat_gof.p_values .<= 1.0)
        end
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
end
