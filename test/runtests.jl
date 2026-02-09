using ERGM
using Network
using Test

@testset "ERGM.jl" begin
    @testset "Edges Term" begin
        net = Network(5)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        add_edge!(net, 3, 4)

        edges_term = Edges()
        @test compute(edges_term, net) == 3.0

        # Change stat for adding new edge
        @test change_stat(edges_term, net, 1, 3) == 1.0

        # Change stat for removing existing edge
        @test change_stat(edges_term, net, 1, 2) == -1.0
    end

    @testset "Mutual Term" begin
        net = Network(3)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 1)  # Mutual
        add_edge!(net, 2, 3)  # Asymmetric

        mutual_term = Mutual()
        @test compute(mutual_term, net) == 1.0

        # Adding reciprocal edge increases mutual by 1
        @test change_stat(mutual_term, net, 3, 2) == 1.0

        # Adding edge where no reverse exists doesn't change mutual
        @test change_stat(mutual_term, net, 1, 3) == 0.0
    end

    @testset "Triangle Term" begin
        net = Network(4; directed=false)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        add_edge!(net, 1, 3)  # Complete triangle

        tri_term = Triangle()
        @test compute(tri_term, net) >= 1.0

        # Adding edge that completes another triangle
        add_edge!(net, 3, 4)
        add_edge!(net, 1, 4)  # This should form another triangle
        @test compute(tri_term, net) >= 2.0
    end

    @testset "NodeMatch Term" begin
        net = Network(4)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        add_edge!(net, 3, 4)

        # Set attribute: 1 and 2 same group, 3 and 4 same group
        set_vertex_attribute!(net, :group, Dict(1 => "A", 2 => "A", 3 => "B", 4 => "B"))

        match_term = NodeMatch(:group)

        # 1→2 matches (both A)
        # 2→3 doesn't match (A→B)
        # 3→4 matches (both B)
        @test compute(match_term, net) == 2.0
    end

    @testset "MPLE Estimation" begin
        # Create a simple network
        net = Network(10; directed=false)
        for i in 1:9
            add_edge!(net, i, i+1)
        end
        add_edge!(net, 1, 10)  # Close the circle

        # Fit edges-only model
        result = fit_ergm(net, [Edges()])

        @test length(result.coefficients) == 1
        @test result.method == :mple
        @test !isnan(result.coefficients[1])
    end

    @testset "Network Simulation" begin
        net = Network(5)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)

        # Fit model
        result = fit_ergm(net, [Edges()])

        # Simulate from model
        sims = simulate_ergm(result; n_sim=2, burnin=100, interval=10)

        @test length(sims) == 2
        @test all(nv(s) == 5 for s in sims)
    end

    @testset "GOF" begin
        net = Network(5; directed=false)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 3)
        add_edge!(net, 3, 4)
        add_edge!(net, 4, 5)

        result = fit_ergm(net, [Edges()])
        gof_result = gof(result; n_sim=10, stats=[:degree])

        @test haskey(gof_result.results, :degree)
        @test length(gof_result.results[:degree].observed) > 0
    end

    @testset "Term Set" begin
        terms = [Edges(), Mutual(), Triangle()]
        ts = TermSet(terms)

        @test length(ts) == 3
        @test ts.names == ["edges", "mutual", "triangle"]

        net = Network(3)
        add_edge!(net, 1, 2)
        add_edge!(net, 2, 1)

        stats = compute_all(ts, net)
        @test length(stats) == 3
    end
end
