#!/usr/bin/env julia
# benchmark/benchmarks.jl — BenchmarkTools suite for ERGM.jl's hot loops.
#
# Locks in the O(degree) per-toggle change statistics (Triangle, GWESP,
# GWDSP): each term is benchmarked on sparse Erdős–Rényi networks with the
# SAME expected mean degree at n = 500 and n = 2000, and the n-scaling of
# the measured per-toggle cost is asserted (an O(n) regression would show
# up as a ≈4× ratio; O(degree) stays ≈1×).
#
# Defines the standard `SUITE::BenchmarkGroup`. Run standalone with
#     julia --project=benchmark benchmark/benchmarks.jl
# which tunes + runs the suite, prints one tab-separated `BENCHJL` line per
# benchmark (consumed by the site repo's tools/run_benchmarks.jl), and exits
# non-zero if the scaling assertion fails.

using BenchmarkTools
using ERGM
using Network
using Random

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

const MEAN_DEGREE = 10          # identical at both sizes: cost must not grow
const N_SMALL = 500
const N_LARGE = 2000
const N_DYADS = 200             # dyads swept per benchmark evaluation
const SCALING_LIMIT = 3.0       # tolerated t(n=2000)/t(n=500) ratio

"Sparse Erdős–Rényi network with expected mean degree `MEAN_DEGREE`."
function er_network(rng::AbstractRNG, n::Int; directed::Bool=false)
    net = network(n; directed=directed)
    m = directed ? MEAN_DEGREE * n : (MEAN_DEGREE * n) ÷ 2
    while ne(net) < m
        i, j = rand(rng, 1:n), rand(rng, 1:n)
        i == j && continue
        add_edge!(net, i, j)
    end
    return net
end

"Fixed sample of `N_DYADS` random dyads (i ≠ j) to sweep per evaluation."
function sample_dyads(rng::AbstractRNG, n::Int)
    dyads = Tuple{Int, Int}[]
    while length(dyads) < N_DYADS
        i, j = rand(rng, 1:n), rand(rng, 1:n)
        i == j || push!(dyads, (i, j))
    end
    return dyads
end

"Sum of add-direction change statistics over a fixed dyad sample."
function sweep_change_stat(term, net, dyads)
    s = 0.0
    for (i, j) in dyads
        s += change_stat(term, net, i, j)
    end
    return s
end

const NETS = Dict(n => er_network(Random.Xoshiro(n), n) for n in (N_SMALL, N_LARGE))
const DYADS = Dict(n => sample_dyads(Random.Xoshiro(n + 1), n) for n in (N_SMALL, N_LARGE))
const NET_DIRECTED = er_network(Random.Xoshiro(3), N_SMALL; directed=true)
const DYADS_DIRECTED = sample_dyads(Random.Xoshiro(4), N_SMALL)

const TERMS = [("triangle", Triangle()),
               ("gwesp", GWESP(0.5)),
               ("gwdsp", GWDSP(0.5))]

# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------

const SUITE = BenchmarkGroup()

let g = addgroup!(SUITE, "change_stat")
    for (label, term) in TERMS
        for n in (N_SMALL, N_LARGE)
            g["$(label)_n$(n)"] =
                @benchmarkable sweep_change_stat($term, $(NETS[n]), $(DYADS[n]))
        end
        # Directed variants exercise the union/typed merge paths
        g["$(label)_directed_n$(N_SMALL)"] =
            @benchmarkable sweep_change_stat($term, $NET_DIRECTED, $DYADS_DIRECTED)
    end
end

# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

function print_benchjl(results::BenchmarkGroup)
    for (path, trial) in BenchmarkTools.leaves(results)
        est = median(trial)
        println("BENCHJL\t", join(path, "/"), "\t",
                BenchmarkTools.time(est), "\t",
                BenchmarkTools.allocs(est), "\t",
                BenchmarkTools.memory(est))
    end
end

"Assert that per-toggle cost did not grow with n (O(degree), not O(n))."
function assert_scaling(results::BenchmarkGroup)
    ok = true
    for (label, _) in TERMS
        t_small = BenchmarkTools.time(median(results["change_stat"]["$(label)_n$(N_SMALL)"]))
        t_large = BenchmarkTools.time(median(results["change_stat"]["$(label)_n$(N_LARGE)"]))
        ratio = t_large / t_small
        println("SCALING\t", label, "\tn", N_LARGE, "/n", N_SMALL, "\t",
                round(ratio, digits=2))
        if ratio > SCALING_LIMIT
            println(stderr, "SCALING FAILURE: $label change statistic is ",
                    round(ratio, digits=2), "x slower at n=$(N_LARGE) than at ",
                    "n=$(N_SMALL) (same mean degree; limit $(SCALING_LIMIT)x). ",
                    "The per-toggle cost is no longer O(degree).")
            ok = false
        end
    end
    return ok
end

function main()
    tune!(SUITE)
    results = run(SUITE; verbose=false, seconds=1)
    print_benchjl(results)
    assert_scaling(results) || exit(1)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
