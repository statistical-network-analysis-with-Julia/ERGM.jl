"""
ERGM diagnostics.

Provides functions for assessing ERGM fit and MCMC convergence.
"""

using Statistics

"""
    gof(result::ERGMResult; n_sim::Int=100, stats::Vector{Symbol}=[:degree, :esp, :distance]) -> NamedTuple

Goodness-of-fit assessment for a fitted ERGM.

Compares observed network statistics to distributions from simulated networks.

# Arguments
- `result::ERGMResult`: Fitted ERGM result
- `n_sim::Int=100`: Number of networks to simulate
- `stats::Vector{Symbol}`: Statistics to evaluate

# Returns
NamedTuple with GOF results for each statistic.
"""
function gof(result::ERGMResult; n_sim::Int=100,
             stats::Vector{Symbol}=[:degree, :esp, :distance])
    model = result.model
    obs_net = model.network

    # Simulate networks
    sim_nets = simulate_ergm(result; n_sim=n_sim)

    gof_results = Dict{Symbol, Any}()

    for stat in stats
        if stat == :degree
            gof_results[:degree] = _gof_degree(obs_net, sim_nets)
        elseif stat == :esp
            gof_results[:esp] = _gof_esp(obs_net, sim_nets)
        elseif stat == :distance
            gof_results[:distance] = _gof_distance(obs_net, sim_nets)
        end
    end

    return (results=gof_results, n_sim=n_sim)
end

"""
    _gof_degree(obs_net, sim_nets) -> NamedTuple

GOF for degree distribution.
"""
function _gof_degree(obs_net, sim_nets)
    n = nv(obs_net)
    max_degree = n - 1

    # Observed degree distribution
    obs_degrees = [length(neighbors(obs_net, v)) for v in vertices(obs_net)]
    obs_dist = [count(==(d), obs_degrees) for d in 0:max_degree]

    # Simulated distributions
    n_sim = length(sim_nets)
    sim_dists = zeros(n_sim, max_degree + 1)

    for (i, sim_net) in enumerate(sim_nets)
        sim_degrees = [length(neighbors(sim_net, v)) for v in vertices(sim_net)]
        for d in 0:max_degree
            sim_dists[i, d+1] = count(==(d), sim_degrees)
        end
    end

    # Compute p-values (proportion of simulations with stat >= observed)
    p_values = zeros(max_degree + 1)
    for d in 0:max_degree
        p_values[d+1] = mean(sim_dists[:, d+1] .>= obs_dist[d+1])
    end

    return (
        observed = obs_dist,
        simulated_mean = vec(mean(sim_dists, dims=1)),
        simulated_sd = vec(std(sim_dists, dims=1)),
        p_values = p_values
    )
end

"""
    _gof_esp(obs_net, sim_nets) -> NamedTuple

GOF for edgewise shared partner distribution.
"""
function _gof_esp(obs_net, sim_nets)
    n = nv(obs_net)

    function compute_esp_dist(net)
        esp_counts = Dict{Int, Int}()
        for e in edges(net)
            i, j = src(e), dst(e)
            # Count shared partners
            esp = 0
            for k in vertices(net)
                k == i && continue
                k == j && continue
                if (has_edge(net, i, k) || has_edge(net, k, i)) &&
                   (has_edge(net, j, k) || has_edge(net, k, j))
                    esp += 1
                end
            end
            esp_counts[esp] = get(esp_counts, esp, 0) + 1
        end
        return esp_counts
    end

    obs_esp = compute_esp_dist(obs_net)
    max_esp = isempty(obs_esp) ? 0 : maximum(keys(obs_esp))

    # Also check simulations for max ESP
    for sim_net in sim_nets
        sim_esp = compute_esp_dist(sim_net)
        if !isempty(sim_esp)
            max_esp = max(max_esp, maximum(keys(sim_esp)))
        end
    end

    obs_dist = [get(obs_esp, e, 0) for e in 0:max_esp]

    n_sim = length(sim_nets)
    sim_dists = zeros(n_sim, max_esp + 1)

    for (i, sim_net) in enumerate(sim_nets)
        sim_esp = compute_esp_dist(sim_net)
        for e in 0:max_esp
            sim_dists[i, e+1] = get(sim_esp, e, 0)
        end
    end

    p_values = [mean(sim_dists[:, e+1] .>= obs_dist[e+1]) for e in 0:max_esp]

    return (
        observed = obs_dist,
        simulated_mean = vec(mean(sim_dists, dims=1)),
        simulated_sd = vec(std(sim_dists, dims=1)),
        p_values = p_values
    )
end

"""
    _gof_distance(obs_net, sim_nets) -> NamedTuple

GOF for geodesic distance distribution.
"""
function _gof_distance(obs_net, sim_nets)
    n = nv(obs_net)

    function compute_dist_dist(net)
        dist_counts = Dict{Int, Int}()
        for i in 1:n
            distances = Graphs.gdistances(net.graph, i)
            for j in 1:n
                j == i && continue
                d = distances[j]
                if d < typemax(Int)
                    dist_counts[d] = get(dist_counts, d, 0) + 1
                end
            end
        end
        return dist_counts
    end

    obs_dist = compute_dist_dist(obs_net)
    max_dist = isempty(obs_dist) ? 1 : maximum(keys(obs_dist))

    for sim_net in sim_nets
        sim_dist = compute_dist_dist(sim_net)
        if !isempty(sim_dist)
            max_dist = max(max_dist, maximum(keys(sim_dist)))
        end
    end

    obs_vec = [get(obs_dist, d, 0) for d in 1:max_dist]

    n_sim = length(sim_nets)
    sim_dists = zeros(n_sim, max_dist)

    for (i, sim_net) in enumerate(sim_nets)
        sim_dist = compute_dist_dist(sim_net)
        for d in 1:max_dist
            sim_dists[i, d] = get(sim_dist, d, 0)
        end
    end

    p_values = [mean(sim_dists[:, d] .>= obs_vec[d]) for d in 1:max_dist]

    return (
        observed = obs_vec,
        simulated_mean = vec(mean(sim_dists, dims=1)),
        simulated_sd = vec(std(sim_dists, dims=1)),
        p_values = p_values
    )
end

"""
    mcmc_diagnostics(result::ERGMResult) -> NamedTuple

MCMC diagnostics for MCMLE results.

# Returns
NamedTuple with:
- `autocorrelation`: Autocorrelation at various lags
- `effective_sample_size`: Effective sample size estimates
"""
function mcmc_diagnostics(result::ERGMResult)
    if isnothing(result.mcmc_samples)
        return (error="No MCMC samples available (model not fit with MCMLE)",)
    end

    samples = result.mcmc_samples
    n_samples, p = size(samples)
    term_names = result.model.formula.terms.names

    # Autocorrelation at lag 1
    autocorr = zeros(p)
    for j in 1:p
        x = samples[:, j]
        x_mean = mean(x)
        var_x = var(x)
        if var_x > 0
            autocorr[j] = mean((x[1:end-1] .- x_mean) .* (x[2:end] .- x_mean)) / var_x
        end
    end

    # Effective sample size (simple estimate)
    ess = zeros(p)
    for j in 1:p
        if abs(autocorr[j]) < 1
            ess[j] = n_samples * (1 - autocorr[j]) / (1 + autocorr[j])
        else
            ess[j] = n_samples
        end
    end

    return (
        term_names = term_names,
        autocorrelation = autocorr,
        effective_sample_size = ess,
        n_samples = n_samples
    )
end
