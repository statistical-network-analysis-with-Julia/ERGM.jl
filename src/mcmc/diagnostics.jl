"""
ERGM diagnostics.

Provides functions for assessing ERGM fit and MCMC convergence.
"""

"""
    gof(result::ERGMResult; n_sim::Int=100,
        stats::Vector{Symbol}=[:degree, :esp, :distance],
        burnin::Int=10000, interval::Int=1000,
        rng::AbstractRNG=Random.default_rng(),
        n_chains::Int=min(n_sim, 4), esp_type::Symbol=:OTP) -> GOFResult

Goodness-of-fit assessment for a fitted ERGM.

Compares observed network statistics to distributions from simulated
networks. The simulations are split over `n_chains` independent MCMC
chains run in parallel and seeded deterministically from `rng`, so results
are reproducible and independent of the thread count (see
[`sample_networks`](@ref)).

For **directed** networks the `:degree` statistic is split into separate
in- and out-degree distributions, reported under the keys `:idegree` and
`:odegree` (as R ergm's GOF does); `:idegree`/`:odegree` may also be
requested individually. The `:esp` distribution for directed networks uses
the `esp_type` shared-partner definition (default `:OTP`, statnet's directed
default — the same types as [`GWESP`](@ref), including `:union` for the
either-direction count).

# Arguments
- `result::ERGMResult`: Fitted ERGM result
- `n_sim::Int=100`: Number of networks to simulate
- `stats::Vector{Symbol}`: Statistics to evaluate (`:degree`, `:idegree`,
  `:odegree`, `:esp`, `:distance`)
- `burnin::Int=10000`, `interval::Int=1000`: MCMC controls for the
  simulations
- `rng::AbstractRNG`: Source of all random draws
- `n_chains::Int`: Number of independent simulation chains
- `esp_type::Symbol=:OTP`: Directed shared-partner type for `:esp`
  (ignored for undirected networks)

# Returns
A `Network.GOFResult` with one `GOFStatistic` panel per requested statistic
(named `"degree"`, `"idegree"`, `"odegree"`, `"esp"`, `"distance"`).
Per-level p-values are two-sided Monte-Carlo p-values computed with the
shared `(1 + k)/(N + 1)` estimator ([`Network.mc_pvalue`](@ref)), so they
are never exactly zero. `show` renders the observed value, simulation
envelope, and p-value per level.
"""
function gof(result::ERGMResult; n_sim::Int=100,
             stats::Vector{Symbol}=[:degree, :esp, :distance],
             burnin::Int=10000, interval::Int=1000,
             rng::AbstractRNG=Random.default_rng(),
             n_chains::Int=min(n_sim, 4),
             esp_type::Symbol=:OTP)
    model = result.model
    obs_net = model.network
    directed = is_directed(obs_net)

    # Simulate networks (parallel chains, deterministic per-chain seeds)
    sim_nets = simulate_ergm(result; n_sim=n_sim, burnin=burnin,
                             interval=interval, rng=rng, n_chains=n_chains)

    panels = GOFStatistic[]

    for stat in stats
        if stat == :degree
            if directed
                # Directed degree GOF is split by direction: a single
                # "degree" panel would silently report out-degrees only
                push!(panels, _gof_degree(obs_net, sim_nets, :in))
                push!(panels, _gof_degree(obs_net, sim_nets, :out))
            else
                push!(panels, _gof_degree(obs_net, sim_nets))
            end
        elseif stat == :idegree
            push!(panels, _gof_degree(obs_net, sim_nets, :in))
        elseif stat == :odegree
            push!(panels, _gof_degree(obs_net, sim_nets, :out))
        elseif stat == :esp
            push!(panels, _gof_esp(obs_net, sim_nets; type=esp_type))
        elseif stat == :distance
            push!(panels, _gof_distance(obs_net, sim_nets))
        end
    end

    return GOFResult(panels; model="ERGM")
end

"""
    _gof_degree(obs_net, sim_nets, mode::Symbol=:total) -> GOFStatistic

GOF panel for a degree distribution. `mode` selects total (`:total`),
in- (`:in`), or out- (`:out`) degrees; the panel is named accordingly
(`"degree"`, `"idegree"`, `"odegree"`).
"""
function _gof_degree(obs_net, sim_nets, mode::Symbol=:total)
    degf = mode === :in ? ((net, v) -> length(inneighbors(net, v))) :
           mode === :out ? ((net, v) -> length(outneighbors(net, v))) :
           ((net, v) -> length(neighbors(net, v)))
    panel_name = mode === :in ? "idegree" : mode === :out ? "odegree" : "degree"

    n = nv(obs_net)
    max_degree = n - 1

    # Observed degree distribution
    obs_degrees = [degf(obs_net, v) for v in vertices(obs_net)]
    obs_dist = [count(==(d), obs_degrees) for d in 0:max_degree]

    # Simulated distributions
    n_sim = length(sim_nets)
    sim_dists = zeros(n_sim, max_degree + 1)

    for (i, sim_net) in enumerate(sim_nets)
        sim_degrees = [degf(sim_net, v) for v in vertices(sim_net)]
        for d in 0:max_degree
            sim_dists[i, d+1] = count(==(d), sim_degrees)
        end
    end

    # Per-bin p-values are computed by the GOFStatistic constructor with the
    # shared two-sided (1 + k)/(N + 1) Monte-Carlo estimator
    return GOFStatistic(panel_name, string.(0:max_degree), obs_dist, sim_dists)
end

"""
    _gof_esp(obs_net, sim_nets; type::Symbol=:OTP) -> GOFStatistic

GOF for the edgewise shared partner distribution.

For directed networks the shared partners of each edge i→j are counted
under the `type` definition (`:OTP`/`:ITP`/`:OSP`/`:ISP`, as in
[`GWESP`](@ref), plus `:union` for either-direction adjacency); `type` is
ignored for undirected networks. Counts use the O(degree) sorted
neighbor-list intersections from the terms layer.
"""
function _gof_esp(obs_net, sim_nets; type::Symbol=:OTP)
    type in (:OTP, :ITP, :OSP, :ISP, :union) ||
        throw(ArgumentError("type must be :OTP, :ITP, :OSP, :ISP, or :union"))
    typed = is_directed(obs_net) && type !== :union

    function compute_esp_dist(net)
        esp_counts = Dict{Int, Int}()
        for e in edges(net)
            i, j = Int(src(e)), Int(dst(e))
            i == j && continue
            esp = typed ? _sp_typed_masked(net, i, j, type, 0, 0) :
                          _shared_partners_masked(net, i, j, 0, 0)
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

    return GOFStatistic("esp", string.(0:max_esp), obs_dist, sim_dists)
end

"""
    _gof_distance(obs_net, sim_nets) -> GOFStatistic

GOF for geodesic distance distribution.

For directed networks, distances are out-distances (following edge
direction), matching a breadth-first search from each source vertex.
"""
function _gof_distance(obs_net, sim_nets)
    n = nv(obs_net)

    function compute_dist_dist(net)
        dist_counts = Dict{Int, Int}()
        for i in 1:n
            distances = Graphs.gdistances(net, i)
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

    return GOFStatistic("distance", string.(1:max_dist), obs_vec, sim_dists)
end

# Autocovariance of x at lag k around the precomputed mean x̄ (1/n
# normalization, the standard biased spectral estimate)
function _autocov(x::AbstractVector{<:Real}, x̄::Float64, k::Int)
    n = length(x)
    s = 0.0
    @inbounds for t in 1:(n - k)
        s += (x[t] - x̄) * (x[t + k] - x̄)
    end
    return s / n
end

"""
    _geyer_var(x) -> (σ², γ₀)

Geyer (1992) initial monotone sequence estimate of the asymptotic variance
`σ² = limₙ n·Var(x̄)` of a stationary MCMC trace, together with the lag-0
autocovariance `γ₀`: sums of adjacent autocovariance pairs
`Γₘ = γ₂ₘ + γ₂ₘ₊₁` are accumulated while positive, enforcing monotone
non-increase — the standard initial-sequence truncation that is consistent
for reversible chains.
"""
function _geyer_var(x::AbstractVector{<:Real})
    n = length(x)
    x̄ = mean(x)
    γ0 = _autocov(x, x̄, 0)
    γ0 > 0 || return (0.0, γ0)

    σ² = -γ0
    prev = Inf
    m = 0
    while 2m + 1 <= n - 1
        Γ = _autocov(x, x̄, 2m) + _autocov(x, x̄, 2m + 1)
        Γ > 0 || break
        Γ = min(Γ, prev)          # initial monotone sequence
        σ² += 2Γ
        prev = Γ
        m += 1
    end
    return (σ², γ0)
end

"""
    _geyer_ess(x) -> Float64

Effective sample size `n·γ₀/σ²` from the Geyer initial-sequence variance
estimate — unlike the lag-1 estimate, this accounts for autocorrelation at
all lags. Degenerate traces (zero variance, or a non-positive variance
estimate) report `n`.
"""
function _geyer_ess(x::AbstractVector{<:Real})
    σ², γ0 = _geyer_var(x)
    (γ0 > 0 && σ² > 0) || return Float64(length(x))
    return length(x) * γ0 / σ²
end

"""
    _geweke_z(x; first_frac=0.1, last_frac=0.5) -> Float64

Geweke (1992) convergence z-score: the difference between the means of the
first `first_frac` and last `last_frac` of the trace, standardized by
spectral (Geyer initial-sequence) estimates of each segment's variance.
Under stationarity z ~ N(0,1); large |z| indicates the chain had not
converged when sampling started.
"""
function _geweke_z(x::AbstractVector{<:Real}; first_frac::Float64=0.1,
                   last_frac::Float64=0.5)
    n = length(x)
    na = max(2, floor(Int, first_frac * n))
    nb = max(2, floor(Int, last_frac * n))
    a = view(x, 1:na)
    b = view(x, (n - nb + 1):n)

    σa², _ = _geyer_var(a)
    σb², _ = _geyer_var(b)
    se = sqrt(σa² / na + σb² / nb)
    se > 0 || return mean(a) == mean(b) ? 0.0 : Inf
    return (mean(a) - mean(b)) / se
end

"""
    mcmc_diagnostics(result::ERGMResult) -> NamedTuple

MCMC diagnostics for MCMLE results.

Throws an `ArgumentError` if the fit has no MCMC samples (i.e. it was
estimated by MPLE): there is no chain to diagnose. Refit with
`method=:mcmle` to obtain a result this function accepts.

# Returns
NamedTuple with one entry per model statistic:
- `term_names`: Statistic names
- `autocorrelation`: Lag-1 autocorrelation
- `effective_sample_size`: ESS from the lag-1 autocorrelation (the simple
  `n(1−ρ₁)/(1+ρ₁)` estimate, kept for backward compatibility — it is the
  most optimistic of the two ESS columns)
- `ess_geyer`: Geyer initial-sequence ESS, accounting for autocorrelation
  at all lags (preferred; see [`_geyer_ess`](@ref))
- `geweke_z`, `geweke_p`: Geweke convergence z-scores comparing the first
  10% of the chain against the last 50%, and their two-sided normal
  p-values — small p-values flag a chain that had not reached
  stationarity
- `n_samples`: Number of MCMC samples
"""
function mcmc_diagnostics(result::ERGMResult)
    if isnothing(result.mcmc_samples)
        throw(ArgumentError(
            "this fit has no MCMC samples to diagnose: it was estimated with " *
            "$(result.method) (maximum pseudo-likelihood), which draws no MCMC " *
            "sample. Refit with method=:mcmle — e.g. " *
            "fit_ergm(net, terms; method=:mcmle) — and call mcmc_diagnostics " *
            "on that result."))
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

    # Effective sample size (simple lag-1 estimate)
    ess = zeros(p)
    for j in 1:p
        if abs(autocorr[j]) < 1
            ess[j] = n_samples * (1 - autocorr[j]) / (1 + autocorr[j])
        else
            ess[j] = n_samples
        end
    end

    # Geyer initial-sequence ESS and Geweke convergence diagnostics
    ess_geyer = [_geyer_ess(view(samples, :, j)) for j in 1:p]
    geweke_z = [_geweke_z(view(samples, :, j)) for j in 1:p]
    geweke_p = _z_pvalues(geweke_z)

    return (
        term_names = term_names,
        autocorrelation = autocorr,
        effective_sample_size = ess,
        ess_geyer = ess_geyer,
        geweke_z = geweke_z,
        geweke_p = geweke_p,
        n_samples = n_samples
    )
end
