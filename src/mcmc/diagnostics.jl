"""
ERGM diagnostics.

Provides functions for assessing ERGM fit and MCMC convergence.
"""

"""
    gof(result::ERGMResult; n_sim::Int=100,
        stats::Vector{Symbol}=[:degree, :esp, :distance],
        burnin=nothing, interval=nothing,
        rng::AbstractRNG=Random.default_rng(),
        n_chains::Int=min(n_sim, 4), esp_type::Symbol=:OTP,
        missing::Symbol=:error) -> GOFResult

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
  `:odegree`, `:esp`, `:distance`); any other symbol is an `ArgumentError`
  raised before a single network is simulated
- `burnin::Int`, `interval::Int`: MCMC controls for the simulations.
  Default to the dyad-scaled rule every sampler shares
  ([`_mcmc_defaults`](@ref)): `20 × n_dyads` and `max(100, n_dyads ÷ 10)`
- `rng::AbstractRNG`: Source of all random draws
- `n_chains::Int`: Number of independent simulation chains
- `esp_type::Symbol=:OTP`: Directed shared-partner type for `:esp`
  (ignored for undirected networks)
- `missing::Symbol=:error`: Missing-dyad policy. GOF reads the *observed*
  network's degree/ESP/geodesic distributions at face value **and** simulates
  from the model, so a masked dyad would be silently reinterpreted as an
  observed tie on both sides of the comparison. `:error` (the default)
  refuses; `:condition_on_face` is the explicit, warned opt-in.

# Returns
A `Networks.GOFResult` with one `GOFStatistic` panel per requested statistic
(named `"degree"`, `"idegree"`, `"odegree"`, `"esp"`, `"distance"`).
Per-level p-values are two-sided Monte-Carlo p-values computed with the
shared `(1 + k)/(N + 1)` estimator (`Networks.mc_pvalue`), so they
are never exactly zero. `show` renders the observed value, simulation
envelope, and p-value per level.

# Example
```julia
using ERGM, Random
net = load_dataset(:florentine_marriage)
fit = fit_ergm(net, [Edges(), NodeCov(:wealth)])
g = gof(fit; n_sim=20, stats=[:degree, :esp], rng=Xoshiro(1))
n_simulations(g)                         # 20
[s.name for s in g.statistics]           # ["degree", "esp"]
```
"""
function gof(result::ERGMResult; n_sim::Int=100,
             stats::Vector{Symbol}=[:degree, :esp, :distance],
             burnin::Union{Nothing,Int}=nothing,
             interval::Union{Nothing,Int}=nothing,
             rng::AbstractRNG=Random.default_rng(),
             n_chains::Int=min(n_sim, 4),
             esp_type::Symbol=:OTP,
             missing::Symbol=:error)
    # Validate the request before spending the simulations: an unknown
    # symbol used to be dropped silently (or, alone, to surface as "GOFResult
    # needs at least one GOFStatistic" after every network had been drawn)
    unknown = setdiff(stats, _GOF_STATS)
    isempty(unknown) || throw(ArgumentError(
        "gof: unknown statistic(s) $(join(repr.(unknown), ", ")); stats must be " *
        "drawn from " * join(repr.(_GOF_STATS), ", ") * "."))
    isempty(stats) && throw(ArgumentError(
        "gof: stats is empty; request at least one of " *
        join(repr.(_GOF_STATS), ", ") * "."))
    model = result.model
    obs_net = model.network
    directed = is_directed(obs_net)
    burnin, interval = _resolve_mcmc_controls(model, burnin, interval)

    # Missing-data guard: the observed panels below read the face value of
    # every masked dyad, and the simulations freeze it there. Neither may
    # happen without the caller having asked. (`simulate_ergm` emits the
    # one warning on the opt-in path.)
    _guard_missing(obs_net, missing; context="gof")

    # Simulate networks (parallel chains, deterministic per-chain seeds)
    sim_nets = simulate_ergm(result; n_sim=n_sim, burnin=burnin,
                             interval=interval, rng=rng, n_chains=n_chains,
                             missing=missing)

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

const _GOF_STATS = (:degree, :idegree, :odegree, :esp, :distance)

# `gof` is the shared Networks.jl generic, so a one-argument
# `missing_policies(gof)` cannot speak for ERGM's method alone; the
# per-result-type form declares what the `ERGMResult` method accepts (the same
# vocabulary as `simulate_ergm`, which it delegates to).
missing_policies(::typeof(gof), ::Type{<:ERGMResult}) = _MISSING_POLICIES

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

GOF panel for the geodesic-distance distribution, R ergm's `GOF = ~distance`:
the number of dyads at each shortest-path length `1, 2, …`, plus a final
`"Inf"` level counting the dyads with no path between them — the
reachability row, the one thing this panel adds over the degree and ESP
panels. Undirected networks count UNORDERED pairs (20 + 35 + 32 + 15 + 3 +
15 unreachable = 120 on flomarriage, R's `obs.dist`); directed networks count
ORDERED pairs with out-distances (a breadth-first search following edge
direction from each source vertex), as R does. Both panels are pinned to R
at 1e-9 by the provenanced fixtures. Finite levels run to the largest
distance seen in the observed or any simulated network.
"""
function _gof_distance(obs_net, sim_nets)
    n = nv(obs_net)
    directed = is_directed(obs_net)

    # Finite-distance counts and the number of unreachable pairs
    function compute_dist_dist(net)
        dist_counts = Dict{Int, Int}()
        unreachable = 0
        for i in 1:n
            distances = Graphs.gdistances(net, i)
            for j in (directed ? (1:n) : ((i + 1):n))
                j == i && continue
                d = distances[j]
                if d < typemax(Int)
                    dist_counts[d] = get(dist_counts, d, 0) + 1
                else
                    unreachable += 1
                end
            end
        end
        return dist_counts, unreachable
    end

    obs_dist, obs_inf = compute_dist_dist(obs_net)
    max_dist = isempty(obs_dist) ? 1 : maximum(keys(obs_dist))

    sims = [compute_dist_dist(sim_net) for sim_net in sim_nets]
    for (sim_dist, _) in sims
        if !isempty(sim_dist)
            max_dist = max(max_dist, maximum(keys(sim_dist)))
        end
    end

    obs_vec = [[get(obs_dist, d, 0) for d in 1:max_dist]; obs_inf]

    n_sim = length(sim_nets)
    sim_dists = zeros(n_sim, max_dist + 1)
    for (i, (sim_dist, sim_inf)) in enumerate(sims)
        for d in 1:max_dist
            sim_dists[i, d] = get(sim_dist, d, 0)
        end
        sim_dists[i, max_dist + 1] = sim_inf
    end

    return GOFStatistic("distance", [string.(1:max_dist); "Inf"], obs_vec, sim_dists)
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
    _mc_cov_of_mean(samples, chain_lengths=[size(samples, 1)]) -> Matrix{Float64}

Monte-Carlo covariance `Σ_mc = Var(ḡ)` of the mean of the sampled statistics
— the input to the MCMC-error component of the MCMLE standard errors
(`V·Σ_mc·V`, Hunter & Handcock 2006 §3.3). Per chain `c` of length `L_c`
the per-draw asymptotic variances `σ²_j` come from the Geyer
initial-sequence estimate ([`_geyer_var`](@ref)), the cross terms from the
lag-0 correlations `ρ_jk` of the chain (`Σ_c[j,k] = ρ_jk·σ_j·σ_k` — the
diagonal is exact Geyer, the off-diagonal assumes the autocorrelation
structure of statistics `j` and `k` is shared, the same simplification
`_effective_sample_size` makes by using per-statistic ESSs); independent
chains combine as `Var(ḡ) = Σ_c L_c·Σ_c / n²`. A degenerate statistic
(zero variance) contributes zero.
"""
function _mc_cov_of_mean(samples::AbstractMatrix{<:Real},
                         chain_lengths::AbstractVector{<:Integer}=[size(samples, 1)])
    n, p = size(samples)
    Σ = zeros(p, p)
    σ = Vector{Float64}(undef, p)
    offset = 0
    for L in chain_lengths
        block = view(samples, (offset + 1):(offset + L), :)
        offset += L
        L >= 2 || continue
        for j in 1:p
            σ²j, _ = _geyer_var(view(block, :, j))
            σ[j] = sqrt(max(σ²j, 0.0))
        end
        C = cov(block)
        for k in 1:p, j in 1:p
            ρ = if j == k
                1.0
            elseif C[j, j] > 0 && C[k, k] > 0
                C[j, k] / sqrt(C[j, j] * C[k, k])
            else
                0.0
            end
            Σ[j, k] += L * ρ * σ[j] * σ[k]
        end
    end
    Σ ./= n^2
    return Σ
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
    MCMCDiagnostics

What [`mcmc_diagnostics`](@ref) returns: one row of chain diagnostics per
model statistic, computed **chain by chain** when the fit ran `n_chains > 1`
(the sample is the chains' concatenation; `chain_lengths` says where the
seams are) and printed as a table by `show`. Fields:

- `term_names::Vector{String}`: statistic names
- `autocorrelation::Vector{Float64}`: lag-1 autocorrelation (within chains,
  length-weighted over chains)
- `effective_sample_size::Vector{Float64}`: ESS from the lag-1
  autocorrelation, `Σ_c L_c (1−ρ_c)/(1+ρ_c)` — the most optimistic column
- `ess_geyer::Vector{Float64}`: Geyer initial-sequence ESS summed over
  chains — all lags, the preferred column, and the same estimator the
  MCMLE's own convergence test uses (`fit.mcmc_convergence.n_eff`)
- `geweke_z::Vector{Float64}`, `geweke_p::Vector{Float64}`: Geweke
  stationarity z-scores (first 10 % vs last 50 % of a chain) — with several
  chains the z of largest magnitude, and its two-sided p-value (the
  smallest); a small p flags a chain that had not reached stationarity
- `n_samples::Int`, `chain_lengths::Vector{Int}`: total draws and the
  per-chain counts

# Example
```julia
using ERGM, Random
net = load_dataset(:florentine_marriage)
fit = mcmle(ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), net);
            n_samples=400, n_chains=2, bridge_rungs=0, rng=Xoshiro(1))
d = mcmc_diagnostics(fit)
d isa MCMCDiagnostics              # true
d.chain_lengths                    # [200, 200]
println(d)                         # the per-term table
```
"""
struct MCMCDiagnostics
    term_names::Vector{String}
    autocorrelation::Vector{Float64}
    effective_sample_size::Vector{Float64}
    ess_geyer::Vector{Float64}
    geweke_z::Vector{Float64}
    geweke_p::Vector{Float64}
    n_samples::Int
    chain_lengths::Vector{Int}
end

# Fixed-width number for the diagnostics table
_fmt_col(x::Real, width::Int, digits::Int) =
    lpad(isfinite(x) ? string(round(x; digits=digits)) : string(x), width)

function Base.show(io::IO, d::MCMCDiagnostics)
    nc = length(d.chain_lengths)
    println(io, "MCMC diagnostics: $(d.n_samples) draws in $nc chain$(nc == 1 ? "" : "s")" *
                (nc == 1 ? "" : " of $(join(d.chain_lengths, ", "))"))
    w = max(4, maximum(length.(d.term_names)))
    println(io, rpad("", w), "  ", lpad("lag-1 AC", 9), lpad("ESS (lag-1)", 13),
            lpad("ESS (Geyer)", 13), lpad("Geweke z", 10), lpad("Geweke p", 10))
    for k in eachindex(d.term_names)
        println(io, rpad(d.term_names[k], w), "  ",
                _fmt_col(d.autocorrelation[k], 9, 3),
                _fmt_col(d.effective_sample_size[k], 13, 1),
                _fmt_col(d.ess_geyer[k], 13, 1),
                _fmt_col(d.geweke_z[k], 10, 2),
                _fmt_col(d.geweke_p[k], 10, 3))
    end
    print(io, "Prefer ESS (Geyer): it accounts for autocorrelation at every lag. ",
              "A small Geweke p flags a chain that had not reached stationarity.")
end

"""
    mcmc_diagnostics(result::ERGMResult) -> MCMCDiagnostics

MCMC diagnostics for MCMLE results: per statistic, the lag-1
autocorrelation, two effective sample sizes and the Geweke stationarity
test, as an [`MCMCDiagnostics`](@ref) whose `show` prints the table R's
`mcmc.diagnostics` prints.

Throws an `ArgumentError` if the fit has no MCMC samples (i.e. it was
estimated by MPLE): there is no chain to diagnose. Refit with
`method=:mcmle` to obtain a result this function accepts.

With `n_chains > 1` every quantity is computed **within each chain** and
combined — the lag-1 autocorrelation length-weighted, both effective sample
sizes summed (the Geyer column is then exactly the MCMLE's own
`fit.mcmc_convergence.n_eff` estimator), the Geweke test reported for the
chain of largest |z| — never across the seams of the concatenated sample,
where the first 10 % of chain 1 would otherwise be compared with the tail
of the last chain.

# Returns
An `MCMCDiagnostics` with fields `term_names`, `autocorrelation`,
`effective_sample_size` (lag-1 estimate, `Σ_c L_c(1−ρ_c)/(1+ρ_c)`, the most
optimistic column), `ess_geyer` (Geyer initial-sequence ESS, all lags,
summed over chains — preferred), `geweke_z`/`geweke_p` (first 10 % vs last
50 % of a chain; small p flags a chain that had not reached stationarity),
`n_samples` and `chain_lengths`.

# Example
```julia
using ERGM, Random
net = load_dataset(:florentine_marriage)
fit = mcmle(ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), net);
            n_samples=300, bridge_rungs=0, rng=Xoshiro(1))
d = mcmc_diagnostics(fit)
d.term_names                       # ["edges", "gwesp.fixed.0.5"]
all(d.ess_geyer .> 20)             # true on a mixing chain
all(d.geweke_p .> 0.001)           # true
println(d)                         # one row per term: AC, ESS (lag-1), ESS (Geyer), Geweke z, p
```
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
    chain_lengths = isempty(result.chain_lengths) ? [n_samples] : result.chain_lengths
    sum(chain_lengths) == n_samples || throw(ArgumentError(
        "mcmc_diagnostics: chain_lengths sum to $(sum(chain_lengths)) but the " *
        "sample has $n_samples rows"))

    autocorr = zeros(p)
    ess = zeros(p)
    ess_geyer = zeros(p)
    geweke_z = zeros(p)
    offset = 0
    for L in chain_lengths
        block = view(samples, (offset + 1):(offset + L), :)
        offset += L
        for j in 1:p
            x = view(block, :, j)
            # Lag-1 autocorrelation of this chain (around its own mean)
            ρ = 0.0
            if L >= 2
                x̄ = mean(x)
                γ0 = _autocov(x, x̄, 0)
                γ0 > 0 && (ρ = _autocov(x, x̄, 1) / γ0)
            end
            autocorr[j] += L * ρ
            ess[j] += abs(ρ) < 1 ? L * (1 - ρ) / (1 + ρ) : L
            ess_geyer[j] += _geyer_ess(x)
            z = L >= 4 ? _geweke_z(x) : 0.0
            abs(z) > abs(geweke_z[j]) && (geweke_z[j] = z)
        end
    end
    autocorr ./= n_samples
    geweke_p = z_pvalues(geweke_z)

    return MCMCDiagnostics(term_names, autocorr, ess, ess_geyer, geweke_z, geweke_p,
                           n_samples, collect(Int, chain_lengths))
end
