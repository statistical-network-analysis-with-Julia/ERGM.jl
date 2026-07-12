"""
Monte Carlo Maximum Likelihood Estimation (MCMLE) for ERGMs.

MCMLE uses MCMC sampling to approximate the likelihood function,
providing more accurate estimates than MPLE for models with strong dependencies.
"""

"""
    mcmle(model::ERGMModel; n_samples::Int=1000, burnin=nothing,
          interval=nothing, max_iter::Int=20, conv_threshold::Float64=0.1,
          hotelling_alpha::Float64=0.05, gamma0::Float64=0.1,
          max_step_norm::Float64=5.0, verbose::Bool=false) -> ERGMResult

Fit an ERGM using Monte Carlo Maximum Likelihood Estimation.

Starting from the MPLE estimates, each iteration samples networks at the
current coefficients and takes a Hummel-style partial Newton step toward the
pseudo-target `γ·g(y_obs) + (1−γ)·ḡ`. The step length `γ ∈ (0, 1]` starts at
`gamma0` and adapts upward (at most doubling per iteration) while the observed
statistics lie outside the sampled statistic cloud; it reaches `γ = 1` once
the cloud covers them. Each Newton step is capped at Euclidean norm
`max_step_norm`.

Convergence is declared only at full step length (`γ = 1`) and when the
sampled statistics are statistically indistinguishable from the observed
ones: every per-statistic convergence t-ratio `(g_obs − ḡ)/sd(g)` must be
below `conv_threshold`, and a Hotelling T² test of the mean difference (using
the sampled covariance and an autocorrelation-adjusted effective sample size)
must be non-significant at level `hotelling_alpha`.

# Arguments
- `model::ERGMModel`: The ERGM model specification
- `n_samples::Int=1000`: Number of MCMC samples per iteration
- `burnin::Int`: Burn-in steps. Defaults to `20 * n_dyads`, so mixing scales
  with network size
- `interval::Int`: Thinning interval. Defaults to `max(100, n_dyads ÷ 10)`
- `max_iter::Int=20`: Maximum MCMLE iterations
- `conv_threshold::Float64=0.1`: Threshold for the per-statistic convergence
  t-ratios
- `hotelling_alpha::Float64=0.05`: Significance level of the Hotelling T²
  convergence test
- `gamma0::Float64=0.1`: Initial Hummel step length
- `max_step_norm::Float64=5.0`: Cap on the Euclidean norm of each Newton step
- `rng::AbstractRNG=Random.default_rng()`: Source of all random draws; runs
  with the same rng state are exactly reproducible
- `bridge_rungs::Int=16`: Number of path-sampling segments used for the
  final log-likelihood estimate (see `_bridge_loglik`)
- `bridge_samples::Int`: MCMC samples per bridge rung (default `n_samples`)
- `verbose::Bool=false`: Print progress

The `tol` keyword accepted by earlier versions is deprecated and ignored;
convergence is now assessed with the statistical tests described above.

If the network has dyads masked as missing (`Network.set_missing_dyad!`),
MCMLE **conditions on them at their face value**: the MH sampler never
toggles them and the observed statistics include them as stored. This is an
honest approximation, not full missing-data ML (which is future work); a
one-time warning is emitted. MPLE, in contrast, excludes masked dyads from
the pseudo-likelihood entirely.

The reported log-likelihood (and hence AIC/BIC) is estimated by path
sampling along a `bridge_rungs`-segment ladder from a dyad-independent
reference distribution to the fitted coefficients — the standard
ergm-style bridge estimator — rather than a one-jump importance-sampling
estimate, which has unusably high variance when θ̂ is far from the
reference.

# Returns
- `ERGMResult`: Fitted model results. `mcmc_samples` holds the statistics
  sampled at the final coefficients (suitable for `mcmc_diagnostics`);
  samples from earlier iterations are discarded since they target different
  coefficient values.
"""
function mcmle(model::ERGMModel{T};
               n_samples::Int=1000,
               burnin::Union{Nothing,Int}=nothing,
               interval::Union{Nothing,Int}=nothing,
               max_iter::Int=20,
               conv_threshold::Float64=0.1,
               hotelling_alpha::Float64=0.05,
               gamma0::Float64=0.1,
               max_step_norm::Float64=5.0,
               rng::AbstractRNG=Random.default_rng(),
               bridge_rungs::Int=16,
               bridge_samples::Union{Nothing,Int}=nothing,
               tol::Union{Nothing,Real}=nothing,
               verbose::Bool=false) where T
    if !isnothing(tol)
        @warn "The `tol` keyword to `mcmle` is deprecated and ignored: convergence " *
              "is now assessed with per-statistic t-ratios (`conv_threshold`) and a " *
              "Hotelling T² test (`hotelling_alpha`)." maxlog=1
    end

    n_masked = n_missing_dyads(model.network)
    if n_masked > 0
        @warn "The network has $n_masked missing (unobserved) dyads. MCMLE " *
              "conditions on them at their face value: masked dyads are held " *
              "fixed (edge present/absent as stored) during MCMC and enter " *
              "the observed sufficient statistics at that face value. Full " *
              "maximum likelihood under missing data (statnet-style missing-" *
              "data MCMC) is not yet implemented." maxlog=1
    end

    # Dyad-scaled MCMC defaults: larger networks need proportionally more
    # toggles to mix
    n_dyads = _n_dyads(model)
    burnin = something(burnin, 20 * n_dyads)
    interval = something(interval, max(100, n_dyads ÷ 10))

    # Start with MPLE estimates
    if verbose
        println("Getting initial estimates via MPLE...")
    end

    mple_result = mple(model)
    θ = copy(mple_result.coefficients)

    net = model.network
    terms = model.formula.terms
    term_names = terms.names
    p = length(terms)

    # Observed statistics
    obs_stats = compute_all(terms, net)

    # Squared Mahalanobis radius within which the sampled cloud is considered
    # to cover a target point
    chisq_cut = quantile(Chisq(p), 0.95)

    converged = false
    γ = gamma0

    for iter in 1:max_iter
        if verbose
            println("MCMLE iteration $iter (step length γ = $(round(γ, digits=3)))...")
        end

        # Sample networks from current model
        samples = _mcmc_sample(model, θ, n_samples, burnin, interval; rng=rng)

        # Compute mean and covariance of sampled statistics
        mean_stats = vec(mean(samples, dims=1))
        cov_stats = cov(samples)
        sd_stats = sqrt.(max.(diag(cov_stats), 0.0))

        _warn_degenerate_stats(sd_stats, term_names)

        diff = obs_stats .- mean_stats

        F = cholesky(Symmetric(cov_stats); check=false)
        if !issuccess(F)
            source = iter == 1 ? "the initial MPLE estimates" :
                                 "the iteration-$(iter - 1) MCMLE update"
            @warn "The covariance matrix of the sampled statistics is singular at " *
                  "iteration $iter (collinear statistics, a degenerate model, or a " *
                  "collapsed sampler). MCMLE cannot take further Newton steps; the " *
                  "returned coefficients are $source, unrefined, and standard errors " *
                  "will be NaN. Check the model for degeneracy or redundant terms."
            break
        end

        # Squared Mahalanobis distance of the observed statistics from the
        # sampled cloud
        d2 = max(dot(diff, F \ diff), 0.0)

        # Hummel step-length adaptation: use the largest fraction of the way
        # from the sampled mean to the observed statistics that stays inside
        # the cloud, allowing γ to at most double per iteration; γ = 1 once
        # the cloud covers the target.
        if d2 <= chisq_cut
            γ = 1.0
        else
            γ = clamp(min(sqrt(chisq_cut / d2), 2.0 * γ), 0.01, 1.0)
        end

        # Check convergence: only at full step length, and only when both
        # statistical tests pass
        if γ == 1.0
            t_ratios = [sd_stats[j] > 0 ? abs(diff[j]) / sd_stats[j] : Inf for j in 1:p]
            n_eff = _effective_sample_size(samples)
            hotelling_p = _hotelling_pvalue(d2, n_eff, p)
            if maximum(t_ratios) < conv_threshold && hotelling_p > hotelling_alpha
                converged = true
                if verbose
                    println("Converged at iteration $iter (max t-ratio " *
                            "$(round(maximum(t_ratios), digits=4)), Hotelling T² " *
                            "p-value $(round(hotelling_p, digits=4)))")
                end
                break
            end
        end

        # Partial Newton step toward the pseudo-target x_γ = γ·obs + (1−γ)·mean:
        # θ_new = θ + Σ⁻¹(x_γ − ḡ) = θ + γ·Σ⁻¹(g_obs − ḡ), capped in norm
        delta = F \ (γ .* diff)
        step_norm = norm(delta)
        if step_norm > max_step_norm
            delta .*= max_step_norm / step_norm
        end
        θ .+= delta
    end

    # Compute final statistics
    final_samples = _mcmc_sample(model, θ, n_samples, burnin, interval; rng=rng)
    cov_stats = cov(final_samples)

    # Covariance of θ̂ from the inverse Fisher information
    var_cov, std_errors = try
        V = Matrix(inv(Symmetric(cov_stats)))
        V, sqrt.(diag(V))
    catch
        @warn "The covariance matrix of the final MCMC sample could not be inverted, " *
              "so no MCMC-based standard errors are available: `stderror`, z-values " *
              "and p-values are all NaN. The point estimates are the last MCMLE " *
              "iterates (the MPLE estimates if no Newton step succeeded)."
        fill(NaN, p, p), fill(NaN, p)
    end

    z_values = θ ./ std_errors
    p_values = _z_pvalues(z_values)

    # Path-sampled (bridge) log-likelihood for AIC/BIC
    loglik = _bridge_loglik(model, θ, obs_stats;
                            nrungs=bridge_rungs,
                            n_samples=something(bridge_samples, n_samples),
                            burnin=burnin, interval=interval, rng=rng)

    aic = -2 * loglik + 2 * p
    bic = -2 * loglik + p * log(n_dyads)

    return ERGMResult(
        model,
        θ,
        std_errors,
        z_values,
        p_values,
        var_cov,
        loglik,
        aic,
        bic,
        :mcmle,
        converged,
        final_samples,
        :mcmc
    )
end

"""
    _warn_degenerate_stats(sd_stats, term_names)

Warn when any sampled statistic has near-zero variance across the MCMC
sample. This indicates either a degenerate model (the sampler collapsed onto
a full, empty, or otherwise frozen graph) or a sampler that is not mixing.
"""
function _warn_degenerate_stats(sd_stats::Vector{Float64}, term_names::Vector{String})
    for j in eachindex(sd_stats)
        if sd_stats[j] < 1e-8
            @warn "Sampled statistic '$(term_names[j])' has near-zero variance " *
                  "across the MCMC sample. The model is likely degenerate or the " *
                  "sampler has collapsed (all sampled networks nearly identical); " *
                  "estimates and standard errors from this fit are unreliable." maxlog=1
        end
    end
end

"""
    _effective_sample_size(samples) -> Float64

Smallest per-statistic effective sample size across the columns of `samples`,
using the Geyer initial-sequence estimator (see `_geyer_ess` in
`src/mcmc/diagnostics.jl`), which accounts for autocorrelation at all lags —
the lag-1-only estimate it replaces was systematically optimistic. Clamped
to `[2, n_samples]`.
"""
function _effective_sample_size(samples::Matrix{Float64})
    n_samples = size(samples, 1)
    n_eff = Float64(n_samples)
    for j in 1:size(samples, 2)
        n_eff = min(n_eff, _geyer_ess(view(samples, :, j)))
    end
    return max(n_eff, 2.0)
end

"""
    _hotelling_pvalue(d2, n_eff, p) -> Float64

P-value of the Hotelling T² test that the sampled statistic mean equals the
observed statistics, given the squared Mahalanobis distance `d2` between them
(under the per-draw covariance), effective sample size `n_eff`, and number of
statistics `p`. Returns 0.0 when `n_eff ≤ p`, where the test is undefined and
convergence cannot be confirmed.
"""
function _hotelling_pvalue(d2::Float64, n_eff::Float64, p::Int)
    n_eff > p || return 0.0
    T2 = n_eff * d2
    f_stat = T2 * (n_eff - p) / (p * (n_eff - 1))
    return ccdf(FDist(p, n_eff - p), f_stat)
end

"""
    _mcmc_sample(model, θ, n_samples, burnin, interval;
                 rng=Random.default_rng()) -> Matrix{Float64}

Generate MCMC samples of network statistics, starting from the observed
network. Thin compatibility wrapper over the public [`mh_sample`](@ref).
"""
function _mcmc_sample(model::ERGMModel, θ::Vector{Float64},
                      n_samples::Int, burnin::Int, interval::Int;
                      rng::AbstractRNG=Random.default_rng())
    return mh_sample(model, θ; n_samples=n_samples, burnin=burnin,
                     interval=interval, rng=rng).stats
end

"""
    _copy_network(net) -> Network

Create an attribute-preserving copy of a network. Delegates to `Base.copy`,
which duplicates the graph and all vertex/edge/network attributes and
preserves the `directed`, `bipartite`, and `loops` settings.
"""
_copy_network(net::Network) = copy(net)

"""
    _dyad_independent_logZ(model, θ) -> Float64

Exact log-normalizer of a **dyad-independent** ERGM: when every coordinate
of `θ` with a dyad-dependent term is zero, the model factorizes over dyads
and

    log Z(θ) = Σ_dyads log(1 + exp(θ'δ(i,j))),

where `δ(i,j)` are the (state-independent) change statistics. Uses the
compressed MPLE rows, so the cost is O(unique rows).

With masked (missing) dyads the normalizer is that of the *conditional*
distribution given the masked dyads' face values: the sum above runs over
the free dyads only (the compressed rows already exclude masked dyads), and
each masked dyad whose face value is an edge adds the constant `θ'δ(i,j)`.
"""
function _dyad_independent_logZ(model::ERGMModel, θ::Vector{Float64})
    X, n_tot, _ = _mple_data(model.network, model.formula.terms, model.directed)
    η = X * θ
    logZ = 0.0
    @inbounds for r in eachindex(η)
        logZ += n_tot[r] * _log1pexp(η[r])
    end
    # Masked dyads are conditioned on at face value: a masked present edge
    # contributes θ'δ(i,j) to every configuration of the free dyads.
    net = model.network
    for (i, j) in missing_dyads(net)
        has_edge(net, i, j) || continue
        logZ += dot(θ, change_stat_all(model.formula.terms, net, i, j))
    end
    return logZ
end

"""
    _bridge_loglik(model, θ, obs_stats; nrungs=16, n_samples=500,
                   burnin, interval, rng) -> Float64

Path-sampling (bridge) estimate of the log-likelihood
`θ'g(y_obs) − log Z(θ)` — the estimator behind MCMLE's AIC/BIC.

Let `θ₀` be `θ` with every dyad-dependent coordinate set to zero (per
[`is_dyad_dependent`](@ref)). `θ₀` defines a dyad-independent reference
whose normalizer `log Z(θ₀)` is computed exactly
([`_dyad_independent_logZ`](@ref)). Along the linear path
`θ(u) = θ₀ + u·(θ − θ₀)`, the thermodynamic identity

    d/du log Z(θ(u)) = E_{θ(u)}[g(Y)]' (θ − θ₀)

is integrated by the trapezoid rule over `nrungs` equal segments
(`nrungs + 1` grid points), with `E_{θ(u)}[g]` estimated by an MCMC run of
`n_samples` draws at each grid point. Rungs are embarrassingly parallel and
run on separate threads, each with an RNG seeded deterministically from
`rng` (results are thread-count-independent). This is the standard
ergm-style bridge estimator; the one-jump reverse importance-sampling
estimate it replaces has unusably high variance when `θ` is far from the
reference.

For fully dyad-independent models `θ₀ = θ` and the exact log-likelihood is
returned with no Monte Carlo error.
"""
function _bridge_loglik(model::ERGMModel, θ::Vector{Float64},
                        obs_stats::Vector{Float64};
                        nrungs::Int=16, n_samples::Int=500,
                        burnin::Int=10000, interval::Int=100,
                        rng::AbstractRNG=Random.default_rng())
    nrungs >= 1 || throw(ArgumentError("nrungs must be at least 1"))
    terms = model.formula.terms

    # Dyad-independent reference: zero out the dyad-dependent coordinates
    θ0 = copy(θ)
    for (k, t) in enumerate(terms)
        is_dyad_dependent(t) && (θ0[k] = 0.0)
    end
    logZ0 = _dyad_independent_logZ(model, θ0)

    Δ = θ .- θ0
    if all(iszero, Δ)
        # Dyad-independent model: the log-likelihood is exact
        return dot(θ, obs_stats) - logZ0
    end

    # Trapezoid ladder over u ∈ [0, 1]; per-rung E_{θ(u)}[g]'Δ estimated by
    # MCMC, one deterministic seed per rung
    us = range(0.0, 1.0; length=nrungs + 1)
    seeds = rand(rng, UInt64, length(us))
    contrib = Vector{Float64}(undef, length(us))
    @sync for k in eachindex(us)
        Threads.@spawn begin
            θu = θ0 .+ us[k] .* Δ
            rung_rng = Random.Xoshiro(seeds[k])
            stats = mh_sample(model, θu; n_samples=n_samples, burnin=burnin,
                              interval=interval, rng=rung_rng).stats
            contrib[k] = dot(Δ, vec(mean(stats, dims=1)))
        end
    end

    h = 1.0 / nrungs
    logZ = logZ0 + h * (0.5 * contrib[1] + sum(@view contrib[2:end-1]) +
                        0.5 * contrib[end])
    return dot(θ, obs_stats) - logZ
end
