"""
Monte Carlo Maximum Likelihood Estimation (MCMLE) for ERGMs.

MCMLE uses MCMC sampling to approximate the likelihood function,
providing more accurate estimates than MPLE for models with strong dependencies.
"""

"""
    mcmle(model::ERGMModel; n_samples::Int=1000, burnin=nothing,
          interval=nothing, maxiter::Int=20, n_chains::Int=1,
          conv_threshold::Float64=0.1, hotelling_alpha::Float64=0.05,
          gamma0::Float64=0.1, max_step_norm::Float64=5.0,
          init=nothing, missing::Symbol=:error,
          obs_burnin=nothing, obs_interval=nothing,
          rng=Random.default_rng(), bridge_rungs::Int=16,
          bridge_samples=nothing, verbose::Bool=false) -> ERGMResult

Fit an ERGM using Monte Carlo Maximum Likelihood Estimation.

Starting from the MPLE estimates (or `init`), each iteration samples networks
at the current coefficients and takes a Hummel-style partial Newton step
toward the pseudo-target `γ·g(y_obs) + (1−γ)·ḡ`. The step length `γ ∈ (0, 1]`
starts at `gamma0` and adapts upward (at most doubling per iteration) while
the observed statistics lie outside the sampled statistic cloud; it reaches
`γ = 1` once the cloud covers them. Each Newton step is capped at Euclidean
norm `max_step_norm`.

Convergence is declared only at full step length (`γ = 1`) and when the
sampled statistics are statistically indistinguishable from the observed
ones: every per-statistic convergence t-ratio `(g_obs − ḡ)/sd(g)` must be
below `conv_threshold`, and a Hotelling T² test of the mean difference (using
the sampled covariance and an autocorrelation-adjusted effective sample size)
must be non-significant at level `hotelling_alpha`. The tests are
[`mcmc_convergence`](@ref), which is `public` so the variants (ERGMEgo's
moment matching) can use the same machinery.

# Arguments
- `model::ERGMModel`: The ERGM model specification
- `n_samples::Int=1000`: Number of MCMC samples per iteration (in total,
  over all chains)
- `burnin::Int`: Burn-in steps per chain. Defaults to `20 * n_dyads`
  ([`_mcmc_defaults`](@ref), the one dyad-scaled rule shared with every
  sampler), so mixing scales with network size
- `interval::Int`: Thinning interval. Defaults to `max(100, n_dyads ÷ 10)`
- `maxiter::Int=20`: Maximum MCMLE iterations (the ecosystem's iteration-cap
  keyword; `max_iter` is accepted as a deprecated spelling, with a one-time
  warning, until the next release)
- `n_chains::Int=1`: Independent MH chains per iteration (and for the final
  sample). The `n_samples` draws are split over the chains, each burned in
  separately from the observed network and seeded deterministically from
  `rng` exactly as [`sample_networks`](@ref) does, then concatenated; the
  effective sample size behind the Hotelling test is the sum of the
  per-chain Geyer ESSs. With several Julia threads the chains run in
  parallel — the result is identical at any thread count, which is why
  `n_chains` never defaults to `Threads.nthreads()`. `n_chains=1` is
  bit-for-bit the single-chain sampler.
- `conv_threshold::Float64=0.1`: Threshold for the per-statistic convergence
  t-ratios
- `hotelling_alpha::Float64=0.05`: Significance level of the Hotelling T²
  convergence test
- `gamma0::Float64=0.1`: Initial Hummel step length
- `max_step_norm::Float64=5.0`: Cap on the Euclidean norm of each Newton step
- `init::Vector{Float64}`: Starting coefficients (statnet's
  `control.ergm(init=)`). Default: the MPLE. Pass `coef(previous_fit)` to
  continue an unconverged fit from where it stopped.
- `missing::Symbol=:error`: Treatment of dyads masked as missing
  (`Networks.set_missing_dyad!`): `:error`, `:mle` or `:condition_on_face`
  — see the missing-data section below
- `obs_burnin::Int`, `obs_interval::Int`: Burn-in and thinning of the
  *constrained* chain under `missing=:mle` (statnet's `obs.MCMC.burnin` /
  `obs.MCMC.interval`). Default to the dyad-scaled rule applied to the
  number of masked dyads — the only dyads that chain moves
- `rng::AbstractRNG=Random.default_rng()`: Source of all random draws; runs
  with the same rng state are exactly reproducible
- `bridge_rungs::Int=16`: Number of path-sampling segments used for the
  final log-likelihood estimate (see `_bridge_loglik`). **`bridge_rungs=0`
  skips the estimate**: `loglik`, `aic` and `bic` are `NaN`, `show` prints
  "not estimated", and `approximations(fit)` records it. The bridge draws
  `bridge_rungs + 1` further MCMC samples of `bridge_samples` draws each
  after the fit is finished — on a single thread it was ~70 % of the total
  `mcmle` time in the panel's measurement — and it consumes randomness
  only *after* the final sample, so coefficients and standard errors are
  bit-identical with and without it.
- `bridge_samples::Int`: MCMC samples per bridge rung (default `n_samples`)
- `verbose::Bool=false`: Print progress

The `tol` keyword accepted by earlier versions is deprecated and ignored;
convergence is now assessed with the statistical tests described above.

# Non-convergence is loud

A fit that exhausts `maxiter` without passing both tests is returned with
`converged == false` **and** a warning quoting the last max t-ratio, the
Hotelling p-value and the step length; `Networks.approximations(fit)` lists
the non-convergence too, `show` prints the caveat under `Converged: false`,
and `fit.mcmc_convergence` carries the t-ratios, Hotelling p, effective
sample size, iteration count and step length recomputed on the final sample
at the returned coefficients. Continue such a fit with
`mcmle(model; init=coef(fit), ...)`.

# Standard errors

`vcov(fit)` is the inverse Fisher information estimated from the final MCMC
sample, `V = Σ̂⁻¹`, **plus the Monte-Carlo component** of the estimate
(Hunter & Handcock 2006, §3.3): the estimating equation `ḡ(θ) = g_obs` is
itself Monte Carlo, with `Var(ḡ) = Σ_mc` — the Geyer initial-sequence
asymptotic covariance of the sampled statistics, per-statistic on the
diagonal with the lag-0 correlations off-diagonal, divided by the number
of draws — so `vcov = V + V·Σ_mc·V`. The Fisher part alone is
`fit.vcov_fisher`, the Monte-Carlo standard error `sqrt(diag(V·Σ_mc·V))` is
[`mcmc_se`](@ref)`(fit)`, and `show` prints R's `summary.ergm` "MCMC %"
column: `round(100 · (se − se_fisher) / se)` per coefficient — the share
of the *standard error* (not of its variance) that the Monte-Carlo term
adds, R's `100 * (tot.se - mod.se) / tot.se`. At the default
`n_samples=1000` it rounds to 0 on a well-mixing chain; a large share says
the sample, not the data, is limiting the precision — raise `n_samples` or
`n_chains`.

# Boundary statistics

A statistic at the boundary of its attainable range (a `NodeMatch` with no
within-group tie, …) has no finite MLE; R ergm fixes the coefficient at
`-Inf` and drops the term. `mcmle` **refuses** such a model with an
`ArgumentError` quoting R's sentence: a `-Inf` coefficient has no Newton
update and the sampler could not honour it. Remove the term, or fit by
[`mple`](@ref), which reports the `∓Inf` coefficient and R's remaining
estimates.

# Missing data

A network with dyads masked as missing (`set_missing_dyad!`) is **rejected
by default**: `missing=:error` throws the shared `Networks.require_observed`
error, whose last bullets name the two policies `mcmle` accepts
(`missing_policies(mcmle) == (:error, :condition_on_face, :mle)`).

- `missing=:mle` — **missing-data maximum likelihood** (Handcock & Gile
  2010; what R `ergm()` does with NA ties). The likelihood is
  `Z_obs(θ) / Z(θ)`, the masked dyads integrated out. Each iteration runs
  two chains: the *free* chain toggles every dyad (masked ones included —
  the unconditional model) and gives `ḡ = Ê[g(Y)]` and `Σ_free`; the
  *constrained* chain toggles only the masked dyads, the observed dyads
  fixed at their observed values, and gives `ĝ_obs = Ê[g(Y) | Y_obs]` and
  `Σ_obs`. `ĝ_obs` replaces the observed statistics as the Newton target
  (`θ += γ Σ_free⁻¹ (ĝ_obs − ḡ)`), the convergence tests compare the two
  chains (the difference's variance is `Σ_free/n_eff_free +
  Σ_obs/n_eff_obs`), the Fisher information is `Σ_free − Σ_obs` (Eq. 9
  there; standard errors are `NaN`, with a warning, when it is not
  positive definite), and both chains contribute Monte-Carlo error to the
  standard errors. The log-likelihood `log Z_obs(θ) − log Z(θ)` is two
  path-sampling bridges (a `toggleable=:all` one from the exact
  dyad-independent normalizer over all dyads, a `toggleable=:masked` one
  from the exact normalizer over the masked dyads with the observed dyads
  fixed). The fit records `missing_method = :mle`; `nobs` is the number of
  observed dyads. Under dyad independence the estimate coincides with the
  available-case MPLE (the exact MLE of the observed dyads).
- `missing=:condition_on_face` — hold each masked dyad fixed at its stored
  face value: never toggled, and scored as recorded in the observed
  sufficient statistics. That is the model *conditional on the face
  values* — a different estimand from the missing-data MLE, right only when
  the stored values are true by construction (structural zeros, ties fixed
  by design). Explicit and warned; the fit records
  `missing_method = :condition_on_face`.

`mple` (which declares `Networks.supports_missing(mple) == true`) drops the
masked dyads from the pseudo-likelihood instead. The provenanced
`flomarriage_missing_ergm.toml` fixture pins `missing=:mle` against R
`ergm` on a masked flomarriage.

The reported log-likelihood (and hence AIC/BIC) is estimated by path
sampling along a `bridge_rungs`-segment ladder from a dyad-independent
reference distribution to the fitted coefficients — the standard
ergm-style bridge estimator — rather than a one-jump importance-sampling
estimate, which has unusably high variance when θ̂ is far from the
reference.

# Returns
- `ERGMResult`: Fitted model results. `mcmc_samples` holds the statistics
  sampled at the final coefficients (all chains concatenated; the free
  chain under `missing=:mle`; suitable for `mcmc_diagnostics`); samples
  from earlier iterations are discarded since they target different
  coefficient values.

# Example
```julia
using ERGM, Random
net = load_dataset(:florentine_marriage)
model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), net)
fit = mcmle(model; n_samples=500, rng=Xoshiro(1))
fit.converged                     # true
se_method(fit)                    # :fisher — inverse Fisher information (+ MC term) from the MCMC sample
maximum(mcmc_se(fit) ./ stderror(fit)) < 0.2   # true: the MC share is small
fast = mcmle(model; n_samples=500, rng=Xoshiro(1), bridge_rungs=0)
coef(fast) == coef(fit)           # true: the bridge only runs after the fit
isnan(loglikelihood(fast))        # true

# Missing-data MLE on a copy with two unobserved dyads
masked = copy(net); set_missing_dyad!(masked, 3, 4); set_missing_dyad!(masked, 1, 9)
fit_na = mcmle(ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), masked);
               n_samples=500, missing=:mle, rng=Xoshiro(2))
missing_method(fit_na)            # :mle
nobs(fit_na)                      # 118
```
"""
function mcmle(model::ERGMModel{T,D};
               n_samples::Int=1000,
               burnin::Union{Nothing,Int}=nothing,
               interval::Union{Nothing,Int}=nothing,
               maxiter::Int=20,
               n_chains::Int=1,
               conv_threshold::Float64=0.1,
               hotelling_alpha::Float64=0.05,
               gamma0::Float64=0.1,
               max_step_norm::Float64=5.0,
               init::Union{Nothing,AbstractVector{<:Real}}=nothing,
               rng::AbstractRNG=Random.default_rng(),
               bridge_rungs::Int=16,
               bridge_samples::Union{Nothing,Int}=nothing,
               tol::Union{Nothing,Real}=nothing,
               max_iter::Union{Nothing,Int}=nothing,
               missing::Symbol=:error,
               obs_burnin::Union{Nothing,Int}=nothing,
               obs_interval::Union{Nothing,Int}=nothing,
               verbose::Bool=false) where {T,D}
    if !isnothing(tol)
        @warn "The `tol` keyword to `mcmle` is deprecated and ignored: convergence " *
              "is now assessed with per-statistic t-ratios (`conv_threshold`) and a " *
              "Hotelling T² test (`hotelling_alpha`)." maxlog=1
    end
    if !isnothing(max_iter)
        # Deprecated spelling of the ecosystem-wide `maxiter` keyword (panel
        # 2026-09, item 16): honoured, with a one-time warning.
        @warn "The `max_iter` keyword to `mcmle` is deprecated; use `maxiter` " *
              "(the ecosystem-wide spelling). `max_iter=$max_iter` is honoured " *
              "for now." maxlog=1
        maxiter = max_iter
    end
    n_samples >= 2 || throw(ArgumentError("mcmle: n_samples must be ≥ 2 (got $n_samples)"))
    n_chains >= 1 || throw(ArgumentError("mcmle: n_chains must be ≥ 1 (got $n_chains)"))
    bridge_rungs >= 0 || throw(ArgumentError(
        "mcmle: bridge_rungs must be ≥ 0 (got $bridge_rungs); 0 skips the " *
        "log-likelihood estimate (loglik/AIC/BIC are then NaN)"))

    # Missing-data guard: masked dyads are rejected unless the caller has
    # explicitly asked for missing-data ML (`:mle`) or face-value
    # conditioning (`:condition_on_face`, a different estimand).
    missing_policy = missing
    missing_method = _guard_missing(model.network, missing_policy; context="mcmle",
                                    policies=_MCMLE_MISSING_POLICIES)
    missing_method === :condition_on_face &&
        _warn_condition_on_face(model.network, "MCMLE")
    mle = missing_method === :mle
    # The free chain: every unmasked dyad, or — under `:mle` — every dyad,
    # since the unconditional model integrates the masked ones out. The
    # samplers do not take `:mle` (it is an estimation policy): they run
    # under `:error`, which on an unmasked network is a no-op.
    free_toggle = mle ? :all : :free
    sampler_policy = missing_policy === :mle ? :error : missing_policy

    # Dyad-scaled MCMC defaults (the ONE rule every sampler shares): larger
    # networks need proportionally more toggles to mix. The constrained
    # chain moves only the masked dyads, so its budget scales with those.
    n_dyads = _n_dyads(model)
    burnin, interval = _resolve_mcmc_controls(model, burnin, interval;
                                              toggleable=free_toggle)
    obs_burnin, obs_interval = mle ?
        _resolve_mcmc_controls(model, obs_burnin, obs_interval; toggleable=:masked) :
        (0, 1)

    net = model.network
    terms = model.formula.terms
    term_names = terms.names
    p = length(terms)

    # A statistic at the boundary of its attainable range is refused BEFORE
    # any MPLE start is computed (no Newton step toward ∓Inf exists, and the
    # MPLE's own drop warning would be noise on the way to the refusal); the
    # pseudo-likelihood design it reads is built ONCE and reused for the
    # start below.
    design = _mple_data(net, terms, is_directed(model))
    _refuse_boundary_statistics(model, design...)

    # Start with MPLE estimates, unless the caller supplies a start. A
    # separated design (R: "The MPLE does not exist!") has no start to offer:
    # refuse, pointing at `init=`.
    if isnothing(init)
        verbose && println("Getting initial estimates via MPLE...")
        start = _mple_fit_design(design..., term_names; verbose=verbose, warn=false)
        start.separated && throw(ArgumentError(
            "mcmle: the MPLE used as the starting point does not exist (perfect " *
            "separation: a combination of the model's statistics predicts every " *
            "tie, so the pseudo-likelihood has no finite maximum; R ergm warns " *
            "\"The MPLE does not exist!\"). The observed statistics are likely " *
            "on the boundary of the model's convex hull, where no MLE exists " *
            "either. Remove or coarsen a term, or supply a starting point with " *
            "`init=`."))
        start.converged || @warn "mcmle: the MPLE used as the starting point did " *
            "not converge within its Newton iteration cap; starting from its " *
            "last iterate."
        θ = copy(start.coefficients)
    else
        length(init) == p || throw(ArgumentError(
            "mcmle: init has length $(length(init)) but the model has $p terms"))
        θ = Vector{Float64}(init)
    end

    # Observed statistics (face values; under `:mle` the target is the
    # constrained chain's mean instead, recomputed at every iteration)
    obs_stats = compute_all(terms, net)

    # Squared Mahalanobis radius within which the sampled cloud is considered
    # to cover a target point
    chisq_cut = quantile(Chisq(p), 0.95)

    converged = false
    γ = gamma0
    iterations = 0

    for iter in 1:maxiter
        iterations = iter
        if verbose
            println("MCMLE iteration $iter (step length γ = $(round(γ, digits=3)))...")
        end

        # Sample networks from current model (n_chains chains, concatenated)
        samples, chain_lengths = _mcmc_sample(model, θ, n_samples, burnin, interval;
                                              rng=rng, missing=sampler_policy,
                                              n_chains=n_chains,
                                              toggleable=free_toggle)
        # ... and, under `:mle`, the constrained chain whose mean is the target
        obs_samples, obs_chain_lengths = mle ?
            _mcmc_sample(model, θ, n_samples, obs_burnin, obs_interval;
                         rng=rng, n_chains=n_chains, toggleable=:masked) :
            (nothing, nothing)
        target = mle ? vec(mean(obs_samples, dims=1)) : obs_stats

        # Compute mean and covariance of sampled statistics
        mean_stats = vec(mean(samples, dims=1))
        cov_stats = cov(samples)
        sd_stats = sqrt.(max.(diag(cov_stats), 0.0))

        _warn_degenerate_stats(sd_stats, term_names)

        diff = target .- mean_stats

        F = cholesky(Symmetric(cov_stats); check=false)
        if !issuccess(F)
            source = iter == 1 ? "the initial estimates" :
                                 "the iteration-$(iter - 1) MCMLE update"
            @warn "The covariance matrix of the sampled statistics is singular at " *
                  "iteration $iter (collinear statistics, a degenerate model, or a " *
                  "collapsed sampler). MCMLE cannot take further Newton steps; the " *
                  "returned coefficients are $source, unrefined, and standard errors " *
                  "will be NaN. Check the model for degeneracy or redundant terms."
            break
        end

        # Squared Mahalanobis distance of the target statistics from the
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
            tests = mcmc_convergence(samples, target;
                                     conv_threshold=conv_threshold,
                                     hotelling_alpha=hotelling_alpha,
                                     chain_lengths=chain_lengths,
                                     target_samples=obs_samples,
                                     target_chain_lengths=obs_chain_lengths)
            if tests.converged
                converged = true
                if verbose
                    println("Converged at iteration $iter (max t-ratio " *
                            "$(round(maximum(tests.t_ratios), digits=4)), Hotelling T² " *
                            "p-value $(round(tests.hotelling_p, digits=4)))")
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

    # Final sample at the returned coefficients: the basis of the standard
    # errors, of `mcmc_diagnostics`, and of the recorded convergence report
    final_samples, chain_lengths = _mcmc_sample(model, θ, n_samples, burnin, interval;
                                                rng=rng, missing=sampler_policy,
                                                n_chains=n_chains,
                                                toggleable=free_toggle)
    final_obs, obs_chain_lengths = mle ?
        _mcmc_sample(model, θ, n_samples, obs_burnin, obs_interval;
                     rng=rng, n_chains=n_chains, toggleable=:masked) :
        (nothing, nothing)
    target = mle ? vec(mean(final_obs, dims=1)) : obs_stats
    tests = mcmc_convergence(final_samples, target;
                             conv_threshold=conv_threshold,
                             hotelling_alpha=hotelling_alpha,
                             chain_lengths=chain_lengths,
                             target_samples=final_obs,
                             target_chain_lengths=obs_chain_lengths)
    convergence = MCMLEConvergence((iterations, γ, tests.t_ratios,
                                    tests.hotelling_p, tests.n_eff))

    # Covariance of θ̂: the inverse Fisher information from the final sample
    # (Σ̂⁻¹, or (Σ̂_free − Σ̂_obs)⁻¹ under missing-data ML), plus the
    # Monte-Carlo component V·Σ_mc·V of the estimating equation (Hunter &
    # Handcock 2006 §3.3), Σ_mc = Var(ḡ) (+ Var(ĝ_obs)) from the Geyer
    # initial-sequence estimate (see `_mc_cov_of_mean`)
    vcov_fisher, var_cov, std_errors, mc_se =
        _mcmle_covariance(final_samples, chain_lengths, final_obs, obs_chain_lengths, p)

    z_values = θ ./ std_errors
    p_values = z_pvalues(z_values)

    # Non-convergence is loud: warned here with the diagnostics of the
    # returned estimate, recorded in `converged`/`mcmc_convergence` (hence in
    # `approximations(fit)` and `show`) so a reader and a machine both see it.
    converged || @warn "MCMLE did not converge in maxiter=$maxiter iterations " *
        "(last max t-ratio $(_fmt3(maximum(tests.t_ratios))), Hotelling p " *
        "$(_fmt3(tests.hotelling_p)), step length γ $(_fmt3(γ))): the estimates " *
        "are the last iterate and the standard errors are unreliable; increase " *
        "maxiter/n_samples/burnin, check the model for degeneracy " *
        "(`mcmc_diagnostics`), or refit from these coefficients " *
        "(`mcmle(model; init=coef(fit))`)"

    # Path-sampled (bridge) log-likelihood for AIC/BIC — optional, and run
    # only now so that it consumes randomness after everything above
    if bridge_rungs == 0
        loglik = aic = bic = NaN
    else
        nb = something(bridge_samples, n_samples)
        loglik = if mle
            # log L(θ; y_obs) = log Z_obs(θ) − log Z(θ): two bridges, each
            # from the exact dyad-independent normalizer of its own dyad set
            logZ_all = _bridge_logZ(model, θ; nrungs=bridge_rungs, n_samples=nb,
                                    burnin=burnin, interval=interval, rng=rng,
                                    toggleable=:all)
            logZ_obs = _bridge_logZ(model, θ; nrungs=bridge_rungs, n_samples=nb,
                                    burnin=obs_burnin, interval=obs_interval,
                                    rng=rng, toggleable=:masked)
            logZ_obs - logZ_all
        else
            _bridge_loglik(model, θ, obs_stats;
                           nrungs=bridge_rungs, n_samples=nb,
                           burnin=burnin, interval=interval, rng=rng,
                           missing=sampler_policy)
        end
        aic = -2 * loglik + 2 * p
        bic = -2 * loglik + p * log(n_dyads)
    end

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
        :mcmc,
        missing_method,
        vcov_fisher,
        mc_se,
        convergence,
        chain_lengths,
        nothing
    )
end

# A statistic at the boundary of its attainable range has no finite MLE (R
# fixes the coefficient at -Inf and drops the term); MCMLE cannot take a
# Newton step on it and the sampler could not honour a -Inf coefficient, so
# the model is refused with R's sentence and the two ways out.
function _refuse_boundary_statistics(model::ERGMModel,
                                     X=nothing, n_tot=nothing, n_one=nothing)
    terms = model.formula.terms
    if X === nothing
        X, n_tot, n_one = _mple_data(model.network, terms, is_directed(model))
    end
    boundary = _boundary_columns(X, n_tot, n_one)
    isempty(boundary) && return nothing
    lo = [terms.names[j] for (j, s) in boundary if s === :min]
    hi = [terms.names[j] for (j, s) in boundary if s === :max]
    parts = String[]
    isempty(lo) || push!(parts, "observed statistic(s) $(join(lo, ", ")) are at their " *
                                "smallest attainable values (coefficient -Inf)")
    isempty(hi) || push!(parts, "observed statistic(s) $(join(hi, ", ")) are at their " *
                                "largest attainable values (coefficient +Inf)")
    throw(ArgumentError(
        "mcmle: " * join(parts, "; ") * ". No finite maximum-likelihood estimate " *
        "exists (R ergm warns \"Their coefficients will be fixed at -Inf\" and drops " *
        "the term), and MCMLE cannot take a Newton step toward ±Inf. Remove the " *
        "term(s) from the formula, or fit with `method=:mple`, which reports the " *
        "±Inf coefficient and R's estimates of the remaining terms."))
end

# Three significant digits for log messages (Inf/NaN print as themselves).
# `@sprintf("%.3g")`, not `round(x; sigdigits=3)`: the rounded Float64 is not
# exactly representable, so `string` printed `6.969999999999999e-32` and
# `1.6699999999999998e33` in diagnostics (ERGMCount round 3).
_fmt3(x::Real) = isfinite(x) ? @sprintf("%.3g", x) : string(x)

# The routine-level missing-data vocabulary (Networks.missing_policies): the
# policies `mcmle`'s `missing=` keyword actually accepts — NOT the generic
# `:face`. Tooling (the capability matrix) prints this instead of a literal.
missing_policies(::typeof(mcmle)) = _MCMLE_MISSING_POLICIES

# MCMLE has a principled treatment of masked dyads — missing-data maximum
# likelihood behind `missing=:mle` (the default still refuses, as the trait's
# own example in Networks.jl does: opting in is explicit, never implicit).
supports_missing(::typeof(mcmle)) = true

"""
    _mcmle_covariance(samples, chain_lengths, obs_samples, obs_chain_lengths, p)
        -> (vcov_fisher, vcov, std_errors, mcmc_se)

The MCMLE covariance from the final sample(s). `V = Σ̂⁻¹` with
`Σ̂ = cov(samples)`, or — under missing-data ML, when `obs_samples` (the
constrained chain) is given — `V = (Σ̂_free − Σ̂_obs)⁻¹`, the Fisher
information of the observed-data likelihood (Handcock & Gile 2010, Eq. 9),
which is checked for positive definiteness first: a difference that is not
PD (too few draws for the two chains to separate, or a degenerate model)
yields `NaN` standard errors with a warning rather than a nonsense
covariance. The Monte-Carlo term is `V·Σ_mc·V` with `Σ_mc` the Geyer
covariance of the free chain's mean plus that of the constrained chain's
mean (`_mc_cov_of_mean`). Any failure to invert yields NaNs with a warning.

`public`: a variant with its own MCMC MLE (ERGMRank's `method=:mcmle`) gets
the same Fisher + Monte-Carlo covariance as `mcmle`, so its standard errors
include the MCMC-error component rather than a locally re-derived Geyer
covariance.

# Example
```julia
using ERGM, Random
rng = Xoshiro(1)
samples = randn(rng, 2000, 2)             # statistics sampled at θ̂, one chain of 2000
V, Vtot, se, mcse = ERGM._mcmle_covariance(samples, [2000], nothing, nothing, 2)
isapprox(V, [1.0 0.0; 0.0 1.0]; atol=0.15)  # true: Σ̂⁻¹ of unit-variance draws
all(mcse .< se)                             # true: `se` includes the MC component
```
"""
function _mcmle_covariance(samples::Matrix{Float64}, chain_lengths::Vector{Int},
                           obs_samples::Union{Nothing,Matrix{Float64}},
                           obs_chain_lengths::Union{Nothing,Vector{Int}}, p::Int)
    nan = (fill(NaN, p, p), fill(NaN, p, p), fill(NaN, p), fill(NaN, p))
    cov_stats = cov(samples)
    if obs_samples === nothing
        info = cov_stats
    else
        info = cov_stats .- cov(obs_samples)
        info = (info .+ info') ./ 2
        if !issuccess(cholesky(Symmetric(info); check=false))
            @warn "Missing-data MCMLE: the Fisher information Σ_free − Σ_obs " *
                  "(variance of the sampled statistics minus their variance " *
                  "conditional on the observed dyads) is not positive definite, so " *
                  "no standard errors are available: `stderror`, z-values and " *
                  "p-values are NaN. The point estimates are unaffected. Increase " *
                  "n_samples (both chains), or check the model for degeneracy."
            return nan
        end
    end
    try
        V = Matrix(inv(Symmetric(info)))
        Σ_mc = _mc_cov_of_mean(samples, chain_lengths)
        if obs_samples !== nothing
            Σ_mc = Σ_mc .+ _mc_cov_of_mean(obs_samples, obs_chain_lengths)
        end
        V_mc = V * Σ_mc * V
        V_mc = (V_mc .+ V_mc') ./ 2                  # exactly symmetric
        Vtot = V .+ V_mc
        return V, Vtot, sqrt.(diag(Vtot)), sqrt.(max.(diag(V_mc), 0.0))
    catch
        @warn "The covariance matrix of the final MCMC sample could not be inverted, " *
              "so no MCMC-based standard errors are available: `stderror`, z-values " *
              "and p-values are all NaN. The point estimates are the last MCMLE " *
              "iterates (the starting estimates if no Newton step succeeded)."
        return nan
    end
end

"""
    _warn_degenerate_stats(sd_stats, term_names)

Warn when any sampled statistic has near-zero variance across the MCMC
sample. This indicates either a degenerate model (the sampler collapsed onto
a full, empty, or otherwise frozen graph) or a sampler that is not mixing.
Returns `nothing`; `public` so the variants' MCMC MLEs (ERGMRank) emit the
one sentence.

# Example
```julia
using ERGM
ERGM._warn_degenerate_stats([0.0, 0.8], ["edges", "triangle"])   # warns about `edges`
ERGM._warn_degenerate_stats([0.5, 0.8], ["edges", "triangle"]) === nothing   # true, silent
```
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
    _effective_sample_size(samples, chain_lengths=[size(samples, 1)]) -> Float64

Smallest per-statistic effective sample size across the columns of `samples`,
using the Geyer initial-sequence estimator (see `_geyer_ess` in
`src/mcmc/diagnostics.jl`), which accounts for autocorrelation at all lags —
the lag-1-only estimate it replaces was systematically optimistic. When the
rows are the concatenation of several independent chains (`chain_lengths`
gives the consecutive block lengths), the per-statistic ESS is the SUM of
the per-chain ESSs — independent chains add information, and a Geyer
estimate across the seam between two chains would be meaningless. Clamped
to `[2, n_samples]`.
"""
function _effective_sample_size(samples::AbstractMatrix{<:Real},
                                chain_lengths::AbstractVector{<:Integer}=[size(samples, 1)])
    n_samples = size(samples, 1)
    n_eff = Float64(n_samples)
    for j in 1:size(samples, 2)
        ess_j = 0.0
        offset = 0
        for L in chain_lengths
            ess_j += _geyer_ess(view(samples, (offset + 1):(offset + L), j))
            offset += L
        end
        n_eff = min(n_eff, ess_j)
    end
    return max(n_eff, 2.0)
end

"""
    mcmc_convergence(samples::AbstractMatrix, targets::AbstractVector;
                     conv_threshold=0.1, hotelling_alpha=0.05,
                     chain_lengths=[size(samples, 1)],
                     target_samples=nothing, target_chain_lengths=nothing)
        -> (converged, t_ratios, hotelling_p, n_eff)

The MCMLE convergence tests, as one `public` function (panel 2026-09, item
24): given an `n × p` matrix of statistics sampled at the current
coefficients and the `p` target (observed) statistics, compute

- `t_ratios`: the per-statistic convergence t-ratios
  `|target_j − mean_j| / sd_j` (`Inf` for a statistic with zero sampled
  variance — a collapsed sampler cannot confirm convergence);
- `n_eff`: the effective sample size — the smallest over statistics of the
  Geyer initial-sequence ESS, summed over chains when `chain_lengths` says
  the rows are several independent chains concatenated;
- `hotelling_p`: the p-value of the Hotelling T² test that the sampled mean
  equals the targets, using the sampled covariance and `n_eff` (0.0 when the
  covariance is singular or `n_eff ≤ p`, where the test is undefined);
- `converged`: `maximum(t_ratios) < conv_threshold && hotelling_p >
  hotelling_alpha`.

When the targets are themselves a Monte-Carlo mean — the constrained chain
of the missing-data MCMLE — pass its draws as `target_samples` (with
`target_chain_lengths`): `targets` must then equal their column means, the
per-draw sd behind the t-ratios becomes `sqrt(var_j + var_target_j)`, and
the Hotelling test uses the variance of the *difference* of the two means,
`Σ/n_eff + Σ_target/n_eff_target`.

`mcmle` calls this at every full-step-length iteration and once more on the
final sample (recorded as `fit.mcmc_convergence`). ERGMEgo's moment matching
and any other equation-solving estimator can use it instead of an ad-hoc
relative-change rule.

# Example
```julia
using ERGM, Random, Statistics
rng = Xoshiro(1)
draws = randn(rng, 2000, 2) .+ [1.0 5.0]           # a stationary sample around (1, 5)
ok = ERGM.mcmc_convergence(draws, [1.0, 5.0])
ok.converged                                        # true
off = ERGM.mcmc_convergence(draws, [3.0, 5.0])      # 2 sd away on statistic 1
off.converged, off.t_ratios[1] > 1.5                # (false, true)
two = ERGM.mcmc_convergence(draws, [1.0, 5.0]; chain_lengths=[1000, 1000])
two.n_eff > 1000                                    # true: ESS adds over chains
```
"""
function mcmc_convergence(samples::AbstractMatrix{<:Real},
                          targets::AbstractVector{<:Real};
                          conv_threshold::Real=0.1,
                          hotelling_alpha::Real=0.05,
                          chain_lengths::AbstractVector{<:Integer}=[size(samples, 1)],
                          target_samples::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                          target_chain_lengths::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    n, p = size(samples)
    length(targets) == p || throw(ArgumentError(
        "mcmc_convergence: $(length(targets)) targets for $p sampled statistics"))
    n >= 2 || throw(ArgumentError("mcmc_convergence: need at least 2 samples (got $n)"))
    sum(chain_lengths) == n || throw(ArgumentError(
        "mcmc_convergence: chain_lengths sum to $(sum(chain_lengths)) but there are $n rows"))
    all(>=(1), chain_lengths) || throw(ArgumentError(
        "mcmc_convergence: every chain must have at least one sample"))

    mean_stats = vec(mean(samples, dims=1))
    cov_stats = cov(samples)
    diff = Float64.(targets) .- mean_stats
    n_eff = _effective_sample_size(samples, chain_lengths)

    # Variance of the estimating function per draw, and of the difference of
    # means (scaled to one free draw) for the Hotelling test
    if target_samples === nothing
        var_draw = max.(diag(cov_stats), 0.0)
        cov_diff = cov_stats
    else
        size(target_samples, 2) == p || throw(ArgumentError(
            "mcmc_convergence: target_samples has $(size(target_samples, 2)) " *
            "columns for $p statistics"))
        size(target_samples, 1) >= 2 || throw(ArgumentError(
            "mcmc_convergence: need at least 2 target samples"))
        tcl = something(target_chain_lengths, [size(target_samples, 1)])
        sum(tcl) == size(target_samples, 1) || throw(ArgumentError(
            "mcmc_convergence: target_chain_lengths sum to $(sum(tcl)) but there " *
            "are $(size(target_samples, 1)) target rows"))
        cov_t = cov(target_samples)
        n_eff_t = _effective_sample_size(target_samples, tcl)
        var_draw = max.(diag(cov_stats) .+ diag(cov_t), 0.0)
        cov_diff = cov_stats .+ cov_t .* (n_eff / n_eff_t)
    end
    sd_stats = sqrt.(var_draw)
    t_ratios = [sd_stats[j] > 0 ? abs(diff[j]) / sd_stats[j] : Inf for j in 1:p]

    F = cholesky(Symmetric(cov_diff); check=false)
    hotelling_p = if issuccess(F)
        d2 = max(dot(diff, F \ diff), 0.0)
        _hotelling_pvalue(d2, n_eff, p)
    else
        0.0
    end
    converged = maximum(t_ratios) < conv_threshold && hotelling_p > hotelling_alpha
    return (converged=converged, t_ratios=t_ratios, hotelling_p=hotelling_p,
            n_eff=n_eff)
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
                 rng=Random.default_rng(), missing=:error, n_chains=1,
                 toggleable=:free)
        -> (samples::Matrix{Float64}, chain_lengths::Vector{Int})

Generate MCMC samples of network statistics at `θ`, every chain starting
from the observed network. With `n_chains == 1` this is exactly one
[`mh_sample`](@ref) call on the caller's `rng` (bit-identical to the
single-chain sampler). With more chains the `n_samples` draws are split as
`sample_networks` splits them (the first `n_samples % n_chains` chains get
one extra), one seed per chain is drawn from `rng` in order, and the chains
run on separate tasks; their statistics are concatenated in chain order,
so the result is independent of the thread count. `chain_lengths` gives
the consecutive block lengths for the chain-aware ESS.
`missing` is the caller's already-validated missing-dyad policy;
`toggleable` (`:free`, `:all`, `:masked`) is the dyad set each chain may
toggle — the free and constrained chains of the missing-data MCMLE.
"""
function _mcmc_sample(model::ERGMModel, θ::Vector{Float64},
                      n_samples::Int, burnin::Int, interval::Int;
                      rng::AbstractRNG=Random.default_rng(),
                      missing::Symbol=:error, n_chains::Int=1,
                      toggleable::Symbol=:free)
    n_chains = clamp(n_chains, 1, n_samples)
    if n_chains == 1
        stats = mh_sample(model, θ; n_samples=n_samples, burnin=burnin,
                          interval=interval, rng=rng, missing=missing,
                          toggleable=toggleable).stats
        return stats, [n_samples]
    end

    counts = fill(n_samples ÷ n_chains, n_chains)
    for c in 1:(n_samples % n_chains)
        counts[c] += 1
    end
    seeds = rand(rng, UInt64, n_chains)
    p = length(θ)
    chain_stats = Vector{Matrix{Float64}}(undef, n_chains)
    missing_policy = missing
    @sync for c in 1:n_chains
        Threads.@spawn begin
            chain_rng = Random.Xoshiro(seeds[c])
            chain_stats[c] = mh_sample(model, θ; n_samples=counts[c],
                                       burnin=burnin, interval=interval,
                                       rng=chain_rng, missing=missing_policy,
                                       toggleable=toggleable).stats
        end
    end
    samples = Matrix{Float64}(undef, n_samples, p)
    offset = 0
    for c in 1:n_chains
        samples[(offset + 1):(offset + counts[c]), :] = chain_stats[c]
        offset += counts[c]
    end
    return samples, counts
end

"""
    _copy_network(net) -> Network

Create an attribute-preserving copy of a network. Delegates to `Base.copy`,
which duplicates the graph and all vertex/edge/network attributes and
preserves the `directed`, `bipartite`, and `loops` settings (and the
missing-dyad mask). `public`: the variants copy their sampler states with it.

# Example
```julia
using ERGM
net = load_dataset(:florentine_marriage)
c = ERGM._copy_network(net)
rem_edge!(c, 1, 9)
(ne(net), ne(c))                                            # (20, 19)
get_vertex_attribute(c, :wealth) == get_vertex_attribute(net, :wealth)   # true
```
"""
_copy_network(net::Network) = copy(net)

"""
    _dyad_independent_logZ(model, θ; toggleable=:free) -> Float64

Exact log-normalizer of a **dyad-independent** ERGM: when every coordinate
of `θ` with a dyad-dependent term is zero, the model factorizes over dyads
and

    log Z(θ) = Σ_dyads log(1 + exp(θ'δ(i,j))),

where `δ(i,j)` are the (state-independent) change statistics. Uses the
compressed MPLE rows for the unmasked dyads, so the cost is O(unique rows)
plus O(masked dyads).

`toggleable` names the dyad set the sum runs over — the three normalizers
the bridges need on a network with masked (missing) dyads, each holding the
other dyads fixed at their stored value (a fixed present edge contributes
the constant `θ'δ(i,j)` to every configuration of the free ones):

- `:free` — the unmasked dyads, masked ones fixed at their face value: the
  normalizer of the `:condition_on_face` model.
- `:all` — every dyad: the unconditional `log Z(θ)`.
- `:masked` — the masked dyads only, the observed ones fixed at their
  observed value: `log Z_obs(θ)`, the normalizer of the observed-data
  likelihood `Z_obs(θ)/Z(θ)`.

Without masked dyads the three coincide (`:masked` has no free dyad and
returns `θ'g(y)`).
"""
function _dyad_independent_logZ(model::ERGMModel, θ::Vector{Float64};
                                toggleable::Symbol=:free)
    toggleable in _TOGGLEABLE || throw(ArgumentError(
        "_dyad_independent_logZ: toggleable must be :free, :all or :masked"))
    net = model.network
    terms = model.formula.terms
    logZ = 0.0

    # The unmasked dyads: summed over when free, fixed at their observed
    # value when only the masked dyads are free
    if toggleable === :masked
        for e in edges(net)
            i, j = Int(src(e)), Int(dst(e))
            is_missing_dyad(net, i, j) && continue
            logZ += dot(θ, change_stat_all(terms, net, i, j))
        end
    else
        X, n_tot, _ = _mple_data(net, terms, is_directed(model))
        η = X * θ
        @inbounds for r in eachindex(η)
            logZ += n_tot[r] * _log1pexp(η[r])
        end
    end

    # The masked dyads: fixed at face value (a present edge contributes
    # θ'δ(i,j) to every configuration of the free dyads) or free
    for (i, j) in missing_dyads(net)
        δ = change_stat_all(terms, net, i, j)
        if toggleable === :free
            has_edge(net, i, j) && (logZ += dot(θ, δ))
        else
            logZ += _log1pexp(dot(θ, δ))
        end
    end
    return logZ
end

"""
    _bridge_logZ(model, θ; nrungs=16, n_samples=500, burnin, interval, rng,
                 missing=:error, toggleable=:free) -> Float64

Path-sampling (bridge) estimate of the log-normalizer `log Z(θ)` over the
`toggleable` dyad set (see [`_dyad_independent_logZ`](@ref) for the three
sets), the quantity behind MCMLE's log-likelihood.

Let `θ₀` be `θ` with every dyad-dependent coordinate set to zero (per
[`is_dyad_dependent`](@ref)). `θ₀` defines a dyad-independent reference
whose normalizer `log Z(θ₀)` is computed exactly. Along the linear path
`θ(u) = θ₀ + u·(θ − θ₀)`, the thermodynamic identity

    d/du log Z(θ(u)) = E_{θ(u)}[g(Y)]' (θ − θ₀)

is integrated by the trapezoid rule over `nrungs` equal segments
(`nrungs + 1` grid points), with `E_{θ(u)}[g]` estimated by an MCMC run of
`n_samples` draws at each grid point — a chain over the same `toggleable`
dyad set. Rungs are embarrassingly parallel and run on separate threads,
each with an RNG seeded deterministically from `rng` (results are
thread-count-independent). This is the standard ergm-style bridge
estimator; the one-jump reverse importance-sampling estimate it replaces
has unusably high variance when `θ` is far from the reference.

For fully dyad-independent models `θ₀ = θ` and the exact normalizer is
returned with no Monte Carlo error.

`missing` is the caller's already-validated missing-dyad policy for a
`toggleable=:free` chain (with `:condition_on_face` every rung and the
exact reference are the *conditional* ones given the masked dyads' face
values); the `:all`/`:masked` chains take none.

`public`: ERGMRank's `method=:mcmle` reports an absolute `loglikelihood`
through the same bridge.

# Example
```julia
using ERGM
flo = load_dataset(:florentine_marriage)
model = ERGMModel(ERGMFormula([Edges()]), flo)
# Dyad-independent: the exact normalizer, no Monte Carlo (120 dyads)
ERGM._bridge_logZ(model, [-1.0]) ≈ 120 * log1p(exp(-1.0))   # true
```
"""
function _bridge_logZ(model::ERGMModel, θ::Vector{Float64};
                      nrungs::Int=16, n_samples::Int=500,
                      burnin::Int=10000, interval::Int=100,
                      rng::AbstractRNG=Random.default_rng(),
                      missing::Symbol=:error, toggleable::Symbol=:free)
    nrungs >= 1 || throw(ArgumentError("nrungs must be at least 1 (mcmle skips " *
                                       "the bridge entirely for bridge_rungs=0)"))
    terms = model.formula.terms

    # Dyad-independent reference: zero out the dyad-dependent coordinates
    θ0 = copy(θ)
    for (k, t) in enumerate(terms)
        is_dyad_dependent(t) && (θ0[k] = 0.0)
    end
    logZ0 = _dyad_independent_logZ(model, θ0; toggleable=toggleable)

    Δ = θ .- θ0
    all(iszero, Δ) && return logZ0          # dyad-independent: exact

    # Trapezoid ladder over u ∈ [0, 1]; per-rung E_{θ(u)}[g]'Δ estimated by
    # MCMC, one deterministic seed per rung
    us = range(0.0, 1.0; length=nrungs + 1)
    seeds = rand(rng, UInt64, length(us))
    contrib = Vector{Float64}(undef, length(us))
    missing_policy = missing
    @sync for k in eachindex(us)
        Threads.@spawn begin
            θu = θ0 .+ us[k] .* Δ
            rung_rng = Random.Xoshiro(seeds[k])
            stats = mh_sample(model, θu; n_samples=n_samples, burnin=burnin,
                              interval=interval, rng=rung_rng,
                              missing=missing_policy,
                              toggleable=toggleable).stats
            contrib[k] = dot(Δ, vec(mean(stats, dims=1)))
        end
    end

    h = 1.0 / nrungs
    return logZ0 + h * (0.5 * contrib[1] + sum(@view contrib[2:end-1]) +
                        0.5 * contrib[end])
end

"""
    _bridge_loglik(model, θ, obs_stats; nrungs=16, n_samples=500,
                   burnin, interval, rng, missing=:error) -> Float64

Path-sampling (bridge) estimate of the log-likelihood
`θ'g(y_obs) − log Z(θ)` — the estimator behind MCMLE's AIC/BIC for a fully
observed network (or the `:condition_on_face` model, whose normalizer is
the conditional one): `dot(θ, obs_stats)` minus [`_bridge_logZ`](@ref) over
the free dyads. The missing-data MLE's log-likelihood is instead the
difference of two `_bridge_logZ` calls (`:masked` minus `:all`); see
[`mcmle`](@ref).
"""
function _bridge_loglik(model::ERGMModel, θ::Vector{Float64},
                        obs_stats::Vector{Float64};
                        nrungs::Int=16, n_samples::Int=500,
                        burnin::Int=10000, interval::Int=100,
                        rng::AbstractRNG=Random.default_rng(),
                        missing::Symbol=:error)
    logZ = _bridge_logZ(model, θ; nrungs=nrungs, n_samples=n_samples,
                        burnin=burnin, interval=interval, rng=rng,
                        missing=missing, toggleable=:free)
    return dot(θ, obs_stats) - logZ
end
