"""
Monte Carlo Maximum Likelihood Estimation (MCMLE) for ERGMs.

MCMLE uses MCMC sampling to approximate the likelihood function,
providing more accurate estimates than MPLE for models with strong dependencies.
"""

"""
    mcmle(model::ERGMModel; n_samples::Int=1000, burnin=nothing,
          interval=nothing, maxiter::Int=60, n_chains::Int=1,
          termination::Symbol=:confidence, conv_precision=0.1,
          conv_confidence=0.99, max_n_samples=nothing,
          proposal::Symbol=:spdyad, effective_size=64, drop=true,
          gamma0::Float64=0.1, max_step_norm::Float64=5.0,
          init=nothing, missing::Symbol=:error,
          obs_burnin=nothing, obs_interval=nothing,
          rng=Random.default_rng(), bridge_rungs::Int=16,
          bridge_samples=nothing, verbose::Bool=false) -> ERGMResult

Fit an ERGM using Monte Carlo Maximum Likelihood Estimation.

Starting from the MPLE estimates (or `init`), **every iteration** — the
first included, as in statnet — samples networks at the current
coefficients and takes a Hummel-style partial Newton step toward the
pseudo-target `γ·g(y_obs) + (1−γ)·ḡ` (`_hummel_step` (internal)). The step
length `γ ∈ (0, 1]` starts at `gamma0` and adapts upward (at most doubling
per iteration) while the observed statistics lie outside the sampled
statistic cloud; it reaches `γ = 1` once the cloud covers them. Each Newton
step is capped at Euclidean norm `max_step_norm`. (A dyad-independent
formula started from its MPLE is already at the exact MLE: no step is
taken, the iterations only confirm the moment equation.)

**Convergence** is declared only at full step length (`γ = 1`), by R ergm
4's `confidence` rule (`termination=:confidence`,
[`confidence_test`](@ref ERGM.Extension.confidence_test)): the estimating equation at the updated
coefficients must lie, with `conv_confidence` (99 %) Monte-Carlo
confidence, inside a tolerance region of `conv_precision` (0.1) of the
statistics' variance. When the test fails near the solution the next
iteration's sample is enlarged (up to `max_n_samples`, default
`16 × n_samples`), so the rule is attainable under Monte-Carlo noise. The
estimate returned by a converged fit is the update from the sample that
passed, and that sample is the one the standard errors, `mcmc_samples`,
`fit.mcmc_convergence` and `fit.termination` describe.
`termination=:hotelling` is the pre-0.2 rule: every per-statistic t-ratio
below `conv_threshold` and a non-significant Hotelling T² test at
`hotelling_alpha` ([`mcmc_convergence`](@ref)).

# Arguments
- `model::ERGMModel`: The ERGM model specification
- `n_samples::Int=1000`: Number of MCMC samples per iteration (in total,
  over all chains)
- `burnin::Int`: Burn-in steps per chain. Defaults to `20 * n_dyads`
  ([`mcmc_defaults`](@ref ERGM.Extension.mcmc_defaults), the one dyad-scaled rule shared with every
  sampler), so mixing scales with network size
- `interval::Int`: Thinning interval. Defaults to `max(100, n_dyads ÷ 10)`
- `maxiter::Int=60`: Maximum MCMLE iterations (R's `MCMLE.maxit`; the
  ecosystem's iteration-cap keyword)
- `n_chains::Int=1`: Independent MH chains per iteration (and for the final
  sample). The `n_samples` draws are split over the chains, each burned in
  separately from the observed network and seeded deterministically from
  `rng` exactly as [`sample_networks`](@ref) does, then concatenated; the
  effective sample size behind the Hotelling test is the sum of the
  per-chain Geyer ESSs. With several Julia threads the chains run in
  parallel — the result is identical at any thread count, which is why
  `n_chains` never defaults to `Threads.nthreads()`. `n_chains=1` is
  bit-for-bit the single-chain sampler.
- `termination::Symbol=:confidence`, `conv_precision=0.1`,
  `conv_confidence=0.99`, `max_n_samples=16·n_samples`: the stopping rule
  (above) and the cap on the boosted sample size
- `conv_threshold::Float64=0.1`, `hotelling_alpha::Float64=0.05`: the
  t-ratio threshold and Hotelling level of `termination=:hotelling`
- `proposal::Symbol=:spdyad`: the MH proposal of every chain. The default
  is R ergm 4's: its `MCMC.prop = ~sparse + .triadic` selects the SPDyad
  proposal (TNT mixed with a shared-partner-focused proposal) for every
  one-mode network, whatever the terms — and ERGM.jl's models are one-mode
  (R falls back to TNT only for bipartite networks, which ERGM.jl refuses).
  `:tnt` (tie/no-tie) and `:random` (see
  [`mh_sample`](@ref)) remain available. Both `:spdyad` and `:tnt` are exact
  samplers of the model; SPDyad mixes several times faster on models with
  triadic terms at about twice the cost per toggle.
- `effective_size=64`: **ESS-adaptive sampling** (R's
  `MCMLE.effectiveSize`, 64 in R, the default). An iteration's sample is
  not a fixed `n_samples` draws: the chain is continued from the previous
  iteration and extended — its thinning interval doubling — until the
  effective sample size of the sampled statistics reaches the target, and
  the stopping rule's boost raises that target. `n_samples` then only sets
  the constrained chain's and the bridge's sample sizes, and `interval` the
  starting interval (an eighth of it). `effective_size=nothing` restores
  fixed-size sampling: `n_samples` draws `interval` toggles apart per
  iteration, boosted up to `max_n_samples`.
  `mcmle(model; proposal=:tnt, effective_size=nothing)` is the fixed-size
  tie/no-tie design
- `drop::Bool=true`: R's `control.ergm(drop=TRUE)` — a statistic at the
  boundary of its attainable range is fixed at `∓Inf` and the rest estimated
  (see "Boundary statistics" below); `drop=false` refuses such a model
- `gamma0::Float64=0.1`: Initial Hummel step length
- `max_step_norm::Float64=5.0`: Cap on the Euclidean norm of each Newton step
- `init::Vector{Float64}`: Starting coefficients (statnet's
  `control.ergm(init=)`). Default: the MPLE. Pass `coef(previous_fit)` to
  continue an unconverged fit from where it stopped.
- `missing::Symbol=:error`: Treatment of dyads masked as missing
  (`NetworkCore.set_missing_dyad!`): `:error`, `:mle` or `:condition_on_face`
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
  `mcmle` time on the Florentine and Faux Mesa High fits — and it consumes randomness
  only *after* the final sample, so coefficients and standard errors are
  bit-identical with and without it.
- `bridge_samples::Int`: MCMC samples per bridge rung (default `n_samples`)
- `verbose::Bool=false`: Print progress

# Non-convergence is loud

A fit that exhausts `maxiter` without passing the stopping rule is returned
with `converged == false` **and** a warning quoting the rule's p-value, the
last max t-ratio, the Hotelling p-value and the step length; `NetworkCore.approximations(fit)` lists
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

A statistic at the boundary of its attainable range has no finite MLE: a
`NodeMatch` level with no within-group tie, a `NodeMix` cell of a singleton
level (no dyad at all), a `Triangle` on a triangle-free network, a
`Degree(1)` that no vertex has. As R's `ergm()` does under its default
`drop=TRUE`, `mcmle` fixes such a coefficient at `-Inf` (`+Inf` at the
largest value), warns with R's sentence, and estimates the rest with the
statistic held at its bound — an infinite offset, which the samplers honour
as a constraint (no move off the bound is accepted). The fixed coefficient
is reported with standard error 0 and p-value 0, `dof`/AIC/BIC count only
the estimated coefficients, and `show` and `NetworkCore.approximations(fit)`
say which coefficients were fixed. When a fixed statistic is dyad-dependent
(`Triangle`), the log-likelihood is not estimated (`NaN`), as for an
infinite dyad-dependent `Offset`. Detection is R's `ergm.checkextreme.model`
test (the observed statistic against the term's attainable range) together
with the pseudo-likelihood design's boundary test, which also sees, e.g., a
`NodeCov` whose positive-change dyads are all empty. `drop=false` refuses
such a model with an `ArgumentError` instead (R's `drop=FALSE`, which keeps
the term in a model whose "MLE is poorly defined", is not implemented).

A dyad-independent statistic that does not vary at all, or is a linear
combination of the statistics before it, has no identifiable coefficient:
it is held at 0 (any value gives the same model) and reported as `NaN`,
with R's warning, as R's MPLE reports `NA`. A dyad-dependent one leaves the
MPLE start undefined and is refused unless `init=` is given.

# Missing data

A network with dyads masked as missing (`set_missing_dyad!`) is **rejected
by default**: `missing=:error` throws the shared `NetworkCore.require_observed`
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

`mple` (which declares `NetworkCore.supports_missing(mple) == true`) drops the
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
               maxiter::Int=60,
               n_chains::Int=1,
               termination::Symbol=:confidence,
               conv_precision::Float64=0.1,
               conv_confidence::Float64=0.99,
               max_n_samples::Union{Nothing,Int}=nothing,
               conv_threshold::Float64=0.1,
               hotelling_alpha::Float64=0.05,
               gamma0::Float64=0.1,
               max_step_norm::Float64=5.0,
               init::Union{Nothing,AbstractVector{<:Real}}=nothing,
               rng::AbstractRNG=Random.default_rng(),
               bridge_rungs::Int=16,
               bridge_samples::Union{Nothing,Int}=nothing,
               missing::Symbol=:error,
               obs_burnin::Union{Nothing,Int}=nothing,
               obs_interval::Union{Nothing,Int}=nothing,
               proposal::Symbol=:spdyad,
               effective_size::Union{Nothing,Real}=64,
               drop::Bool=true,
               verbose::Bool=false) where {T,D}
    n_samples >= 2 || throw(ArgumentError("mcmle: n_samples must be ≥ 2 (got $n_samples)"))
    termination in (:confidence, :hotelling) || throw(ArgumentError(
        "mcmle: termination must be :confidence (R ergm's default equivalence " *
        "test) or :hotelling (the pre-0.2 t-ratio + Hotelling rule); got " *
        "$(repr(termination))"))
    0 < conv_precision || throw(ArgumentError(
        "mcmle: conv_precision must be positive (got $conv_precision)"))
    0 < conv_confidence < 1 || throw(ArgumentError(
        "mcmle: conv_confidence must be in (0, 1) (got $conv_confidence)"))
    n_max = something(max_n_samples, 16 * n_samples)
    n_max >= n_samples || throw(ArgumentError(
        "mcmle: max_n_samples ($n_max) must be ≥ n_samples ($n_samples)"))
    _check_proposal(proposal; context="mcmle")
    adaptive = effective_size !== nothing
    adaptive && !(effective_size >= 8) && throw(ArgumentError(
        "mcmle: effective_size must be ≥ 8 (R's MCMLE.effectiveSize is 64); got " *
        "$effective_size"))
    # A model with curved terms (decay estimated) goes to the curved MCMLE
    _has_curved(model) && return _mcmle_curved(model;
        n_samples=n_samples, burnin=burnin, interval=interval, maxiter=maxiter,
        n_chains=n_chains, termination=termination, conv_precision=conv_precision,
        conv_confidence=conv_confidence, n_max=n_max, conv_threshold=conv_threshold,
        hotelling_alpha=hotelling_alpha, gamma0=gamma0, max_step_norm=max_step_norm,
        init=init, rng=rng, bridge_rungs=bridge_rungs, bridge_samples=bridge_samples,
        missing=missing, proposal=proposal, effective_size=effective_size,
        drop=drop, verbose=verbose)
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
    n_dyads = n_observed_dyads(model)
    burnin, interval = _resolve_mcmc_controls(model, burnin, interval;
                                              toggleable=free_toggle)
    obs_burnin, obs_interval = mle ?
        _resolve_mcmc_controls(model, obs_burnin, obs_interval; toggleable=:masked) :
        (0, 1)

    net = model.network
    terms = model.formula.terms
    term_names = terms.names
    p = length(terms)

    # `Offset` terms: their coefficients are fixed, so every Newton step,
    # convergence test and covariance runs on the estimated coordinates
    # `free` only, while the samplers run on the full θ
    offset_mask, offset_vals = _offset_info(model)
    any(offset_mask) && all(offset_mask) && throw(ArgumentError(
        "mcmle: every coefficient is fixed by an offset; there is nothing to " *
        "estimate (use `mple` to evaluate the model)"))

    # A statistic at the boundary of its attainable range has no finite MLE.
    # R's `ergm()` (its default `drop=TRUE`) fixes the coefficient at ∓Inf
    # and fits the rest, and so does `mcmle`: the statistic becomes a fixed
    # coordinate — an infinite offset, which the samplers honour as a
    # constraint on the sample space — with R's message. Decided BEFORE any
    # MPLE start is computed; the pseudo-likelihood design is built ONCE and
    # reused for the start below.
    design = _mple_data(net, terms, is_directed(model))
    extreme = extreme_statistics(terms, net)
    boundary = _model_boundary(model, design..., extreme)
    if !isempty(boundary)
        drop || _refuse_no_drop(model, design, extreme; context="mcmle")
        _warn_drop(term_names, boundary; context="mcmle")
    end
    fixed_mask, fixed_vals = copy(offset_mask), copy(offset_vals)
    for (j, side) in boundary
        fixed_mask[j] = true
        fixed_vals[j] = side === :min ? -Inf : Inf
    end
    dropped = [j for (j, _) in boundary]

    # A statistic that does not vary on the fitted dyads, or is a linear
    # combination of the statistics before it, has no identifiable
    # coefficient. Dyad-independent, that holds on every network: it is held
    # at 0 (any value gives the same model) and reported as NaN, as R's MPLE
    # reports NA. A dyad-dependent one is singular only at the observed
    # network, and its MPLE start does not exist: refused unless `init=` is
    # given.
    # (on the rows the offsets leave free and the fixed statistics do not
    # touch: R's drop leaves the other coefficients those rows)
    isempty(findall(!, fixed_mask)) && throw(ArgumentError(
        "mcmle: every coefficient is fixed — " *
        join(("$(term_names[j]) = $(fixed_vals[j])" for j in 1:p), ", ") *
        " (by an offset, or at the boundary of its attainable range); there is " *
        "nothing to estimate. `mple` reports such a model."))
    free0, keep0, _ = _offset_design(design..., term_names, offset_mask, offset_vals)
    rows0 = [r for r in findall(keep0) if all(design[1][r, j] == 0 for j in dropped)]
    aliased = _aliased_columns(design[1], rows0, setdiff(free0, dropped))
    if !isempty(aliased)
        dep = [term_names[j] for j in aliased if is_dyad_dependent(terms[j])]
        (isempty(dep) || init !== nothing) || throw(ArgumentError(
            "mcmle: statistic(s) $(join(dep, ", ")) do not vary on the observed " *
            "network (every change statistic is 0), or are linear combinations of " *
            "the statistics before them, so the MPLE that starts the MCMLE does not " *
            "exist (R ergm warns \"Model statistics ... are not varying\"). Remove " *
            "the term(s), or give a starting point with `init=`."))
        indep = [j for j in aliased if !is_dyad_dependent(terms[j])]
        _warn_aliased(term_names, indep, design[1], rows0; context="mcmle",
                      estimate="MLE")
        for j in indep
            fixed_mask[j] = true
            fixed_vals[j] = 0.0
        end
    else
        indep = Int[]
    end

    free = findall(!, fixed_mask)
    isempty(free) && throw(ArgumentError(
        "mcmle: every coefficient is fixed — " *
        join(("$(term_names[j]) = $(fixed_vals[j])" for j in 1:p), ", ") *
        " (by an offset, at the boundary of its attainable range, or not " *
        "identifiable); there is nothing to estimate. `mple` reports such a model."))
    # An infinite fixed coefficient on a dyad-DEPENDENT statistic is a
    # state-dependent constraint (e.g. `Offset(Triangle(), -Inf)`, or a
    # triangle-free network's `Triangle()` fixed at -Inf: no tie may close a
    # triangle). The sampler honours it, but there is no dyad-independent
    # reference to bridge from, so the log-likelihood is not estimated.
    constrained_dd = any(k -> isinf(fixed_vals[k]) && is_dyad_dependent(terms[k]),
                         findall(fixed_mask))
    pf = length(free)

    # Start with MPLE estimates, unless the caller supplies a start. A
    # separated design (R: "The MPLE does not exist!") has no start to offer:
    # refuse, pointing at `init=`.
    if isnothing(init)
        verbose && println("Getting initial estimates via MPLE...")
        # the true offsets as offsets; the dropped statistics through the
        # MPLE's own drop (the rows they touch removed, in R's order); an
        # unidentified column comes back NaN and is held at 0 below
        start = _mple_fit_offsets(design..., term_names, offset_mask, offset_vals;
                                  extreme=boundary, verbose=verbose, warn=false)
        start.separated && throw(ArgumentError(
            "mcmle: the MPLE used as the starting point does not exist (separation " *
            "on $(join(("`$t`" for t in start.separated_terms), ", ")): a " *
            "combination of the model's statistics predicts every " *
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
    θ[fixed_mask] .= fixed_vals[fixed_mask]     # offsets and drops stay fixed

    # Observed statistics (face values; under `:mle` the target is the
    # constrained chain's mean instead, recomputed at every iteration)
    obs_stats = Vector{Float64}(compute_all(terms, net))   # (a user term may return an Int)

    # A dyad-independent formula started from its MPLE starts AT the exact
    # MLE (the pseudo-likelihood is the likelihood — also the observed-data
    # likelihood under `missing=:mle`, which factorizes over dyads): a
    # Monte-Carlo step could only add noise to it, so none is taken and the
    # iterations only confirm the moment equation. Every other fit takes a
    # step at EVERY iteration (statnet's order), the first included.
    take_step = has_dyad_dependent(model) || !isnothing(init)

    # The free chain's sampler (`mcmle_sampler`): ESS-adaptive by default
    # (`effective_size=`, R's MCMLE.effectiveSize) — the chain is continued
    # from iteration to iteration and each iteration's sample is extended,
    # its thinning interval doubling, until its effective sample size reaches
    # the target; the stopping rule's boost then raises the TARGET, not a raw
    # draw count
    sampler = mcmle_sampler(model; effective_size=effective_size, n_samples=n_samples,
                            max_n_samples=n_max, n_free=pf, burnin=burnin,
                            interval=interval, n_chains=n_chains, proposal=proposal,
                            toggleable=free_toggle, missing=sampler_policy, rng=rng,
                            verbose=verbose)

    # One iteration's sample at the free coordinates `θf`: the free chain
    # and, under `:mle`, the constrained chain whose mean is the target
    θ_full = copy(θ)
    function draw(θf, n)
        θ_full[free] = θf
        samples, chain_lengths = sampler.draw(θ_full, n)
        obs_samples, obs_chain_lengths = mle ?
            _mcmc_sample(model, θ_full, n, obs_burnin, obs_interval;
                         rng=rng, n_chains=n_chains, toggleable=:masked) :
            (nothing, nothing)
        (mle && pf < p) && (obs_samples = obs_samples[:, free])
        target = mle ? vec(mean(obs_samples, dims=1)) : obs_stats[free]
        return (samples=pf == p ? samples : samples[:, free],
                chain_lengths=chain_lengths, target=target, obs_samples=obs_samples,
                obs_chain_lengths=obs_chain_lengths, aux=samples)
    end
    sol = mcmle_solve(draw, θ[free]; labels=term_names[free],
                       n_samples=sampler.n_samples,
                       maxiter=maxiter, termination=termination,
                       conv_precision=conv_precision, conv_confidence=conv_confidence,
                       conv_threshold=conv_threshold, hotelling_alpha=hotelling_alpha,
                       gamma0=gamma0, max_step_norm=max_step_norm, take_step=take_step,
                       resize=sampler.resize, verbose=verbose)
    θ[free] = sol.coef
    converged, γ, tests = sol.converged, sol.step_length, sol.tests
    final_samples, chain_lengths = sol.final.aux, sol.final.chain_lengths
    convergence = MCMLEConvergence((sol.iterations, γ, tests.t_ratios,
                                    tests.hotelling_p, tests.n_eff))
    termination_report = (rule=termination, p_value=sol.termination_p,
                          precision=conv_precision, confidence=conv_confidence,
                          n_samples=size(final_samples, 1))

    # Covariance of θ̂ (from the driver): the inverse Fisher information from
    # the final sample (Σ̂⁻¹, or (Σ̂_free − Σ̂_obs)⁻¹ under missing-data ML),
    # plus the Monte-Carlo component V·Σ_mc·V of the estimating equation
    vcov_fisher, var_cov, std_errors, mc_se =
        sol.vcov_fisher, sol.vcov, sol.se, sol.mcmc_se
    if pf < p
        # An offset is fixed: zero rows and columns, standard error 0 (R)
        embed(M) = (E = zeros(p, p); E[free, free] = M; E)
        vcov_fisher, var_cov = embed(vcov_fisher), embed(var_cov)
        se_f, mc_f = std_errors, mc_se
        std_errors = zeros(p); std_errors[free] = se_f
        mc_se = zeros(p); mc_se[free] = mc_f
    end

    z_values = θ ./ std_errors
    p_values = z_pvalues(z_values)
    # A coefficient without identification is reported as NaN (R's NA), not
    # as the 0 it was held at (the bridge below still runs at 0)
    θ_report = copy(θ)
    for j in indep
        θ_report[j] = NaN; std_errors[j] = NaN; z_values[j] = NaN; p_values[j] = NaN
        mc_se[j] = NaN
        var_cov[j, :] .= NaN; var_cov[:, j] .= NaN
        vcov_fisher[j, :] .= NaN; vcov_fisher[:, j] .= NaN
    end

    # Non-convergence is loud: warned here with the diagnostics of the
    # returned estimate, recorded in `converged`/`mcmc_convergence` (hence in
    # `approximations(fit)` and `show`) so a reader and a machine both see it.
    converged || @warn "MCMLE did not converge in maxiter=$maxiter iterations " *
        "($(_termination_verdict(termination_report, tests.t_ratios, γ))): the estimates " *
        "are the last iterate and the standard errors are unreliable; increase " *
        "maxiter/n_samples/burnin, check the model for degeneracy " *
        "(`mcmc_diagnostics`), or refit from these coefficients " *
        "(`mcmle(model; init=coef(fit))`)"

    # Path-sampled (bridge) log-likelihood for AIC/BIC — optional, and run
    # only now so that it consumes randomness after everything above
    if bridge_rungs == 0 || constrained_dd
        loglik = aic = bic = NaN
    else
        nb = something(bridge_samples, n_samples)
        loglik = if mle
            # log L(θ; y_obs) = log Z_obs(θ) − log Z(θ): two bridges, each
            # from the exact dyad-independent normalizer of its own dyad set
            logZ_all = _bridge_logZ(model, θ; nrungs=bridge_rungs, n_samples=nb,
                                    burnin=burnin, interval=interval, rng=rng,
                                    toggleable=:all, proposal=proposal)
            logZ_obs = _bridge_logZ(model, θ; nrungs=bridge_rungs, n_samples=nb,
                                    burnin=obs_burnin, interval=obs_interval,
                                    rng=rng, toggleable=:masked)
            logZ_obs - logZ_all
        else
            _bridge_loglik(model, θ, obs_stats;
                           nrungs=bridge_rungs, n_samples=nb,
                           burnin=burnin, interval=interval, rng=rng,
                           missing=sampler_policy, proposal=proposal)
        end
        # The estimated coefficients only (R's df); a -Inf offset's forbidden
        # dyads are not observations (R's logLik nobs)
        n_bic = any(isinf, fixed_vals) ?
            sum(design[2][_offset_design(design..., term_names, fixed_mask,
                                         fixed_vals)[2]]) : sum(design[2])
        aic = -2 * loglik + 2 * pf
        bic = -2 * loglik + pf * log(n_bic)
    end

    return ERGMResult(
        model,
        θ_report,
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
        nothing,
        termination_report
    )
end

# The statistics of `model` at the boundary of their attainable range, as
# `(column, :min | :max)` over the model's columns (offsets excluded): the
# statistics whose observed value equals an end of their attainable range
# (`extreme`, R's `ergm.checkextreme.model`) and, iterated from those, the
# columns the pseudo-likelihood design shows at a boundary on the dyads an
# infinite offset leaves free (`boundary_columns`).
function _model_boundary(model::ERGMModel, X, n_tot, n_one,
                         extreme::AbstractVector{Tuple{Int,Symbol}})
    names = model.formula.terms.names
    mask, vals = _offset_info(model)
    any(mask) || return boundary_columns(X, n_tot, n_one; fixed=extreme)
    free, keep, _ = _offset_design(X, n_tot, n_one, names, mask, vals)
    ext = Tuple{Int,Symbol}[(findfirst(==(j), free)::Int, side) for (j, side) in extreme if !mask[j]]
    b = boundary_columns(X[keep, free], n_tot[keep], n_one[keep]; fixed=ext)
    return [(free[j], side) for (j, side) in b]
end

# R's sentences for the statistics `mcmle` fixes at ∓Inf (R's default
# `drop=TRUE`), one warning per side
function _warn_drop(names::Vector{String}, boundary::Vector{Tuple{Int,Symbol}};
                    context::AbstractString)
    for (side, word, at) in ((:min, "smallest", "-Inf"), (:max, "largest", "+Inf"))
        cols = [names[j] for (j, s) in boundary if s === side]
        isempty(cols) && continue
        @warn "$context: observed statistic(s) $(join(cols, ", ")) are at their " *
              "$word attainable values. Their coefficients will be fixed at $at " *
              "(no finite maximum-likelihood estimate exists; R ergm does the same " *
              "under its default drop=TRUE). The remaining coefficients are estimated " *
              "with these held fixed: the sampler never moves the statistic off its " *
              "observed bound. Pass drop=false to refuse such a model instead."
    end
    return nothing
end

# `drop=false`: a statistic at the boundary of its attainable range is
# refused instead of fixed at ∓Inf. R's `drop=FALSE` keeps the term and fits
# a model whose "MLE is poorly defined"; that is not implemented, so the
# strict mode names the statistics and the two ways out.
function _refuse_no_drop(model::ERGMModel, design, extreme; context::AbstractString,
                         why::AbstractString="drop=false asks to keep such a " *
                             "statistic in the model (R's `drop=FALSE`, whose \"MLE " *
                             "is poorly defined\"), which is not implemented",
                         remedy::AbstractString="Use the default drop=true — the " *
                             "coefficient fixed at ±Inf and the rest estimated, as R " *
                             "ergm does — or write the statistic as an offset at its " *
                             "limit (`Offset(term, -Inf)`, `Offset(term, Inf)`), or " *
                             "remove the term(s).")
    boundary = _model_boundary(model, design..., extreme)
    isempty(boundary) && return nothing
    names = model.formula.terms.names
    lo = [names[j] for (j, s) in boundary if s === :min]
    hi = [names[j] for (j, s) in boundary if s === :max]
    parts = String[]
    isempty(lo) || push!(parts, "observed statistic(s) $(join(lo, ", ")) are at their " *
                                "smallest attainable values (coefficient -Inf)")
    isempty(hi) || push!(parts, "observed statistic(s) $(join(hi, ", ")) are at their " *
                                "largest attainable values (coefficient +Inf)")
    throw(ArgumentError(
        "$context: " * join(parts, "; ") * ". No finite estimate exists, and " *
        "$why. $remedy"))
end

# Three significant digits for log messages (Inf/NaN print as themselves).
# `@sprintf("%.3g")`, not `round(x; sigdigits=3)`: the rounded Float64 is not
# exactly representable, so `string` printed `6.969999999999999e-32` and
# `1.6699999999999998e33` in diagnostics.
_fmt3(x::Real) = isfinite(x) ? @sprintf("%.3g", x) : string(x)

# The routine-level missing-data vocabulary (NetworkCore.missing_policies): the
# policies `mcmle`'s `missing=` keyword actually accepts — NOT the generic
# `:face`. Tooling (the capability matrix) prints this instead of a literal.
missing_policies(::typeof(mcmle)) = _MCMLE_MISSING_POLICIES

# MCMLE has a principled treatment of masked dyads — missing-data maximum
# likelihood behind `missing=:mle` (the default still refuses, as the trait's
# own example in NetworkCore.jl does: opting in is explicit, never implicit).
supports_missing(::typeof(mcmle)) = true

"""
    mcmle_covariance(samples, chain_lengths, obs_samples, obs_chain_lengths, p)
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

Part of [`ERGM.Extension`](@ref): a variant with its own MCMC MLE
(ERGMRank's `method=:mcmle`) gets the same Fisher + Monte-Carlo covariance as `mcmle`, so its standard errors
include the MCMC-error component rather than a locally re-derived Geyer
covariance.

# Example
```julia
using ERGM, Random
rng = Xoshiro(1)
samples = randn(rng, 2000, 2)             # statistics sampled at θ̂, one chain of 2000
V, Vtot, se, mcse = ERGM.Extension.mcmle_covariance(samples, [2000], nothing, nothing, 2)
isapprox(V, [1.0 0.0; 0.0 1.0]; atol=0.15)  # true: Σ̂⁻¹ of unit-variance draws
all(mcse .< se)                             # true: `se` includes the MC component
```
"""
function mcmle_covariance(samples::Matrix{Float64}, chain_lengths::Vector{Int},
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
Returns `nothing`. Internal: the variants' MCMC MLEs get the warning from
[`mcmle_solve`](@ref ERGM.Extension.mcmle_solve).

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

The classical MCMLE convergence tests, as one `public` (not exported) function: given an
`n × p` matrix of statistics sampled at the current
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

Under `termination=:hotelling` (the pre-0.2 rule) `converged` is `mcmle`'s
stopping verdict. Under the default `termination=:confidence` the verdict is
R ergm's equivalence test (`confidence_test`) and this function only
supplies the classical diagnostics recorded in `fit.mcmc_convergence`,
computed on the final sample — which, for a converged fit, is the sample
drawn at the last iterate, before the final Newton step (see
[`MCMLEConvergence`](@ref)). ERGMEgo's moment matching and any other
equation-solving estimator can use it instead of an ad-hoc relative-change
rule.

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

# ----------------------------------------------------------------------------
# R ergm's `confidence` termination (`MCMLE.termination = "confidence"`, its
# default since ergm 4): an equivalence test. The estimating equation
# ḡ(θ) − g_obs is declared solved when we are `confidence`-sure (99 %) that
# its true value lies inside a TOLERANCE REGION — the ellipsoid
# {x : x'(precision·Σ)⁻¹x ≤ 1}, Σ the per-draw covariance of the statistics
# (minus that of the constrained chain under missing-data ML), i.e. within
# √precision ≈ 0.32 standard deviations — given the Monte-Carlo error of the
# sample. Unlike a fixed max |t| < 0.1, this is attainable under MC noise
# (the sample is simply enlarged until the confidence region fits), and
# unlike a non-significant Hotelling test it cannot pass because the sample
# is too small to see a difference.
# ----------------------------------------------------------------------------

"""
    mcmle_solve(draw, θ0; labels, n_samples, maxiter=60, termination=:confidence,
                 conv_precision=0.1, conv_confidence=0.99, conv_threshold=0.1,
                 hotelling_alpha=0.05, gamma0=0.1, max_step_norm=5.0,
                 max_n_samples=16 * n_samples, resize=nothing, take_step=true,
                 adjust=nothing, target=nothing, verbose=false,
                 context="MCMLE") -> NamedTuple

The Monte-Carlo MLE **iteration**, as [`mcmle`](@ref) runs it, written
against a sampling callback so that it knows nothing about networks. Part
of [`ERGM.Extension`](@ref), for every estimator of the family that solves
`E_θ[g] = target` by MCMC (TERGM's CMLE over constrained sample spaces and pooled
transitions, ERGMEgo's moment matching, the rank and count MCMLEs).
`mcmle(::ERGMModel)` itself is this function plus a sampler.

`draw(θ, n)` samples at coefficients `θ` with size request `n` and returns
either an `n × p` matrix of statistics (one chain; the target is then the
`target` keyword, default zeros — statistics measured relative to the
observed ones), or a NamedTuple `(samples, chain_lengths, target,
obs_samples, obs_chain_lengths, aux)` — of which only `samples` is required
— for several chains, a per-draw target (the missing-data MCMLE's
constrained chain, given as `obs_samples`) or anything the caller wants
back (`aux`).

Every iteration draws, takes the Hummel-stepped Newton update
(`_hummel_step` (internal); skipped when `take_step == false`, the iterations
then only test), and — at full step length — applies the stopping rule:
`termination=:confidence` is R ergm 4's equivalence test on the updated
coefficients ([`confidence_test`](@ref ERGM.Extension.confidence_test)), `:hotelling` the t-ratio +
Hotelling rule ([`mcmc_convergence`](@ref)). When the confidence test fails
near the solution, or the estimating equations stop approaching the
tolerance region, the next request is `resize(n, boost)` (default:
`min(max_n_samples, ceil(n·boost))`). A singular sampled covariance stops
the iteration with a warning prefixed `context`. `adjust(θ, δ, iter) ->
(δ′, allowed, stop)`, when given, is a family's own step rule, applied to
the Hummel step before the stopping rule tests it: `δ′` is the step taken,
`allowed == false` forbids declaring convergence on it, and `stop == true`
ends the iteration, unconverged, after it (the curved MCMLE's trust region
on its decays).

Returns `(coef, converged, iterations, step_length, termination_p, final,
tests, vcov_fisher, vcov, se, mcmc_se)`: `final` is the draw the returned
coefficients were stepped from when converged (the one whose test passed),
otherwise a fresh draw at them; `tests` is `mcmc_convergence` on it; the
covariance is [`mcmle_covariance`](@ref ERGM.Extension.mcmle_covariance)'s (Fisher + Monte-Carlo).

# Example
```julia
using ERGM, Random, Statistics
# A toy exponential family: g ~ N(θ, I₂); solve E_θ[g] = (1, −2)
rng = Xoshiro(1)
draw(θ, n) = randn(rng, n, 2) .+ θ'
sol = ERGM.Extension.mcmle_solve(draw, [0.0, 0.0]; labels=["a", "b"], n_samples=2000,
                        target=[1.0, -2.0])
sol.converged                                   # true
isapprox(sol.coef, [1.0, -2.0]; atol=0.1)       # true
isapprox(sol.se, [1.0, 1.0]; atol=0.1)          # true: Σ = I, so the Fisher SEs are 1
```
"""
function mcmle_solve(draw::F, θ0::AbstractVector{<:Real};
                      labels::Vector{String}, n_samples::Int, maxiter::Int=60,
                      termination::Symbol=:confidence, conv_precision::Float64=0.1,
                      conv_confidence::Float64=0.99, conv_threshold::Float64=0.1,
                      hotelling_alpha::Float64=0.05, gamma0::Float64=0.1,
                      max_step_norm::Float64=5.0, max_n_samples::Int=16 * n_samples,
                      resize::R=nothing, take_step::Bool=true,
                      adjust::A=nothing,
                      target::Union{Nothing,AbstractVector{<:Real}}=nothing,
                      verbose::Bool=false,
                      context::AbstractString="MCMLE") where {F,R,A}
    termination in (:confidence, :hotelling) || throw(ArgumentError(
        "mcmle_solve: termination must be :confidence or :hotelling"))
    θ = Vector{Float64}(θ0)
    pf = length(θ)
    length(labels) == pf || throw(ArgumentError(
        "mcmle_solve: $(length(labels)) labels for $pf coefficients"))
    default_target = target === nothing ? zeros(pf) : Vector{Float64}(target)
    # Normalize what `draw` returns
    unpack(d::AbstractMatrix) = (samples=d, chain_lengths=[size(d, 1)],
                                 target=default_target, obs_samples=nothing,
                                 obs_chain_lengths=nothing, aux=nothing)
    unpack(d::NamedTuple) = (samples=d.samples,
        chain_lengths=get(d, :chain_lengths, [size(d.samples, 1)]),
        target=get(d, :target, default_target),
        obs_samples=get(d, :obs_samples, nothing),
        obs_chain_lengths=get(d, :obs_chain_lengths, nothing),
        aux=get(d, :aux, nothing))
    grow(n, boost) = resize === nothing ?
        (n >= max_n_samples ? n : min(max_n_samples, ceil(Int, n * boost))) :
        resize(n, boost)

    converged = false
    γ = gamma0
    iterations = 0
    n_cur = n_samples                    # per-iteration sample size (boosted)
    term_test = nothing                  # the last termination test
    final = nothing
    not_improved = falses(4)             # R's MCMLE.confidence.boost.lag window
    prev_diff = nothing

    for iter in 1:maxiter
        iterations = iter
        if verbose
            println("$context iteration $iter (step length γ = $(round(γ, digits=3)), " *
                    "$n_cur draws)...")
        end
        d = unpack(draw(θ, n_cur))
        samples, target_d = d.samples, d.target

        mean_stats = vec(mean(samples, dims=1))
        cov_stats = cov(samples)
        sd_stats = sqrt.(max.(diag(cov_stats), 0.0))
        _warn_degenerate_stats(sd_stats, labels)
        diff = target_d .- mean_stats

        step = _hummel_step(samples, target_d; step_length=γ,
                            max_step_norm=max_step_norm, cov_stats=cov_stats)
        if step.singular
            source = iter == 1 ? "the initial estimates" :
                                 "the iteration-$(iter - 1) $context update"
            @warn "The covariance matrix of the sampled statistics is singular at " *
                  "iteration $iter (collinear statistics, a degenerate model, or a " *
                  "collapsed sampler). $context cannot take further Newton steps; the " *
                  "returned coefficients are $source, unrefined, and standard errors " *
                  "will be NaN. Check the model for degeneracy or redundant terms."
            break
        end
        # The Hummel step length (γ = 1 once the sampled cloud covers the
        # target) and the Newton step toward the pseudo-target — under the
        # lognormal approximation of the MC likelihood ratio (R's default
        # metric) a full step is statnet's MCMLE update
        γ = step.step_length
        delta = take_step ? step.delta : zeros(pf)
        # A family's own step rule (the curved MCMLE's trust region on its
        # decays): it may shorten the step, forbid declaring convergence on
        # it, or stop the iteration unconverged after it
        allowed, stop = true, false
        if adjust !== nothing && take_step
            delta, allowed, stop = adjust(θ, delta, iter)
        end

        # Termination, tested only at full step length on THIS sample — the
        # sample the standard errors will come from if it passes. The
        # `:confidence` rule tests the UPDATED estimate θ + δ (importance-
        # reweighting this sample to it, as R does); `:hotelling` is the
        # pre-0.2 rule on the sample at θ.
        if termination === :confidence
            term_test = confidence_test(samples, d.chain_lengths, target_d, delta;
                                         obs_samples=d.obs_samples,
                                         obs_chain_lengths=d.obs_chain_lengths,
                                         precision=conv_precision,
                                         confidence=conv_confidence)
            passed = γ == 1.0 && term_test.converged && allowed
        else
            tests = mcmc_convergence(samples, target_d;
                                     conv_threshold=conv_threshold,
                                     hotelling_alpha=hotelling_alpha,
                                     chain_lengths=d.chain_lengths,
                                     target_samples=d.obs_samples,
                                     target_chain_lengths=d.obs_chain_lengths)
            term_test = (converged=tests.converged, p_value=tests.hotelling_p,
                         d2=NaN, boost=1.0)
            passed = γ == 1.0 && tests.converged && allowed
        end

        θ .+= delta
        stop && break
        if passed
            converged = true
            final = d
            if verbose
                println("Converged at iteration $iter ($termination rule, p = " *
                        "$(_fmt3(term_test.p_value)), $n_cur draws)")
            end
            break
        end

        # R's sample-size boost under the confidence rule: near the solution
        # (full step length) the MC noise is what keeps the test from
        # passing, so the next sample is larger — by the factor the test
        # asks for, at most 2 — and so is it when the estimating equations
        # stop moving toward the tolerance region (more than once in the
        # last four iterations)
        if termination === :confidence
            boost = γ == 1.0 && term_test.d2 < 2 ? term_test.boost : 1.0
            if prev_diff !== nothing
                d2_now = term_test.d2
                d2_prev = _tolerance_distance(prev_diff, samples, d.obs_samples,
                                              conv_precision)
                popfirst!(not_improved)
                push!(not_improved, d2_now >= d2_prev)
                if sum(not_improved) > 1
                    boost = max(boost, 2.0)
                    fill!(not_improved, false)
                end
            end
            prev_diff = diff
            boost > 1.0 && (n_cur = grow(n_cur, boost))
        end
    end

    # The final sample: the one the converged estimate was stepped from (its
    # tests are the ones that passed), or — when the loop ended unconverged —
    # a fresh sample at the returned coefficients, so the recorded
    # diagnostics describe the estimate that is returned
    if final === nothing
        final = unpack(draw(θ, n_cur))
        term_test = termination === :confidence ?
            confidence_test(final.samples, final.chain_lengths, final.target, zeros(pf);
                             obs_samples=final.obs_samples,
                             obs_chain_lengths=final.obs_chain_lengths,
                             precision=conv_precision, confidence=conv_confidence) :
            term_test
    end
    tests = mcmc_convergence(final.samples, final.target;
                             conv_threshold=conv_threshold,
                             hotelling_alpha=hotelling_alpha,
                             chain_lengths=final.chain_lengths,
                             target_samples=final.obs_samples,
                             target_chain_lengths=final.obs_chain_lengths)
    vcov_fisher, var_cov, std_errors, mc_se =
        mcmle_covariance(Matrix{Float64}(final.samples), final.chain_lengths,
                          final.obs_samples, final.obs_chain_lengths, pf)
    return (coef=θ, converged=converged, iterations=iterations, step_length=γ,
            termination_p=term_test === nothing ? NaN : Float64(term_test.p_value),
            final=final, tests=tests, vcov_fisher=vcov_fisher, vcov=var_cov,
            se=std_errors, mcmc_se=mc_se)
end

"""
    _hummel_step(samples, target; step_length=1.0, max_step_norm=5.0,
                 coverage=0.95, cov_stats=cov(samples))
        -> (delta, step_length, d2, singular)

One Hummel-stepped Monte-Carlo Newton update of an equation-solving
estimator (Hummel, Hunter & Handcock 2012) — the step `mcmle` takes at every
iteration. Internal: the variants' own MCMC estimators take the same step
through [`mcmle_solve`](@ref ERGM.Extension.mcmle_solve).

`samples` is the `n × p` matrix of statistics sampled at the current
coefficients and `target` the `p` statistics to match (the observed ones, or
the constrained chain's mean). With `Σ = cov(samples)` and `ḡ` the sampled
mean, the squared Mahalanobis distance `d2 = (target − ḡ)'Σ⁻¹(target − ḡ)`
decides the step length `γ`: 1 when the sampled cloud covers the target
(`d2` within the `coverage` quantile of χ²ₚ), otherwise the largest fraction
of the way to the target that stays inside the cloud, allowed to at most
double from the previous `step_length` and never below 0.01. The step is
`delta = γ·Σ⁻¹(target − ḡ)`, capped at Euclidean norm `max_step_norm`; add
it to the coefficients. `singular == true` (with a zero `delta`) when `Σ` is
not positive definite — a collapsed sampler or collinear statistics.

# Example
```julia
using ERGM, Random, Statistics
rng = Xoshiro(1)
draws = randn(rng, 2000, 2) .+ [1.0 5.0]          # statistics sampled at θ
near = ERGM._hummel_step(draws, [1.1, 5.0])
near.step_length                                   # 1.0: the cloud covers the target
isapprox(near.delta, [0.1, 0.0]; atol=0.05)        # true: Σ ≈ I, so δ ≈ target − mean
far = ERGM._hummel_step(draws, [9.0, 5.0]; step_length=0.1)
far.step_length                                    # 0.2: at most doubled per iteration
```
"""
function _hummel_step(samples::AbstractMatrix{<:Real}, target::AbstractVector{<:Real};
                      step_length::Real=1.0, max_step_norm::Real=5.0,
                      coverage::Real=0.95,
                      cov_stats::AbstractMatrix{<:Real}=cov(samples))
    p = size(samples, 2)
    length(target) == p || throw(ArgumentError(
        "_hummel_step: $(length(target)) targets for $p sampled statistics"))
    diff = Float64.(target) .- vec(mean(samples, dims=1))
    F = cholesky(Symmetric(Matrix{Float64}(cov_stats)); check=false)
    issuccess(F) || return (delta=zeros(p), step_length=Float64(step_length),
                            d2=Inf, singular=true)
    # Squared Mahalanobis distance of the target from the sampled cloud, and
    # the radius within which the cloud is considered to cover it
    d2 = max(dot(diff, F \ diff), 0.0)
    cut = quantile(Chisq(p), coverage)
    γ = d2 <= cut ? 1.0 : clamp(min(sqrt(cut / d2), 2.0 * step_length), 0.01, 1.0)
    delta = F \ (γ .* diff)
    step_norm = norm(delta)
    step_norm > max_step_norm && (delta .*= max_step_norm / step_norm)
    return (delta=delta, step_length=γ, d2=d2, singular=false)
end

# The squared distance of `x` on the tolerance-region scale,
# x'(precision·Σ)⁻¹x, Σ the per-draw covariance of the estimating function
function _tolerance_distance(x::AbstractVector, samples::AbstractMatrix,
                             obs_samples, precision::Float64)
    U = cov(samples)
    obs_samples === nothing || (U = U .- cov(obs_samples))
    U = precision .* (U .+ U') ./ 2
    F = cholesky(Symmetric(U); check=false)
    issuccess(F) || return Inf
    return max(dot(x, F \ x), 0.0)
end

# The squared Mahalanobis distance, under `W`, from the interior point `y` of
# the ellipsoid {x : x'U⁻¹x ≤ 1} to its boundary (R ergm's
# `.ellipsoid_mahalanobis`): with U = LLᵀ, W̃ = L⁻¹WL⁻ᵀ = QΛQᵀ and
# a = QᵀL⁻¹y, the nearest boundary point is bᵢ = aᵢ/(1 + μλᵢ) with
# μ ∈ (−1/λ_max, 0) solving Σ aᵢ²/(1 + μλᵢ)² = 1 (bisection), and the
# distance is Σ aᵢ²μ²λᵢ/(1 + μλᵢ)².
function _ellipsoid_mahalanobis(y::AbstractVector, W::AbstractMatrix, U::AbstractMatrix)
    FU = cholesky(Symmetric(Matrix(U)); check=false)
    issuccess(FU) || return 0.0
    Linv = inv(FU.L)
    ỹ = Linv * y
    dot(ỹ, ỹ) >= 1 && return 0.0                 # not an interior point
    W̃ = Symmetric(Linv * W * Linv')
    E = eigen(W̃)
    λ = max.(E.values, 0.0)
    a = E.vectors' * ỹ
    λmax = maximum(λ)
    λmax > 0 || return Inf                        # no Monte-Carlo error at all
    f(μ) = sum(a[k]^2 / (1 + μ * λ[k])^2 for k in eachindex(a))
    lo, hi = -1 / λmax, 0.0
    # f increases from ‖a‖² < 1 at μ = 0 toward ∞ as μ ↓ −1/λ_max when a has
    # a component along the top eigenvector: bisect for the crossing
    if f(lo * (1 - 1e-12)) > 1
        for _ in 1:200
            mid = (lo + hi) / 2
            f(mid) > 1 ? (lo = mid) : (hi = mid)
        end
        μ = (lo + hi) / 2
        return sum(a[k]^2 * μ^2 * λ[k] / (1 + μ * λ[k])^2 for k in eachindex(a))
    end
    # Otherwise (no, or a numerically negligible, component of a along the
    # top eigenvector — y at the centre, say) the nearest boundary point sits
    # at μ = −1/λ_max: its other components are aᵢ/(1 − λᵢ/λ_max) and the
    # top-eigenvector component c makes up the unit length
    top = [λ[k] >= λmax * (1 - 1e-12) for k in eachindex(λ)]
    dist = 0.0
    used = 0.0
    for k in eachindex(a)
        top[k] && continue
        b = a[k] / (1 - λ[k] / λmax)
        used += b^2
        λ[k] > 0 && (dist += (a[k] - b)^2 / λ[k])
    end
    atop = sqrt(sum(a[k]^2 for k in eachindex(a) if top[k]; init=0.0))
    c = sqrt(max(1 - used, 0.0))
    return dist + (c - atop)^2 / λmax
end

# Upper tail of the Hotelling T² distribution with `param` parameters and
# `df` degrees of freedom (R ergm's `.ptsq`), and its quantile (`.qtsq`)
_ptsq_upper(T2::Float64, param::Int, df::Float64) =
    ccdf(FDist(param, df - param + 1), T2 * (df - param + 1) / (param * df))
_qtsq(prob::Float64, param::Int, df::Float64) =
    quantile(FDist(param, df - param + 1), prob) / ((df - param + 1) / (param * df))

"""
    confidence_test(samples, chain_lengths, target, δ; obs_samples=nothing,
                     obs_chain_lengths=nothing, precision=0.1, confidence=0.99)
        -> (converged, p_value, d2, boost)

R ergm's `confidence` MCMLE termination test for the estimate `θ + δ`,
from statistics `samples` drawn at `θ` (importance-reweighted to `θ + δ`
with weights ∝ exp(δ'g), the Monte-Carlo covariance of the mean inflated by
the weights' Kish factor `n·Σw²`). `d2` is the gate distance of the
unweighted equation at `θ` on the tolerance scale (R tests only when
`d2 < 2`); `converged` requires the reweighted equation inside the tolerance
region and the Hotelling-T² equivalence p-value below `1 − confidence`;
`boost` is the factor by which R would enlarge the next sample when the test
fails near the solution (`min(critval/T², 2)`).

Part of [`ERGM.Extension`](@ref): an estimator that solves moment equations
by MCMC (ERGMEgo, a TERGM CMLE) stops on the same rule as `mcmle` — pass the step it is about
to take as `δ` (zeros to test the current coefficients), and multiply its
next sample size by `boost` when the test fails. With `obs_samples` (a
constrained chain whose mean is the target) the test is the two-sample one
of the missing-data MCMLE.

# Example
```julia
using ERGM, Random, Statistics
rng = Xoshiro(11)
draws = randn(rng, 4000, 2)                               # statistics sampled at θ
on = ERGM.Extension.confidence_test(draws, [4000], vec(mean(draws, dims=1)), zeros(2))
on.converged                                              # true: solved, with 99 % confidence
off = ERGM.Extension.confidence_test(draws, [4000], [0.5, 0.0], zeros(2))
off.converged                                             # false: 0.5 sd off target
few = ERGM.Extension.confidence_test(draws[1:40, :], [40], vec(mean(draws[1:40, :], dims=1)), zeros(2))
few.converged, few.boost > 1                              # (false, true): too few draws to be sure
```
"""
function confidence_test(samples::AbstractMatrix, chain_lengths::AbstractVector{<:Integer},
                          target::AbstractVector, δ::AbstractVector;
                          obs_samples=nothing, obs_chain_lengths=nothing,
                          precision::Float64=0.1, confidence::Float64=0.99)
    fail = (converged=false, p_value=1.0, d2=Inf, boost=2.0)
    n, p = size(samples)
    Σ = cov(samples)
    obs_samples === nothing || (Σ = Σ .- cov(obs_samples))
    Σ = (Σ .+ Σ') ./ 2
    # Statistics without sampled variance cannot be tested (R drops them,
    # and refuses when their difference is nonzero)
    active = [Σ[k, k] > 1e-12 * max(1.0, abs(target[k])) for k in 1:p]
    any(active) || return fail
    diff0 = target .- vec(mean(samples, dims=1))
    any(abs(diff0[k]) > 1e-9 for k in 1:p if !active[k]) && return fail
    idx = findall(active)
    pa = length(idx)
    U = precision .* Σ[idx, idx]
    FU = cholesky(Symmetric(U); check=false)
    issuccess(FU) || return fail
    d2 = max(dot(diff0[idx], FU \ diff0[idx]), 0.0)

    # Importance weights to θ + δ (log-weights centred for stability)
    lw = samples * δ
    w = exp.(lw .- maximum(lw)); w ./= sum(w)
    est = target .- vec(samples' * w)
    Σmc = _mc_cov_of_mean(samples, chain_lengths) .* (n * sum(abs2, w))
    n_eff = _effective_sample_size(samples, chain_lengths)
    if obs_samples !== nothing
        no = size(obs_samples, 1)
        lwo = obs_samples * δ
        wo = exp.(lwo .- maximum(lwo)); wo ./= sum(wo)
        est = vec(obs_samples' * wo) .- vec(samples' * w)
        ocl = something(obs_chain_lengths, [no])
        Σmc_obs = _mc_cov_of_mean(obs_samples, ocl) .* (no * sum(abs2, wo))
        n_eff_obs = _effective_sample_size(obs_samples, ocl)
        # Welch-type degrees of freedom of the two-sample Hotelling test
        # (R's `approx.hotelling.diff.test`)
        Vx = Σmc[idx, idx]; Vy = Σmc_obs[idx, idx]
        Vi = pinv(Vx .+ Vy)
        tr2(V) = (tr(V * Vi * V * Vi) + tr(V * Vi)^2)
        df = (pa + pa^2) / (tr2(Vx) / n_eff + tr2(Vy) / n_eff_obs)
        W = Vx .+ Vy
    else
        df = n_eff - 1
        W = Σmc[idx, idx]
    end
    e = est[idx]
    d2e = max(dot(e, FU \ e), 0.0)
    (isfinite(df) && df > pa) || return (converged=false, p_value=1.0, d2=d2, boost=2.0)
    d2e < 1 || return (converged=false, p_value=1.0, d2=d2, boost=1.0)
    W = (W .+ W') ./ 2
    T2 = _ellipsoid_mahalanobis(e, W, U)
    pval = _ptsq_upper(T2, pa, df)
    converged = d2 < 2 && pval < 1 - confidence
    critval = _qtsq(confidence, pa, df)
    boost = T2 > 0 ? clamp(critval / T2, 1.0, 2.0) : 2.0
    return (converged=converged, p_value=pval, d2=d2, boost=boost)
end

# One sentence naming the termination rule's verdict, for the warning,
# `show` and `approximations`
# The verdict of an unconverged fit, in the terms of the stopping rule that
# decided it: R's confidence test reports its own p-value and the step
# length; the legacy `:hotelling` rule adds the max t-ratio it also tests.
# The classical t-ratios / Hotelling p of `fit.mcmc_convergence` are not the
# verdict under `:confidence` (and describe the pre-step sample of a
# converged fit), so they are not quoted as if they were.
function _termination_verdict(t, t_ratios, γ, iterations=nothing)
    detail = _termination_detail(t)
    t !== nothing && t.rule === :hotelling &&
        (detail *= ", max t-ratio $(_fmt3(maximum(t_ratios)))")
    detail *= ", step length γ $(_fmt3(γ))"
    iterations === nothing ||
        (detail *= " after $iterations iteration$(iterations == 1 ? "" : "s")")
    return detail
end

function _termination_detail(t)
    t === nothing && return "no termination test"
    if t.rule === :confidence
        return "$(round(Int, 100 * t.confidence))% equivalence test p " *
               "$(_fmt3(t.p_value)) (needs < $(_fmt3(1 - t.confidence)); tolerance " *
               "precision $(_fmt3(t.precision)), $(t.n_samples) draws)"
    else
        return "Hotelling p $(_fmt3(t.p_value)) ($(t.n_samples) draws)"
    end
end

# The state of the ESS-adaptive sampler: one network per chain, carried
# from one MCMLE iteration to the next (R's `MCMLE.sequential`), the current
# thinning interval, and whether the chains have been burned in
mutable struct _AdaptiveChains{N}
    nets::Vector{N}
    interval::Int
    burned::Bool
end

"""
    ess_sample(extend, target, m; interval, n_chains=1, maxruns=16)
        -> (samples, chain_lengths, interval)

The ESS-adaptive sample-size / interval chooser (R ergm's adaptive MCMC,
`MCMC.effectiveSize`), written against a chain callback. Part of
[`ERGM.Extension`](@ref), so a variant's sampler adapts the way
`mcmle(...; effective_size=)` does.

`extend(c, n, interval)` continues chain `c` (1 … `n_chains`) by `n` draws
`interval` steps apart and returns their `n × p` statistics. Each chain
first contributes `m / n_chains` draws at `interval`; while the effective
sample size of the pooled draws (the smallest over statistics of the Geyer
ESS, summed over chains) is below `target`, every chain drops every other
draw, the interval doubles, and each chain draws the missing half at the
new interval — so every round doubles the chain length at a constant stored
size — at most `maxruns` times. Chains are extended on separate tasks.
Returns the pooled `m × p` sample, the per-chain row counts and the final
interval (start the next call at half of it, as R does).

# Example
```julia
using ERGM, Random
# A strongly autocorrelated AR(1) "chain": thinning is what buys ESS
rng = Xoshiro(1); state = Ref(0.0)
function extend(c, n, interval)
    out = zeros(n, 1)
    for k in 1:n
        for _ in 1:interval; state[] = 0.95 * state[] + randn(rng); end
        out[k, 1] = state[]
    end
    return out
end
S, counts, interval = ERGM.Extension.ess_sample(extend, 150.0, 256; interval=1)
size(S), counts            # ((256, 1), [256])
interval >= 8              # true: the interval was doubled until ESS ≥ 150
```
"""
function ess_sample(extend::F, target::Real, m::Int; interval::Int,
                     n_chains::Int=1, maxruns::Int=16) where {F}
    (m >= 2 && n_chains >= 1 && interval >= 1) || throw(ArgumentError(
        "ess_sample: need m ≥ 2, n_chains ≥ 1 and interval ≥ 1"))
    nc = n_chains
    counts = fill(m ÷ nc, nc)
    for c in 1:(m % nc)
        counts[c] += 1
    end
    parts = Vector{Matrix{Float64}}(undef, nc)
    iv = interval
    spawn_all(nc) do c
        parts[c] = extend(c, counts[c], iv)
    end
    for run in 1:maxruns
        samples = reduce(vcat, parts)
        (run == maxruns || _effective_sample_size(samples, counts) >= target) &&
            return samples, counts, iv
        iv *= 2
        iv2 = iv
        spawn_all(nc) do c
            kept = parts[c][2:2:end, :]
            parts[c] = vcat(kept, extend(c, counts[c] - size(kept, 1), iv2))
        end
    end
end

"""
    _mcmc_sample_ess!(chains, model, θ, target, m, burnin; rng, toggleable, proposal,
                      maxruns=16) -> (samples, chain_lengths)

Sample statistics at `θ` until their effective sample size reaches `target`
([`ess_sample`](@ref ERGM.Extension.ess_sample) on network chains). The chains in `chains` continue
from their current state: burned in for `burnin` toggles the first time and
`16 × interval` afterwards (the coefficients move little between MCMLE
iterations), their interval halved from the last call's. One seed per chain
is drawn from `rng` (a single chain uses `rng` itself), so the result is
thread-count independent.
"""
function _mcmc_sample_ess!(chains::_AdaptiveChains, model::ERGMModel, θ::Vector{Float64},
                           target::Float64, m::Int, burnin::Int;
                           rng::AbstractRNG, toggleable::Symbol=:free,
                           proposal::Symbol=:tnt, maxruns::Int=16)
    nc = length(chains.nets)
    terms = model.formula.terms
    rngs = nc == 1 ? AbstractRNG[rng] :
                     AbstractRNG[Random.Xoshiro(sd) for sd in rand(rng, UInt64, nc)]
    b = chains.burned ? 16 * chains.interval : burnin
    chains.burned && (chains.interval = max(1, chains.interval ÷ 2))
    spawn_all(nc) do c
        _mh_run!(rngs[c], chains.nets[c], terms, θ, 0, b, 1, false, toggleable;
                 proposal=proposal)
    end
    chains.burned = true
    samples, counts, iv = ess_sample(target, m; interval=chains.interval, n_chains=nc,
                                      maxruns=maxruns) do c, n, interval
        _mh_run!(rngs[c], chains.nets[c], terms, θ, n, 0, interval, false, toggleable;
                 proposal=proposal)[1]
    end
    chains.interval = iv
    return samples, counts
end

"""
    mcmle_sampler(model; effective_size=64, n_samples=1000,
                  max_n_samples=16 * n_samples, n_free=length(model.formula.terms),
                  burnin=nothing, interval=nothing, n_chains=1, proposal=:spdyad,
                  toggleable=:free, missing=:error, rng=Random.default_rng(),
                  verbose=false) -> (draw, resize, n_samples)

The sampler of [`mcmle`](@ref), packaged as the `draw`/`resize` pair that
[`mcmle_solve`](@ref ERGM.Extension.mcmle_solve) takes. Part of
[`ERGM.Extension`](@ref): an estimator that solves moment equations by
MCMC on an `ERGMModel` (ERGMEgo's moment matching, the curved MCMLE) samples
exactly as `mcmle` does, R ergm 4's design, by calling it.

- **ESS-adaptive** (`effective_size=64`, R's `MCMLE.effectiveSize`; the
  default): the chains are continued from one call to the next (R's
  `MCMLE.sequential`): burned in for `burnin` toggles on the first call and
  `16 × interval` afterwards, their thinning interval starting at
  `interval ÷ 8` and halved from the last call's, then doubled until the
  effective sample size of the draws reaches the target
  ([`ess_sample`](@ref ERGM.Extension.ess_sample)). `resize(n, boost)` — the
  stopping rule's sample-size boost — multiplies the TARGET (capped at
  `max_n_samples / 4`) and returns the number of draws to store,
  `min(max_n_samples, max(256, 32·n_free, 4·target))`.
- **Fixed-size** (`effective_size=nothing`): every call draws `n` statistics
  `interval` toggles apart after `burnin` toggles from `model.network`, split
  over `n_chains` chains as `mcmle` splits them; `resize` multiplies `n` by
  the boost, up to `max_n_samples`.

`draw(θ, n; model=model)` returns the NamedTuple `(samples, chain_lengths)`
— a valid `mcmle_solve` draw, with the target given by its `target=` — at the full
coefficient vector `θ` (an infinite entry holds its statistic at its bound).
`model=` lets a family change the statistics between calls (the curved
MCMLE's working statistics at the current decays); the chains always run on
networks over `model.network`'s vertex set. `n_samples` is the first
request to pass to `mcmle_solve`. One seed per chain is drawn from `rng` at
each call (a single chain uses `rng` itself), so the result does not depend
on the thread count.

# Example
```julia
using ERGM, Random
net = load_dataset(:florentine_marriage)
model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), net)
s = ERGM.Extension.mcmle_sampler(model; rng=Xoshiro(1))
S, chain_lengths = s.draw([-1.7, 0.4], s.n_samples)
size(S), chain_lengths                             # ((256, 2), [256]): one chain
sol = ERGM.Extension.mcmle_solve(s.draw, [-1.7, 0.4]; labels=["edges", "gwesp"],
        n_samples=s.n_samples, resize=s.resize, target=compute_all(model.formula.terms, net))
sol.converged                                      # true: mcmle's fit, by hand
s.resize(256, 2.0) >= 256                          # true: a boost raises the target ESS
```
"""
function mcmle_sampler(model::ERGMModel;
                       effective_size::Union{Nothing,Real}=64,
                       n_samples::Int=1000,
                       max_n_samples::Int=16 * n_samples,
                       n_free::Int=length(model.formula.terms),
                       burnin::Union{Nothing,Int}=nothing,
                       interval::Union{Nothing,Int}=nothing,
                       n_chains::Int=1,
                       proposal::Symbol=:spdyad,
                       toggleable::Symbol=:free,
                       missing::Symbol=:error,
                       rng::AbstractRNG=Random.default_rng(),
                       verbose::Bool=false)
    n_samples >= 2 || throw(ArgumentError(
        "mcmle_sampler: n_samples must be ≥ 2 (got $n_samples)"))
    max_n_samples >= n_samples || throw(ArgumentError(
        "mcmle_sampler: max_n_samples ($max_n_samples) must be ≥ n_samples ($n_samples)"))
    n_chains >= 1 || throw(ArgumentError(
        "mcmle_sampler: n_chains must be ≥ 1 (got $n_chains)"))
    _check_proposal(proposal; context="mcmle_sampler")
    _check_toggleable(model.network, toggleable, missing; context="mcmle_sampler")
    adaptive = effective_size !== nothing
    adaptive && !(effective_size >= 8) && throw(ArgumentError(
        "mcmle_sampler: effective_size must be ≥ 8 (R's MCMLE.effectiveSize is 64); " *
        "got $effective_size"))
    if burnin === nothing || interval === nothing
        burnin, interval = _resolve_mcmc_controls(model, burnin, interval;
                                                  toggleable=toggleable)
    end
    n_max = max_n_samples
    ess_target = adaptive ? Float64(effective_size) : NaN
    ess_stored(t) = min(n_max, max(256, 32 * n_free, ceil(Int, 4 * t)))
    chains = adaptive ? _AdaptiveChains(
        [_copy_network(model.network) for _ in 1:clamp(n_chains, 1, 64)],
        max(1, interval ÷ 8), false) : nothing
    m0 = model
    function draw(θ, n; model=m0)
        samples, chain_lengths = adaptive ?
            _mcmc_sample_ess!(chains, model, θ, ess_target, n, burnin; rng=rng,
                              toggleable=toggleable, proposal=proposal) :
            _mcmc_sample(model, θ, n, burnin, interval; rng=rng, missing=missing,
                         n_chains=n_chains, toggleable=toggleable, proposal=proposal)
        return (samples=samples, chain_lengths=chain_lengths)
    end
    # The stopping rule's boost: a larger draw count, or — adaptive — a larger
    # target effective sample size
    function resize(n, boost)
        if adaptive
            ess_target = min(ess_target * boost, n_max / 4)
            verbose && println("  increasing the target effective sample size to " *
                               "$(round(Int, ess_target))")
            return ess_stored(ess_target)
        end
        n >= n_max && return n
        verbose && println("  increasing the MCMC sample size to " *
                           "$(min(n_max, ceil(Int, n * boost)))")
        return min(n_max, ceil(Int, n * boost))
    end
    return (draw=draw, resize=resize,
            n_samples=adaptive ? ess_stored(ess_target) : n_samples)
end

"""
    _mcmc_sample(model, θ, n_samples, burnin, interval;
                 rng=Random.default_rng(), missing=:error, n_chains=1,
                 toggleable=:free, proposal=:tnt)
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
toggle — the free and constrained chains of the missing-data MCMLE;
`proposal` the MH proposal (`:tnt` or `:random`).
"""
function _mcmc_sample(model::ERGMModel, θ::Vector{Float64},
                      n_samples::Int, burnin::Int, interval::Int;
                      rng::AbstractRNG=Random.default_rng(),
                      missing::Symbol=:error, n_chains::Int=1,
                      toggleable::Symbol=:free, proposal::Symbol=:tnt)
    n_chains = clamp(n_chains, 1, n_samples)
    if n_chains == 1
        stats = mh_sample(model, θ; n_samples=n_samples, burnin=burnin,
                          interval=interval, rng=rng, missing=missing,
                          toggleable=toggleable, proposal=proposal).stats
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
    spawn_all(n_chains) do c
        chain_rng = Random.Xoshiro(seeds[c])
        chain_stats[c] = mh_sample(model, θ; n_samples=counts[c],
                                   burnin=burnin, interval=interval,
                                   rng=chain_rng, missing=missing_policy,
                                   toggleable=toggleable, proposal=proposal).stats
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
missing-dyad mask). Internal.

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

# An attribute-preserving copy of `net` with every tie removed (the mask is
# kept): the network on which the statistics g(∅) are evaluated
function _empty_copy(net::Network)
    e = copy(net)
    for edge in collect(edges(e))
        rem_edge!(e, src(edge), dst(edge))
    end
    return e
end

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

Every sum includes `θ'g(∅)`, the statistics of the EMPTY network: a
dyad-independent model has `g(y) = g(∅) + Σ_{ij ∈ y} δ(i,j)`, so
`Z(θ) = exp(θ'g(∅)) · Π(1 + exp(θ'δ(i,j)))`. Every built-in term has
`g(∅) = 0` on its dyad-independent coordinates, but a user term need not
(a non-edge count, a covariate sum over absent ties): before 0.2 the term
was dropped and the log-likelihood was off by exactly `θ'g(∅)`.
"""
function _dyad_independent_logZ(model::ERGMModel, θ::Vector{Float64};
                                toggleable::Symbol=:free)
    toggleable in _TOGGLEABLE || throw(ArgumentError(
        "_dyad_independent_logZ: toggleable must be :free, :all or :masked"))
    net = model.network
    terms = model.formula.terms
    # θ'g(∅): the empty network's statistics (zero for every built-in
    # dyad-independent term, not necessarily for a user's)
    logZ = _finite_dot(θ, compute_all(terms, _empty_copy(net)))

    # The unmasked dyads: summed over when free, fixed at their observed
    # value when only the masked dyads are free
    if toggleable === :masked
        for e in edges(net)
            i, j = Int(src(e)), Int(dst(e))
            is_missing_dyad(net, i, j) && continue
            logZ += _finite_dot(θ, change_stat_all(terms, net, i, j))
        end
    else
        X, n_tot, _ = _mple_data(net, terms, is_directed(model))
        @inbounds for r in eachindex(n_tot)
            logZ += n_tot[r] * _dyad_logZ(θ, view(X, r, :))
        end
    end

    # The masked dyads: fixed at face value (a present edge contributes
    # θ'δ(i,j) to every configuration of the free dyads) or free
    for (i, j) in missing_dyads(net)
        δ = change_stat_all(terms, net, i, j)
        if toggleable === :free
            has_edge(net, i, j) && (logZ += _finite_dot(θ, δ))
        else
            logZ += _dyad_logZ(θ, δ)
        end
    end
    return logZ
end

"""
    _bridge_logZ(model, θ; nrungs=16, n_samples=500, burnin, interval, rng,
                 missing=:error, toggleable=:free, proposal=:tnt) -> Float64

Path-sampling (bridge) estimate of the log-normalizer `log Z(θ)` over the
`toggleable` dyad set (see [`_dyad_independent_logZ`](@ref) for the three
sets), the quantity behind MCMLE's log-likelihood.

Let `θ₀` be `θ` with every dyad-dependent coordinate set to zero (per
[`is_dyad_dependent`](@ref)). `θ₀` defines a dyad-independent reference
whose normalizer `log Z(θ₀)` is computed exactly. Along the linear path
`θ(u) = θ₀ + u·(θ − θ₀)`, the thermodynamic identity

    d/du log Z(θ(u)) = E_{θ(u)}[g(Y)]' (θ − θ₀)

is integrated by **composite Simpson's rule** over `nrungs` equal segments
(`nrungs + 1` grid points; an odd `nrungs` is raised by one so the rule
applies), with `E_{θ(u)}[g]` estimated by an MCMC run of `n_samples` draws
at each grid point — a chain over the same `toggleable` dyad set. Simpson's
error is O(h⁴): the trapezoid rule used before 0.2 had an O(h²) bias of
−0.3 to −0.5 log-likelihood units at 16 rungs on a 205-node gwesp model,
larger than R ergm's own bridge error, which Simpson removes at the same
cost. Rungs are embarrassingly parallel and run on separate threads, each
with an RNG seeded deterministically from `rng` (results are
thread-count-independent). This is the standard ergm-style bridge
estimator; the one-jump reverse importance-sampling estimate it replaces
has unusably high variance when `θ` is far from the reference.

For fully dyad-independent models `θ₀ = θ` and the exact normalizer is
returned with no Monte Carlo error.

`missing` is the caller's already-validated missing-dyad policy for a
`toggleable=:free` chain (with `:condition_on_face` every rung and the
exact reference are the *conditional* ones given the masked dyads' face
values); the `:all`/`:masked` chains take none.

Internal: a variant's bridge is
[`bridge_integrate`](@ref ERGM.Extension.bridge_integrate) with its own
sampler and its own exact reference.

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
                      missing::Symbol=:error, toggleable::Symbol=:free,
                      proposal::Symbol=:tnt)
    nrungs >= 1 || throw(ArgumentError("nrungs must be at least 1 (mcmle skips " *
                                       "the bridge entirely for bridge_rungs=0)"))
    terms = model.formula.terms

    # Dyad-independent reference: zero out the dyad-dependent coordinates
    θ0 = copy(θ)
    for (k, t) in enumerate(terms)
        is_dyad_dependent(t) && (θ0[k] = 0.0)
    end
    logZ0 = _dyad_independent_logZ(model, θ0; toggleable=toggleable)

    # (a -Inf offset of a dyad-independent statistic is in the reference and
    # does not move: its Δ is 0, not -Inf − (-Inf))
    Δ = [θ[k] == θ0[k] ? 0.0 : θ[k] - θ0[k] for k in eachindex(θ)]
    all(iszero, Δ) && return logZ0          # dyad-independent: exact

    # Simpson ladder over u ∈ [0, 1] (`bridge_integrate`); per-rung
    # E_{θ(u)}[g] estimated by MCMC, one deterministic seed per rung
    m = iseven(nrungs) ? nrungs : nrungs + 1
    seeds = rand(rng, UInt64, m + 1)
    missing_policy = missing
    return logZ0 + bridge_integrate(θ0, θ; rungs=nrungs, threaded=true) do θu, k
        stats = mh_sample(model, θu; n_samples=n_samples, burnin=burnin,
                          interval=interval, rng=Random.Xoshiro(seeds[k]),
                          missing=missing_policy,
                          toggleable=toggleable, proposal=proposal).stats
        vec(mean(stats, dims=1))
    end
end

"""
    bridge_integrate(mean_stats, θ_ref, θ; rungs=16, threaded=false) -> Float64

The path-sampling (bridge) integral `log Z(θ) − log Z(θ_ref)` of an
exponential family, by the thermodynamic identity
`d/du log Z(θ(u)) = E_{θ(u)}[g]'(θ − θ_ref)` along `θ(u) = θ_ref + u·(θ −
θ_ref)`, integrated by composite Simpson's rule (`_bridge_quadrature` (internal))
over `rungs` equal segments (an odd `rungs` is raised by one).
`mean_stats(θu, k)` returns the (Monte-Carlo) mean of the statistics at the
`k`-th grid point `θu`, `k = 1, …, rungs + 1` — `k` lets the caller seed each
rung deterministically. With `threaded=true` the rungs run on separate tasks
(the callback must then be thread-safe; an error in a rung is rethrown as
itself). Coordinates equal in `θ` and `θ_ref` — an infinite offset
included — do not move.

Part of [`ERGM.Extension`](@ref): [`mcmle`](@ref)'s log-likelihood is `θ'g_obs − (log Z(θ_ref) +
this)` with the exact dyad-independent `log Z(θ_ref)`; a variant supplies
its own sampler and its own exact reference (TERGM: the conditional
logistic likelihood of a side's dyad-independent terms).

# Example
```julia
using ERGM
# One Bernoulli dyad with statistic y ∈ {0, 1}: E_θ[y] = logistic(θ), and
# log Z(θ) = log(1 + e^θ), so the integral from 0 to 2 is known
exact = log1p(exp(2.0)) - log(2.0)
est = ERGM.Extension.bridge_integrate((θu, k) -> [1 / (1 + exp(-θu[1]))], [0.0], [2.0]; rungs=8)
isapprox(est, exact; atol=1e-5)        # true
```
"""
function bridge_integrate(mean_stats::F, θ_ref::AbstractVector{<:Real},
                           θ::AbstractVector{<:Real}; rungs::Int=16,
                           threaded::Bool=false) where {F}
    length(θ) == length(θ_ref) || throw(ArgumentError(
        "bridge_integrate: θ has length $(length(θ)), θ_ref $(length(θ_ref))"))
    rungs >= 1 || throw(ArgumentError("bridge_integrate: rungs must be ≥ 1"))
    Δ = [θ[k] == θ_ref[k] ? 0.0 : θ[k] - θ_ref[k] for k in eachindex(θ)]
    all(iszero, Δ) && return 0.0
    m = iseven(rungs) ? rungs : rungs + 1
    us = range(0.0, 1.0; length=m + 1)
    contrib = Vector{Float64}(undef, length(us))
    rung(k) = (contrib[k] = dot(Δ, mean_stats(θ_ref .+ us[k] .* Δ, k)); nothing)
    threaded ? spawn_all(rung, length(us)) : foreach(rung, eachindex(us))
    return _bridge_quadrature(contrib)
end

"""
    _bridge_quadrature(values) -> Float64

Composite Simpson's rule for `∫₀¹ f(u) du` from `f` on an even number `m` of
equal segments (`values` holds the `m + 1` grid values): O(h⁴) error, where
the trapezoid rule's O(h²) error biased the bridge. Internal: the quadrature
of every bridge in the family, reached through
[`bridge_integrate`](@ref ERGM.Extension.bridge_integrate).

# Example
```julia
using ERGM
ERGM._bridge_quadrature([u^3 for u in 0:0.25:1]) ≈ 0.25     # true: exact for cubics
```
"""
function _bridge_quadrature(contrib::AbstractVector{<:Real})
    m = length(contrib) - 1
    (m >= 2 && iseven(m)) || throw(ArgumentError(
        "_bridge_quadrature: Simpson's rule needs an even number of segments (got $m)"))
    acc = contrib[1] + contrib[end]
    for k in 2:m
        acc += (iseven(k) ? 4.0 : 2.0) * contrib[k]
    end
    return acc / (3 * m)
end

"""
    _bridge_loglik(model, θ, obs_stats; nrungs=16, n_samples=500,
                   burnin, interval, rng, missing=:error, proposal=:tnt) -> Float64

Path-sampling (bridge) estimate of the log-likelihood
`θ'g(y_obs) − log Z(θ)` — the estimator behind MCMLE's AIC/BIC for a fully
observed network (or the `:condition_on_face` model, whose normalizer is
the conditional one): `dot(θ, obs_stats)` minus `_bridge_logZ` (internal) over
the free dyads. The missing-data MLE's log-likelihood is instead the
difference of two `_bridge_logZ` calls (`:masked` minus `:all`); see
[`mcmle`](@ref).
"""
function _bridge_loglik(model::ERGMModel, θ::Vector{Float64},
                        obs_stats::Vector{Float64};
                        nrungs::Int=16, n_samples::Int=500,
                        burnin::Int=10000, interval::Int=100,
                        rng::AbstractRNG=Random.default_rng(),
                        missing::Symbol=:error, proposal::Symbol=:tnt)
    logZ = _bridge_logZ(model, θ; nrungs=nrungs, n_samples=n_samples,
                        burnin=burnin, interval=interval, rng=rng,
                        missing=missing, toggleable=:free, proposal=proposal)
    return _finite_dot(θ, obs_stats) - logZ
end
