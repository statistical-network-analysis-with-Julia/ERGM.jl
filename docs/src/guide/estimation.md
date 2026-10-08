# Model Estimation

ERGM.jl provides two estimation methods: Maximum Pseudo-Likelihood Estimation (MPLE) for fast approximation and Monte Carlo Maximum Likelihood Estimation (MCMLE) for accurate inference. Both methods return an `ERGMResult` object containing coefficients, standard errors, and fit statistics.

## The default: R's rule

`ergm(net, terms)` (= `fit_ergm`) takes `method=:auto`, which chooses the
estimator as R's `ergm()` does: the MPLE when no term is dyad-dependent —
the pseudo-likelihood is then the likelihood, so the MPLE is the exact
MLE — and the MCMLE as soon as one term is dyad-dependent (`Triangle`,
`GWESP`, `Mutual`, `Degree`, …; see [`is_dyad_dependent`](@ref)). The rule
is the `public` [`ERGM.resolve_method`](@ref), which the ERGM variants
share. `method=:mple` and `method=:mcmle` choose explicitly; the blocks
below name the method they illustrate. A keyword of one estimator passed to
the other — `se=:bootstrap` on a formula `:auto` fits by MCMLE — is an
`ArgumentError` that names the estimator taking it.

```julia
using ERGM
flo = load_dataset(:florentine_marriage)
ergm(flo, [Edges(), NodeCov(:wealth)]).method    # :mple — dyad-independent
ERGM.resolve_method(:auto, ERGMModel(ERGMFormula([Edges(), Triangle()]), flo))   # :mcmle
```

## Overview

The estimation process differs by method:

**MPLE**:
1. Build a (compressed) design matrix of change statistics for all observed dyads
2. Fit the logistic regression by Newton–Raphson on the ecosystem's shared kernel (`NetworkCore.logistic_derivatives` + `NetworkCore.newton_fit`)
3. Compute standard errors from the observed information matrix at the solution

**MCMLE**:
1. Initialize with MPLE estimates
2. Iterate: sample networks via MCMC, update parameters via Newton-Raphson
3. Converge when observed statistics match the expected statistics under the model

## Maximum Pseudo-Likelihood Estimation (MPLE)

MPLE treats each potential edge as an independent observation and fits a logistic regression model using change statistics as features:

$$\text{logit}\left(P(Y_{ij} = 1 \mid Y_{-ij})\right) = \theta^\top \delta(y)_{ij}$$

```julia
using NetworkCore, ERGM
using Random

Random.seed!(42)

# Example network: Florentine marriage ties with a categorical attribute
net = load_dataset(:florentine_marriage)
set_vertex_attribute!(net, :gender,
    Dict(v => (isodd(v) ? "F" : "M") for v in 1:nv(net)))
terms = [Edges(), GWESP(0.5), NodeMatch(:gender)]

result = ergm(net, terms; method=:mple)
```

Or equivalently:

```julia
formula = ERGMFormula(terms)
model = ERGMModel(formula, net)
result = mple(model)
```

### How MPLE Works

For a network with $n$ nodes:

| Network Type | Number of Dyads |
|-------------|-----------------|
| Directed | $n(n-1)$ |
| Undirected | $n(n-1)/2$ |

For each dyad $(i,j)$:
- **Response**: $y_{ij} = 1$ if edge exists, $0$ otherwise
- **Features**: Change statistics $\delta(y)_{ij}$ for each model term

Dyads with identical change-statistic rows are collapsed into one binomial
row (`n_tot` dyads, `n_one` of them ties), and the logistic regression is
solved by Newton–Raphson with step halving on NetworkCore.jl's shared
`logistic_derivatives`/`newton_fit` pair, to a tolerance of `1e-8` on the
pseudo-log-likelihood (`maxiter=100`; an unconverged fit warns and records
`converged = false`). On the Florentine marriage network the dyad-independent
fit agrees with R `ergm` to `1e-12` on the coefficients and `5e-8` on the
standard errors; the provenanced fixture asserts `1e-6`.

### MPLE Strengths and Limitations

| Aspect | Detail |
|--------|--------|
| Speed | Very fast — single optimization |
| Consistency | Consistent for dyadic independence models |
| Bias | Can be biased for models with strong dependencies |
| Standard errors | Anticonservative (too small) for dependent models |
| Use case | Exploration, initial estimates, independent-dyad models |

### Honest MPLE Uncertainty

For **dyad-independent** models (only `Edges` plus nodal/dyadic covariate
terms) the pseudo-likelihood *is* the likelihood, and the default
inverse-Hessian standard errors are correct.

For models with **dyad-dependent** terms (`Triangle`, `GWESP`, `Mutual`,
...) the pseudo-likelihood treats dependent dyads as independent
observations, so the inverse-Hessian standard errors are anticonservative:
in a simulation study (n = 30, `edges + gwesp(0.5)`, 200 networks) their
95 % Wald intervals covered the truth only 0.85 and 0.71 of the time,
against 0.945 and 0.970 for the parametric bootstrap. ERGM.jl therefore
**withholds the inference built on them by default**: the fit reports the
point estimates and the naive standard errors, but `z_values` and
`p_values` are `NaN` (the table shows `NaN` in those columns, with a note
saying why), `confint` refuses with an `ArgumentError`, and
`approximations(result)` records the suppression. Passing `se=:hessian`
explicitly is the written opt-in to R's naive Wald table, printed with the
pseudo-likelihood caveat. Two remedies give calibrated inference:

```julia
# Parametric-bootstrap standard errors: simulate n_boot networks at the
# MPLE, refit the MPLE on each, use the empirical covariance
result_boot = ergm(net, terms; method=:mple, se=:bootstrap, n_boot=100,
                   rng=Xoshiro(1))
result_boot.se_type   # :bootstrap

# Or refit with full MCMC maximum likelihood
result_mcmle = ergm(net, terms; method=:mcmle, rng=Xoshiro(2))
```

The bootstrap call above prints, for this model and seed:

```
┌ Warning: mple: se=:bootstrap — 13 of the 100 bootstrap refits had no finite MPLE (the simulated network put a statistic at the boundary of its attainable range — e.g. no triangle, no shared partner — or perfectly separated the ties) and were excluded; the standard errors are the empirical covariance of the 87 finite refits. The standard errors are conditional on a finite refit: the excluded replicates are the extreme ones, so the standard errors are biased downward. This is about the simulated replicates, not about the observed network. …
ERGM Results
============
Method: mple (maximum pseudo-likelihood: an approximation under dyadic dependence; the default method=:auto fits the MCMLE here, as R does)
Log-likelihood: -53.8523
AIC: 113.7, BIC: 122.07
Converged: true

Coefficients:
                  Estimate  Std.Error  z value  Pr(>|z|)
edges              -1.7661     0.5225  -3.3802    0.0007 ***
gwesp.fixed.0.5     0.0973     0.2901   0.3355    0.7373
nodematch.gender    0.1061     0.5138   0.2065    0.8364
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Note: 13 of the 100 parametric-bootstrap refits had no finite MPLE (a statistic at the boundary of its attainable range, or a separated design, in the simulated network) and were excluded from the standard errors, which are the empirical covariance of the remaining 87 refits (fit.boot_replicates). The standard errors are conditional on a finite refit: the excluded replicates are the extreme ones, so the standard errors are biased downward.

Note: this model contains dyad-dependent terms and was fit by maximum
pseudolikelihood (MPLE). Standard errors are parametric-bootstrap
estimates; the MPLE point estimates may still be biased. Consider
refitting with method=:mcmle.
```

A simulated 16-node network with no shared-partner edge has no finite
MPLE for `gwesp.fixed.0.5` (its coefficient is `-Inf` under R's drop
semantics), and such a replicate would make every standard error `NaN`
if it entered the covariance. Replicates without a finite MPLE are
therefore **excluded**, said once (the warning is about the simulated
replicates, not about the data), recorded in `approximations(result_boot)`
and kept as `NaN` rows of `result_boot.boot_replicates`; at least two
finite refits are required. A point estimate that itself carries a `±Inf`
coefficient cannot be simulated from, so `se=:bootstrap` refuses it with
an `ArgumentError`.

Note the bootstrap fixes the *standard errors*, not the MPLE point
estimates, which can themselves be biased under strong dependence —
`method=:mcmle` addresses both. Which terms count as dyad-dependent is
queryable with [`is_dyad_dependent`](@ref).

## Monte Carlo Maximum Likelihood Estimation (MCMLE)

MCMLE uses MCMC sampling to approximate the normalizing constant ratio, providing more accurate estimates for models with dependence:

```julia
result = ergm(net, terms;
    method = :mcmle,
    n_samples = 1000,
    rng = Xoshiro(42),
    verbose = true
)
result.converged
```

`burnin` and `interval` default to the dyad-scaled rule every sampler in
ERGM.jl shares (`20 × n_dyads` and `max(100, n_dyads ÷ 10)`); pass them
explicitly to override. Two further controls change the cost, not the
estimate:

```julia
# Split each iteration's 1000 draws over 4 independent chains (run in
# parallel when Julia has threads; the result is identical at any thread
# count because the chains are seeded from `rng`)
result4 = ergm(net, terms; method=:mcmle, n_samples=1000, n_chains=4, rng=Xoshiro(42))

# Skip the path-sampling log-likelihood (~70 % of a single-threaded fit's
# time on the Florentine and Faux Mesa High fits): coefficients and standard errors are
# bit-identical, `loglikelihood`/`aic`/`bic` are NaN and `show` says so
fast = ergm(net, terms; method=:mcmle, n_samples=1000, rng=Xoshiro(42), bridge_rungs=0)
coef(fast) == coef(ergm(net, terms; method=:mcmle, n_samples=1000, rng=Xoshiro(42)))
isnan(loglikelihood(fast))
```

### How MCMLE Works

The algorithm iterates:

1. **Initialize**: Start with MPLE estimates $\theta^{(0)}$
2. **Sample**: Generate networks from the current model $P_{\theta^{(t)}}$ via Metropolis-Hastings MCMC (tie/no-tie proposal, below)
3. **Update — at every iteration, the first included** (statnet's order):
   a partial Newton-Raphson step toward the Hummel pseudo-target,
   $\theta^{(t+1)} = \theta^{(t)} + \gamma\,\Sigma^{-1}(g(\text{obs}) - \bar{g}(\text{sim}))$,
   with step length $\gamma$ adapting toward 1 as the sampled statistic
   cloud covers the observed statistics. Under R's default lognormal
   approximation of the Monte-Carlo likelihood ratio, the full step *is*
   the maximizer of the approximated log-likelihood.
4. **Test termination** (at full step length, on the sample just drawn):
   R ergm 4's `confidence` rule (`termination=:confidence`, the default).
   The estimating equation $\bar g(\theta^{(t+1)}) - g_{\text{obs}}$ —
   evaluated at the *updated* estimate by importance-reweighting the
   sample — must lie, with 99 % Monte-Carlo confidence (`conv_confidence`),
   inside a tolerance region $\{x : x^\top(0.1\,\Sigma)^{-1}x \le 1\}$
   (`conv_precision = 0.1`, about a third of a standard deviation per
   statistic): a Hotelling-$T^2$ equivalence test. When it fails close to
   the solution the next iteration's sample is enlarged (by up to 2×, to at
   most `max_n_samples`), as R does — so the rule is attainable under
   Monte-Carlo noise, which the pre-0.2 rule (every t-ratio below 0.1 and a
   non-significant Hotelling test, now `termination=:hotelling`) was not.
5. **Repeat** until the test passes or `maxiter` (60, R's default) is
   reached. A converged fit's estimate is the update from the sample that
   passed, and that sample is the one its standard errors, `mcmc_samples`
   and `result.mcmc_convergence` come from. That sample was drawn at the
   last iterate, before the final Newton step, so its classical diagnostics
   (`result.mcmc_convergence`: t-ratios, Hotelling p) describe the
   pre-step sample, not the returned coefficients, and are not the verdict
   — `result.termination` is. An unconverged fit is returned with
   `converged == false`, a warning quoting the stopping rule's p-value and
   the step length, an entry in
   `NetworkCore.approximations(result)`, a caveat line under `Converged: false`
   in `show`, and the numbers themselves in `result.termination` and
   `result.mcmc_convergence` (see [Non-convergence](@ref) below)

A dyad-independent formula started from its MPLE starts at the exact MLE
(the pseudo-likelihood is the likelihood), so no Monte-Carlo step is taken
there: the iterations only confirm the moment equation.

The classical per-sample diagnostics (t-ratios, Hotelling test, effective
sample size) are one `public` function, `ERGM.mcmc_convergence(samples,
targets; conv_threshold, hotelling_alpha, chain_lengths)`, used by the
variants that solve their own moment equations (ERGMEgo, ERGMRank).

The reported log-likelihood (and AIC/BIC) is estimated by **path
sampling**: a `bridge_rungs`-segment ladder from a dyad-independent
reference distribution (whose normalizer is exact) to $\hat\theta$, with
the expected statistics at each rung estimated by MCMC and integrated by
**Simpson's rule** (O(h⁴); the trapezoid rule used before 0.2 was biased by
−0.3 to −0.5 log-likelihood units at 16 rungs on a 205-node gwesp model) —
the standard ergm-style bridge estimator. The reference normaliser is exact
and includes $\theta^\top g(\varnothing)$, the statistics of the empty
network (non-zero for some user terms). For fully dyad-independent models
the exact log-likelihood is returned.

### MCMC Sampling

Each MCMC step:
1. Propose (statnet's **tie/no-tie**, `proposal=:tnt`): with probability ½ a
   uniformly chosen existing tie, otherwise a uniformly chosen dyad
2. Compute: change statistics $\delta(y)_{ij}$ for all terms
3. Accept/reject: toggle the dyad with probability
   $\min(1, \exp(\pm\theta^\top \delta) \cdot H)$, $H$ the proposal's Hastings ratio
   (statnet's `MH_TNT` correction, including the empty-network boundary)

On sparse networks a uniformly random dyad is nearly always an absent tie
the model rejects, so TNT mixes several times faster per toggle than the
uniform random toggle (`proposal=:random`, the sampler of ERGM.jl 0.1).
Both are exact (their stationary distribution is verified against
enumeration in the test suite).

R ergm ≥ 4.6 mixes TNT with a **shared-partner-focused proposal**
(`SPDyad`, what its default `MCMC.prop = ~sparse + .triadic` selects for
every one-mode network), and so does `mcmle` by default: a quarter of the proposals
draw the dyad uniformly from those with at least one shared partner — where
a toggle changes the triadic statistics. That is `proposal=:spdyad`
(likewise exact; verified against enumeration and by an exact
detailed-balance check). On faux.mesa.high's `gwesp` model it gives about
6× TNT's effective sample size per draw at about twice the cost per toggle.

**ESS-adaptive sampling** (`effective_size=64`, R's `MCMLE.effectiveSize`,
the default): the chain is continued from one iteration to the next and each
iteration's sample is extended — its thinning interval doubling — until
the effective sample size of the sampled statistics reaches the target;
the stopping rule's boost then raises the target. Together with SPDyad this
is R ergm 4's default design. Fixed-size tie/no-tie sampling — `n_samples`
draws `interval` toggles apart per iteration — is one keyword pair away:

```julia
result_tnt = ergm(net, terms; method=:mcmle, proposal=:tnt, effective_size=nothing,
                  rng=Xoshiro(3))
result_tnt.converged
```

The sampler uses burn-in to reach stationarity and thinning to reduce autocorrelation. The loop itself is the exported [`mh_toggle!`](@ref) kernel (see the [simulation guide](@ref "The MCMC Algorithm")).

### Standard errors: Fisher information plus Monte-Carlo error

`vcov(result)` for an MCMLE fit is the inverse Fisher information estimated
from the final MCMC sample, $V = \hat\Sigma^{-1}$, **plus** the
Monte-Carlo component of the estimate (Hunter & Handcock 2006, §3.3): the
estimating equation $\bar g(\theta) = g_{\text{obs}}$ is itself a Monte-Carlo
average with $\operatorname{Var}(\bar g) = \Sigma_{mc}$ (the Geyer
initial-sequence covariance of the sampled statistics' mean — per-statistic
Geyer variances on the diagonal, lag-0 correlations off it, summed over
chains), so

$$\operatorname{vcov} = V + V\,\Sigma_{mc}\,V .$$

`result.vcov_fisher` is $V$ alone, `mcmc_se(result)` is
$\sqrt{\operatorname{diag}(V\Sigma_{mc}V)}$, and `show` prints R's
`summary.ergm` "MCMC %" column, which R defines as the share of the
*standard error* (not of its variance) that the Monte-Carlo term adds:

$$\text{MCMC \%} = \operatorname{round}\!\Big(100\,\frac{\text{se} - \text{se}_{\text{Fisher}}}{\text{se}}\Big),
\qquad \text{se}_{\text{Fisher}} = \sqrt{\operatorname{diag}(V)}$$

(`ergm:::summary.ergm`: `100 * (tot.se - mod.se) / tot.se`). On the
Florentine `edges + gwesp` model at `n_samples=4096` it rounds to 0 (as R
reports); at `n_samples=100` it is 1–3 %. The variance share
$100\,\text{mcmc\_se}^2/\text{se}^2$ is a different, larger number — 3–5 %
at `n_samples=100` — and is not what R prints. A large MCMC % says the
sample, not the data, limits the precision: raise `n_samples` or
`n_chains`.

```julia
using LinearAlgebra
result_mcmle = ergm(net, terms; method=:mcmle, n_samples=1000, rng=Xoshiro(1))
mcmc_se(result_mcmle)                                     # the MC part of each SE
se_fisher = sqrt.(diag(result_mcmle.vcov_fisher))
round.(Int, 100 .* (stderror(result_mcmle) .- se_fisher) ./ stderror(result_mcmle))   # R's "MCMC %"
```

### MCMLE Parameters

| Parameter | Description | Default | Guidance |
|-----------|-------------|---------|----------|
| `n_samples` | MCMC samples per iteration (over all chains) | 1000 | More = better approximation, smaller MCMC % |
| `n_chains` | Independent chains per iteration | 1 | Parallel when Julia has threads; identical result at any thread count |
| `burnin` | Steps before sampling (per chain) | `20 × n_dyads` | Increase if poor mixing |
| `interval` | Steps between samples | `max(100, n_dyads ÷ 10)` | Increase to reduce autocorrelation |
| `maxiter` | Maximum NR iterations | 60 | R's `MCMLE.maxit` |
| `termination` | Stopping rule | `:confidence` | R's equivalence test; `:hotelling` is the pre-0.2 t-ratio + Hotelling rule |
| `conv_precision`, `conv_confidence` | Tolerance region (share of the statistics' variance) and confidence of the equivalence test | 0.1, 0.99 | R's `MCMLE.MCMC.precision`, `MCMLE.confidence` |
| `max_n_samples` | Cap on the boosted per-iteration sample | `16 × n_samples` | — |
| `proposal` | MH proposal | `:spdyad` | R's SPDyad (its default); `:tnt` = tie/no-tie; `:random` = uniform random toggle |
| `effective_size` | Target effective sample size of ESS-adaptive sampling | 64 | R's `MCMLE.effectiveSize`; `nothing` = fixed `n_samples` per iteration |
| `missing` | Masked-dyad policy | `:error` | `:mle` (missing-data ML) or `:condition_on_face` (see below) |
| `obs_burnin`, `obs_interval` | Constrained-chain controls under `missing=:mle` | dyad-scaled over the masked dyads | R's `obs.MCMC.burnin` / `obs.MCMC.interval` |
| `init` | Starting coefficients | the MPLE | `coef(fit)` to continue an unconverged fit |
| `conv_threshold`, `hotelling_alpha` | The `termination=:hotelling` rule's t-ratio bound and test level | 0.1, 0.05 | Legacy rule only |
| `rng` | RNG all draws flow from | `Random.default_rng()` | Pass `Xoshiro(seed)` for reproducibility |
| `bridge_rungs` | Path-sampling segments for the log-likelihood (Simpson's rule; odd values are raised by one) | 16 | `0` skips it (loglik/AIC/BIC `NaN`) |
| `bridge_samples` | MCMC samples per bridge rung | `n_samples` | — |

(There is no `tol` keyword: convergence is a statistical test. Development
versions accepted `tol` and `max_iter`; see [Renamed and removed
names](@ref renames).)

### Curved models

A formula with `GWESP(decay; fixed=false)` or `GWDegree(decay; fixed=false)`
is a *curved* exponential family: the decay α is estimated. `mcmle` then
runs the curved MCMLE (Hunter & Handcock 2006). With `g_α` the fixed-decay
statistic and `D_α = ∂g_α/∂α`, the score in (θ, α) is
`(g_α − E g_α, θ·(D_α − E D_α))` and the Fisher information is the covariance
of the working statistics `(g_α, θ·D_α)`; each iteration samples the
fixed-decay model at the current α, recording `D_α` alongside, and takes the
same Hummel step and applies the same stopping rule as the linear MCMLE, on
the working statistics. The start is the fixed-decay MPLE at the term's
`decay`; a decay moves by at most 0.5 per iteration and stays ≥ 0. Standard
errors come from the inverse covariance of the working statistics (plus the
Monte-Carlo term), and the log-likelihood from the bridge of the fixed-decay
model at the fitted decay. On faux.mesa.high the fit agrees with R ergm's
curved fit within R's seed spread (provenanced fixture). `mple` refuses a
curved formula.

### MCMLE Strengths and Limitations

| Aspect | Detail |
|--------|--------|
| Accuracy | Consistent and asymptotically efficient |
| Standard errors | Correctly accounts for dependencies |
| Speed | Slower — requires MCMC at each iteration |
| Initialization | Benefits from good MPLE starting values |
| Use case | Final results, dependent models |

## Missing (Unobserved) Dyads

NetworkCore.jl can mark dyads whose tie status is **unobserved** (statnet-style
NA ties) with `set_missing_dyad!(net, i, j)` — distinct from "no tie". The
estimation routines treat the mask as follows:

- **MPLE excludes masked dyads.** An unobserved tie status is not a
  response, so masked dyads contribute no row to the logistic-regression
  design; `nobs(result)` shrinks by the number of masked dyads. The masked
  dyads' face values (edge present/absent as stored) still enter the change
  statistics of the observed dyads, which are computed conditional on the
  rest of the network. This is the available-case pseudo-likelihood, a
  principled treatment: `NetworkCore.supports_missing(mple) == true`, and
  `mple` takes no `missing` keyword.
- **MCMLE refuses a masked network by default** (`missing=:error` throws
  the shared `ArgumentError`, whose last bullets name the two policies it
  accepts; `missing_policies(mcmle) == (:error, :condition_on_face, :mle)`;
  the generic `:face` is *not* accepted) **and offers two treatments:**

  - `missing=:mle` — **missing-data maximum likelihood**, what R `ergm()`
    does with NA ties (Handcock & Gile 2010). The likelihood of the observed
    dyads is `Z_obs(θ)/Z(θ)`, the masked dyads integrated out. Each MCMLE
    iteration runs two chains: the *free* chain toggles every dyad (masked
    ones included) and estimates `E[g(Y)]`; the *constrained* chain toggles
    only the masked dyads, the observed dyads held at their observed
    values, and estimates `E[g(Y) | Y_obs]`, which replaces the observed
    statistics as the Newton target. The Fisher information is
    `Var[g] − Var[g | Y_obs]` (standard errors are `NaN`, with a warning,
    if that difference is not positive definite), both chains contribute
    Monte-Carlo error to the standard errors, and the log-likelihood is
    `log Z_obs(θ) − log Z(θ)` from two path-sampling bridges. The
    constrained chain's burn-in/thinning are `obs_burnin`/`obs_interval`
    (defaults: the dyad-scaled rule applied to the number of masked
    dyads). `nobs` is the number of observed dyads; the fit records
    `missing_method = :mle`. Use it whenever the masked dyads are genuinely
    unobserved. Under dyad independence it coincides with the
    available-case MPLE.
  - `missing=:condition_on_face` — hold each masked dyad fixed at its
    stored face value: never toggled, and counted in the observed
    statistics as recorded. That is the model *conditional on the face
    values* — a different estimand, right only when the stored values are
    true by construction (a structural zero, a tie fixed by design). It is
    warned, and the fit records `missing_method = :condition_on_face`.

  On the provenanced masked-flomarriage fixture (four NA dyads, two of them
  ties) the two estimands differ by 0.11 on both coefficients of
  `edges + gwesp(0.5)` — twenty times R's seed-to-seed spread — which is why
  neither is the default.

```julia
net_na = copy(net)
set_missing_dyad!(net_na, 3, 4)          # dyad 3–4 was not measured
fit_na = ergm(net_na, terms; method=:mple) # excluded from the pseudo-likelihood
nobs(fit_na)                             # one fewer observation
fit_na.missing_method                    # :available_case

# Missing-data maximum likelihood: the masked dyad is integrated out
fit_mle = ergm(net_na, [Edges(), GWESP(0.5)]; method=:mcmle,
               missing=:mle, n_samples=500, rng=Xoshiro(1))
fit_mle.missing_method                   # :mle
nobs(fit_mle)                            # one fewer observed dyad

# Conditioning on the face value must be asked for in writing, and is warned
fit_cf = ergm(net_na, [Edges(), NodeMatch(:gender)]; method=:mcmle,
              missing=:condition_on_face, n_samples=200, rng=Xoshiro(1))
fit_cf.missing_method                    # :condition_on_face
```

Simulation and GOF (`simulate_ergm`, `sample_networks`, `mh_sample`, `gof`)
accept only `:condition_on_face`: a simulated network has a value at every
dyad, so "simulation under missing data" is not defined, and `missing=:mle`
is refused with a message saying so. `mh_sample(...; toggleable=:all)` and
`toggleable=:masked` expose the two MCMLE chains directly.

## Comparing Methods

```julia
# Quick exploration
result_mple = ergm(net, terms; method=:mple)

# Final analysis (seeded: every Monte-Carlo draw flows from `rng`)
result_mcmle = ergm(net, terms; method=:mcmle, rng=Xoshiro(42), verbose=true)

# Compare coefficients
println("MPLE:  ", round.(coef(result_mple), digits=3))
println("MCMLE: ", round.(coef(result_mcmle), digits=3))
```

For models with weak dependencies (e.g., only `Edges` and `NodeMatch`), MPLE and MCMLE typically agree closely. For models with strong dependencies (e.g., `Triangle`, `GWESP`), MCMLE is preferred.

## Understanding Results

The `ERGMResult` object contains:

| Field | Type | Description |
|-------|------|-------------|
| `coefficients` | `Vector{Float64}` | Estimated coefficients |
| `std_errors` | `Vector{Float64}` | Standard errors |
| `z_values` | `Vector{Float64}` | Z-statistics (coef/SE) |
| `p_values` | `Vector{Float64}` | Two-sided p-values |
| `loglik` | `Float64` | (Pseudo-)log-likelihood |
| `aic` | `Float64` | Akaike Information Criterion (`dof` = the finite coefficients, as R's `logLik` df) |
| `bic` | `Float64` | Bayesian Information Criterion (sample size = the dyads the finite coefficients were estimated on — all of them, or after a drop the dyads the dropped term does not touch, R's `logLik` nobs) |
| `method` | `Symbol` | `:mple` or `:mcmle` |
| `converged` | `Bool` | Convergence status |
| `mcmc_samples` | `Matrix{Float64}` or `nothing` | MCMC samples (MCMLE only) |
| `se_type` | `Symbol` | `:hessian`, `:bootstrap`, or `:mcmc` — how the SEs were obtained |
| `missing_method` | `Symbol` | `:none`, `:available_case` (MPLE dropped masked dyads), `:mle` (integrated out by the constrained chain) or `:condition_on_face` |
| `vcov_fisher` | `Matrix{Float64}` | MCMLE: the inverse Fisher information alone (`vcov` adds the MC term); MPLE: `vcov` |
| `mcmc_se` | `Vector{Float64}` | Monte-Carlo component of each SE (`mcmc_se(result)`); zeros for MPLE |
| `mcmc_convergence` | `MCMLEConvergence` or `nothing` | MCMLE: `(iterations, step_length, t_ratios, hotelling_p, n_eff)` on the final sample |

### Accessor Functions

The full ecosystem StatsAPI surface is defined on `ERGMResult`
(`NetworkCore.check_statsapi(result; strict=true)` passes):

```julia
coef(result)          # Coefficient vector
stderror(result)      # Standard errors vector
vcov(result)          # Variance-covariance matrix
confint(result)       # Normal-theory limits, one row per coefficient (level=0.95)
coeftable(result)     # NetworkCore.CoefficientTable — exactly the table `show` prints
loglikelihood(result); aic(result); bic(result); nobs(result); dof(result)
mcmc_se(result)       # Monte-Carlo part of the SEs (zeros for an MPLE fit)
```

### Displaying Results

```julia
println(result)
```

`show` renders `coeftable(result)` — the shared ecosystem table with an
`Estimate`, `Std.Error`, `z value` and `Pr(>|z|)` column, p-values floored
at `<1e-16` and small ones printed in scientific notation — followed by the
caveats that apply. For the guide's running MPLE fit of
`[Edges(), GWESP(0.5), NodeMatch(:gender)]` on the Florentine data:

```text
ERGM Results
============
Method: mple (maximum pseudo-likelihood: an approximation under dyadic dependence; the default method=:auto fits the MCMLE here, as R does)
Log-likelihood: -53.8523
AIC: 113.7, BIC: 122.07
Converged: true

Coefficients:
                  Estimate  Std.Error  z value  Pr(>|z|)
edges              -1.7661     0.3750      NaN       NaN
gwesp.fixed.0.5     0.0973     0.1697      NaN       NaN
nodematch.gender    0.1061     0.5006      NaN       NaN
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Note: z values and p-values are not reported (NaN). This model contains
dyad-dependent terms and was fit by maximum pseudolikelihood (MPLE); the
standard errors shown are the naive pseudolikelihood ones, which treat
dependent dyads as independent and under-cover (95% Wald intervals
covered 0.71-0.85 in simulation), so no test or interval is built on
them. For inference refit with se=:bootstrap (parametric bootstrap) or
method=:mcmle; se=:hessian requests the naive Wald table explicitly.
```

With `se=:hessian` passed explicitly the same table carries the naive z
values and p-values, followed by the warning that they "should not be
trusted".

An MCMLE fit of the same model prints the stopping rule's verdict under
`Converged:` and R's "MCMC %" line after the table instead of the
pseudo-likelihood note:

```julia
println(ergm(net, terms; method=:mcmle, n_samples=1000, rng=Xoshiro(1)))
```

```text
ERGM Results
============
Method: mcmle (Monte-Carlo maximum likelihood)
Log-likelihood: -53.9448
AIC: 113.89, BIC: 122.25
Converged: true
  Termination: 99% equivalence test p 0.000172 (needs < 0.01; tolerance precision 0.1, 471 draws)

Coefficients:
                  Estimate  Std.Error  z value  Pr(>|z|)
edges              -1.8068     0.4492  -4.0223   5.8e-05 ***
gwesp.fixed.0.5     0.0988     0.2587   0.3818    0.7026
nodematch.gender    0.1922     0.4957   0.3877    0.6982
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

MCMC % of the standard error (100·(se − se_fisher)/se): edges 0, gwesp.fixed.0.5 0, nodematch.gender 0
```

A fit that exhausts `maxiter` prints `Converged: false` with the
non-convergence caveat directly under it (the same sentence the warning
carries and `NetworkCore.approximations(result)` lists) — see
[Convergence Issues](@ref) below for the text and what to do.

## Interpreting Coefficients

### Log-Odds Ratios

Coefficients are log-odds ratios for the conditional probability of an edge:

$$\text{logit}(P(Y_{ij} = 1 \mid Y_{-ij})) = \theta^\top \delta(y)_{ij}$$

### Example Interpretations

| Term | Coefficient | exp(θ) | Interpretation |
|------|-------------|--------|----------------|
| Edges | -2.3 | 0.10 | Low baseline density |
| Triangle | 0.9 | 2.46 | Each shared partner increases odds by 146% |
| NodeMatch | 0.5 | 1.65 | Same-attribute ties are 65% more likely |
| NodeCov | 0.1 | 1.11 | One-unit increase in attribute raises odds by 11% |
| GWESP(0.5) | 1.2 | 3.32 | Strong geometrically weighted triadic closure |

### Confidence Intervals

```julia
ci = confint(result)            # 95% normal-theory limits: column 1 lower, column 2 upper
ci90 = confint(result; level=0.9)

# Odds ratio confidence intervals
or_ci = exp.(ci)
```

The limits are built from the standard errors the fit reports (here the
MCMLE fit's). For an MPLE fit (`method=:mple`) of a dyad-dependent model, whose
naive pseudo-likelihood SEs under-cover, `confint` refuses with an
`ArgumentError` — use `se=:bootstrap` or `method=:mcmle`, or pass
`se=:hessian` explicitly to accept the naive intervals.

## Model Comparison

### AIC and BIC

```julia
# Fit multiple models
terms1 = [Edges()]
terms2 = [Edges(), Triangle()]
terms3 = [Edges(), GWESP(0.5), NodeMatch(:gender)]

# With the default method=:auto the dyad-dependent models are MCMLE fits,
# whose path-sampled log-likelihoods are comparable with the exact one of
# the dyad-independent model (pseudo-log-likelihoods are not)
r1 = ergm(net, terms1)
r2 = ergm(net, terms2; rng=Xoshiro(3))
r3 = ergm(net, terms3; rng=Xoshiro(4))

println("Model 1 — AIC: $(r1.aic), BIC: $(r1.bic)")
println("Model 2 — AIC: $(r2.aic), BIC: $(r2.bic)")
println("Model 3 — AIC: $(r3.aic), BIC: $(r3.bic)")

# Lower AIC/BIC = better fit (with complexity penalty)
```

### Log-Likelihood Comparison

```julia
println("Model 1 LL: ", r1.loglik)
println("Model 2 LL: ", r2.loglik)
println("Model 3 LL: ", r3.loglik)

# Higher log-likelihood = better fit
```

## Convergence Issues

### Non-convergence

An MCMLE fit that exhausts `maxiter` is never quiet. It is returned with
`converged == false` and

- a warning at fit time: `MCMLE did not converge in maxiter=60 iterations
  (99% equivalence test p 0.278 (needs < 0.01; tolerance precision 0.1,
  16000 draws), step length γ 1):
  the estimates are the last iterate and the standard errors are unreliable;
  increase maxiter/n_samples/burnin, check the model for degeneracy
  (mcmc_diagnostics), or refit from these coefficients
  (mcmle(model; init=coef(fit)))`;
- the same sentence under `Converged: false` when the result is shown;
- an entry in `NetworkCore.approximations(result)` (so
  `fit_metadata(result)` sees it);
- the numbers in `result.termination` (the stopping rule, its p-value and
  the final sample size: the verdict) and `result.mcmc_convergence` (the
  classical t-ratios, Hotelling p-value and effective sample size of a
  fresh sample at the returned coefficients, plus the iteration count and
  the last step length).

```julia
if !result.converged
    t = result.termination                   # the verdict: the stopping rule
    println(t.rule, " p ", t.p_value, " on ", t.n_samples, " draws")
    # continue from where it stopped, with a bigger budget
    result = mcmle(ERGMModel(ERGMFormula(terms), net);
                   init=coef(result), n_samples=4000, rng=Xoshiro(2))
end
```

### Common Causes and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Model degeneracy | Very large coefficients, non-convergence | Use GWESP/GWDegree instead of Triangle/Kstar |
| Near-degeneracy | Slow convergence, unstable estimates | Simplify model, use geometrically weighted terms |
| A statistic at the boundary of its attainable range (a `NodeMatch` with no within-group tie; a `NodeCov` whose positive-change dyads are all empty and negative-change dyads all tied, …) | `mple` and `mcmle` warn "observed statistic(s) … are at their smallest attainable values. Their coefficients will be fixed at -Inf", return `-Inf` (SE 0) for that term and R's estimates, `dof`, AIC and BIC of the rest (R's default `drop=TRUE`); `drop=false` refuses instead | Read the `-Inf` as "never", or remove the term |
| A statistic whose change statistics are all zero (a `NodeMix` cell of a singleton level, `Triangle` on a network with no two-path) | at its attainable bound: fixed at `∓Inf` as above (R's `ergm.checkextreme.model`); otherwise not identifiable: reported as `NaN` with R's "not varying" warning (R: `NA`) | Remove the term |
| Perfect separation by a *combination* of statistics (no single one at its boundary, but together they predict every tie) | `mple` decides it exactly (the shared `NetworkCore.logistic_separation` linear programme, as R's `mple.existence`), warns "the MPLE does not exist (separation)" naming the separating terms (R: "The MPLE does not exist!"), returns the last Newton iterate with `converged == false`, the terms in `fit.separated_terms`, NaN z values, p-values and intervals, `is_exact == false` and the caveat in `show`/`approximations`; `mcmle` refuses to start from it | Remove or coarsen a term, or pass `init=` to `mcmle` |
| Multicollinearity | Large standard errors | Remove correlated terms |

### Handling Degeneracy

Degeneracy is the most common issue with ERGMs. It occurs when the model places nearly all probability on either the empty or the complete network.

```julia
# Degenerate specification (avoid!)
terms_bad = [Edges(), Triangle()]

# Better specification
terms_good = [Edges(), GWESP(0.5)]

# Even better — add degree control
terms_best = [Edges(), GWESP(0.5), GWDegree(0.5)]
```

## Advanced Topics

### Computing Summary Statistics

```julia
# Compute observed statistics without fitting
stats = summary_stats(net, terms)
println(stats)
```

### Two-Stage Fitting

For more control over the estimation process:

```julia
# Stage 1: Build the model
formula = ERGMFormula(terms)
model = ERGMModel(formula, net)

# Stage 2: Fit with your chosen method
result = mple(model; verbose=true)
# or
result = mcmle(model; n_samples=2000, verbose=true)
```

## Best Practices

1. **Start with MPLE**: Use MPLE for initial exploration, switch to MCMLE for final results
2. **Check convergence**: Always verify `result.converged == true`
3. **Avoid degeneracy**: Prefer geometrically weighted terms for larger networks
4. **Compare AIC/BIC**: Use information criteria for model selection
5. **Validate with GOF**: Always run goodness-of-fit diagnostics after estimation
6. **Use verbose mode**: Monitor MCMLE progress to diagnose convergence issues
7. **Sufficient network size**: ERGMs require networks with at least ~20 nodes for reliable estimation
8. **Set random seeds**: Pass an explicit `rng` (e.g. `rng=Xoshiro(42)`) to `mcmle`/`ergm` — all Monte Carlo draws flow from it, so runs with the same seed are exactly reproducible
9. **Honest uncertainty**: For dyad-dependent MPLE fits, use `se=:bootstrap` or refit with `method=:mcmle` — an MPLE fit (`method=:mple`) of such a model with the default `se` reports no p-values for this reason
