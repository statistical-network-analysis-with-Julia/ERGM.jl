# Model Estimation

ERGM.jl provides two estimation methods: Maximum Pseudo-Likelihood Estimation (MPLE) for fast approximation and Monte Carlo Maximum Likelihood Estimation (MCMLE) for accurate inference. Both methods return an `ERGMResult` object containing coefficients, standard errors, and fit statistics.

## Overview

The estimation process differs by method:

**MPLE**:
1. Build a (compressed) design matrix of change statistics for all observed dyads
2. Fit the logistic regression by Newton–Raphson on the ecosystem's shared kernel (`Networks.logistic_derivatives` + `Networks.newton_fit`)
3. Compute standard errors from the observed information matrix at the solution

**MCMLE**:
1. Initialize with MPLE estimates
2. Iterate: sample networks via MCMC, update parameters via Newton-Raphson
3. Converge when observed statistics match the expected statistics under the model

## Maximum Pseudo-Likelihood Estimation (MPLE)

MPLE treats each potential edge as an independent observation and fits a logistic regression model using change statistics as features:

$$\text{logit}\left(P(Y_{ij} = 1 \mid Y_{-ij})\right) = \theta^\top \delta(y)_{ij}$$

```julia
using Networks, ERGM
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
solved by Newton–Raphson with step halving on Networks.jl's shared
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
observations, so the inverse-Hessian standard errors are typically
anticonservative and the p-values too optimistic. `show(result)` prints a
caveat in this case (mirroring statnet's warning). Two remedies:

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
┌ Warning: mple: se=:bootstrap — 9 of the 100 bootstrap refits had no finite MPLE (the simulated network put a statistic at the boundary of its attainable range — e.g. no triangle, no shared partner — or perfectly separated the ties) and were excluded; the standard errors are the empirical covariance of the 91 finite refits. This is about the simulated replicates, not about the observed network. …
ERGM Results
============
Method: mple
Log-likelihood: -53.8523
AIC: 113.7, BIC: 122.07
Converged: true

Coefficients:
                  Estimate  Std.Error  z value  Pr(>|z|)
edges              -1.7661     0.4774  -3.6993    0.0002 ***
gwesp.fixed.0.5     0.0973     0.2906   0.3349    0.7377
nodematch.gender    0.1061     0.5782   0.1836    0.8544
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Note: 9 of the 100 parametric-bootstrap refits had no finite MPLE (a statistic at the boundary of its attainable range, or a separated design, in the simulated network) and were excluded from the standard errors, which are the empirical covariance of the remaining 91 refits (fit.boot_replicates)

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
    maxiter = 20,
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
# time on the panel's measurement): coefficients and standard errors are
# bit-identical, `loglikelihood`/`aic`/`bic` are NaN and `show` says so
fast = ergm(net, terms; method=:mcmle, n_samples=1000, rng=Xoshiro(42), bridge_rungs=0)
coef(fast) == coef(ergm(net, terms; method=:mcmle, n_samples=1000, rng=Xoshiro(42)))
isnan(loglikelihood(fast))
```

### How MCMLE Works

The algorithm iterates:

1. **Initialize**: Start with MPLE estimates $\theta^{(0)}$
2. **Sample**: Generate networks from the current model $P_{\theta^{(t)}}$ via Metropolis-Hastings MCMC
3. **Check convergence**: at full Hummel step length, stop when every per-statistic convergence t-ratio $(g_{\text{obs}} - \bar{g})/\text{sd}(g)$ is below `conv_threshold` *and* a Hotelling $T^2$ test of the mean difference is non-significant at `hotelling_alpha`
4. **Update**: partial Newton-Raphson step toward the Hummel pseudo-target: $\theta^{(t+1)} = \theta^{(t)} + \gamma\,\Sigma^{-1}(g(\text{obs}) - \bar{g}(\text{sim}))$, with step length $\gamma$ adapting toward 1 as the sampled statistic cloud covers the observed statistics
5. **Repeat** until convergence or `maxiter` reached — an unconverged fit is returned with `converged == false`, a warning quoting the last max t-ratio, Hotelling p-value and step length, an entry in `Networks.approximations(result)`, a caveat line under `Converged: false` in `show`, and the numbers themselves in `result.mcmc_convergence` (see [Non-convergence](@ref) below)

The convergence tests are one `public` function,
`ERGM.mcmc_convergence(samples, targets; conv_threshold, hotelling_alpha,
chain_lengths)`, so a variant that solves its own moment equations (ERGMEgo)
uses the same rule.

The reported log-likelihood (and AIC/BIC) is estimated by **path
sampling**: a `bridge_rungs`-segment ladder from a dyad-independent
reference distribution (whose normalizer is exact) to $\hat\theta$, with
the expected statistics at each rung estimated by MCMC and integrated by
the trapezoid rule — the standard ergm-style bridge estimator. For fully
dyad-independent models the exact log-likelihood is returned.

### MCMC Sampling

Each MCMC step:
1. Propose: randomly select a dyad $(i,j)$
2. Compute: change statistics $\delta(y)_{ij}$ for all terms
3. Accept/reject: toggle edge with probability $\min(1, \exp(\theta^\top \delta))$

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
| `maxiter` | Maximum NR iterations | 20 | Increase for slow convergence (`max_iter` is a deprecated spelling, warned once) |
| `missing` | Masked-dyad policy | `:error` | `:mle` (missing-data ML) or `:condition_on_face` (see below) |
| `obs_burnin`, `obs_interval` | Constrained-chain controls under `missing=:mle` | dyad-scaled over the masked dyads | R's `obs.MCMC.burnin` / `obs.MCMC.interval` |
| `init` | Starting coefficients | the MPLE | `coef(fit)` to continue an unconverged fit |
| `conv_threshold` | Max allowed convergence t-ratio | 0.1 | Smaller = stricter convergence |
| `hotelling_alpha` | Level of the Hotelling T² convergence test | 0.05 | — |
| `rng` | RNG all draws flow from | `Random.default_rng()` | Pass `Xoshiro(seed)` for reproducibility |
| `bridge_rungs` | Path-sampling segments for the log-likelihood | 16 | More = less bias in AIC/BIC; `0` skips it (loglik/AIC/BIC `NaN`) |
| `bridge_samples` | MCMC samples per bridge rung | `n_samples` | — |

(The `tol` keyword from earlier versions is deprecated and ignored.)

### MCMLE Strengths and Limitations

| Aspect | Detail |
|--------|--------|
| Accuracy | Consistent and asymptotically efficient |
| Standard errors | Correctly accounts for dependencies |
| Speed | Slower — requires MCMC at each iteration |
| Initialization | Benefits from good MPLE starting values |
| Use case | Final results, dependent models |

## Missing (Unobserved) Dyads

Networks.jl can mark dyads whose tie status is **unobserved** (statnet-style
NA ties) with `set_missing_dyad!(net, i, j)` — distinct from "no tie". The
estimation routines treat the mask as follows:

- **MPLE excludes masked dyads.** An unobserved tie status is not a
  response, so masked dyads contribute no row to the logistic-regression
  design; `nobs(result)` shrinks by the number of masked dyads. The masked
  dyads' face values (edge present/absent as stored) still enter the change
  statistics of the observed dyads, which are computed conditional on the
  rest of the network. This is the available-case pseudo-likelihood, a
  principled treatment: `Networks.supports_missing(mple) == true`, and
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
(`Networks.check_statsapi(result; strict=true)` passes):

```julia
coef(result)          # Coefficient vector
stderror(result)      # Standard errors vector
vcov(result)          # Variance-covariance matrix
confint(result)       # Normal-theory limits, one row per coefficient (level=0.95)
coeftable(result)     # Networks.CoefficientTable — exactly the table `show` prints
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
Method: mple
Log-likelihood: -53.8523
AIC: 113.7, BIC: 122.07
Converged: true

Coefficients:
                  Estimate  Std.Error  z value  Pr(>|z|)
edges              -1.7661     0.3750  -4.7096   2.5e-06 ***
gwesp.fixed.0.5     0.0973     0.1697   0.5736    0.5662
nodematch.gender    0.1061     0.5006   0.2120    0.8321
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Warning: this model contains dyad-dependent terms and was fit by
maximum pseudolikelihood (MPLE). The standard errors are based on the
naive pseudolikelihood and are suspect (typically anticonservative);
the p-values should not be trusted. Refit with method=:mcmle, or use
se=:bootstrap for parametric-bootstrap standard errors.
```

An MCMLE fit of the same model prints R's "MCMC %" line after the table
instead of the pseudo-likelihood warning:

```julia
println(ergm(net, terms; method=:mcmle, n_samples=1000, rng=Xoshiro(1)))
```

```text
ERGM Results
============
Method: mcmle
Log-likelihood: -53.9462
AIC: 113.89, BIC: 122.25
Converged: true

Coefficients:
                  Estimate  Std.Error  z value  Pr(>|z|)
edges              -1.7762     0.4301  -4.1302   3.6e-05 ***
gwesp.fixed.0.5     0.1004     0.2619   0.3835    0.7014
nodematch.gender    0.1486     0.4783   0.3107    0.7560
---
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

MCMC % of the standard error (100·(se − se_fisher)/se): edges 0, gwesp.fixed.0.5 0, nodematch.gender 0
```

A fit that exhausts `maxiter` prints `Converged: false` with the
non-convergence caveat directly under it (the same sentence the warning
carries and `Networks.approximations(result)` lists) — see
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

The limits are built from the standard errors the fit reports, so for an MPLE
fit of a dyad-dependent model they inherit the anticonservative
pseudo-likelihood SEs (use `se=:bootstrap` or `method=:mcmle`).

## Model Comparison

### AIC and BIC

```julia
# Fit multiple models
terms1 = [Edges()]
terms2 = [Edges(), Triangle()]
terms3 = [Edges(), GWESP(0.5), NodeMatch(:gender)]

r1 = ergm(net, terms1)
r2 = ergm(net, terms2)
r3 = ergm(net, terms3)

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

- a warning at fit time: `MCMLE did not converge in maxiter=20 iterations
  (last max t-ratio 0.25, Hotelling p 0.007, step length γ 1.0): the
  estimates are the last iterate and the standard errors are unreliable;
  increase maxiter/n_samples/burnin, check the model for degeneracy
  (mcmc_diagnostics), or refit from these coefficients
  (mcmle(model; init=coef(fit)))`;
- the same sentence under `Converged: false` when the result is shown;
- an entry in `Networks.approximations(result)` (so
  `fit_metadata(result)` sees it);
- the numbers in `result.mcmc_convergence`: the t-ratios, Hotelling
  p-value and effective sample size recomputed on the final sample at the
  returned coefficients, plus the iteration count and the last step length.

```julia
if !result.converged
    c = result.mcmc_convergence
    println("max t-ratio ", maximum(c.t_ratios), ", Hotelling p ", c.hotelling_p)
    # continue from where it stopped, with a bigger budget
    result = mcmle(ERGMModel(ERGMFormula(terms), net);
                   init=coef(result), n_samples=4000, maxiter=40, rng=Xoshiro(2))
end
```

### Common Causes and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Model degeneracy | Very large coefficients, non-convergence | Use GWESP/GWDegree instead of Triangle/Kstar |
| Near-degeneracy | Slow convergence, unstable estimates | Simplify model, use geometrically weighted terms |
| A statistic at the boundary of its attainable range (a `NodeMatch` with no within-group tie; a `NodeCov` whose positive-change dyads are all empty and negative-change dyads all tied, …) | `mple` warns "observed statistic(s) … are at their smallest attainable values. Their coefficients will be fixed at -Inf", returns `-Inf` (SE 0) for that term and R's estimates, `dof`, AIC and BIC of the rest; `mcmle` refuses with the same sentence, before computing anything | Remove the term (as R's `drop=TRUE` does), or keep the `mple` fit and read the `-Inf` as "never" |
| Perfect separation by a *combination* of statistics (no single one at its boundary, but together they predict every tie) | `mple` warns "the MPLE does not exist (perfect separation)" (R: "The MPLE does not exist!"), returns the last Newton iterate with `converged == false`, `is_exact == false` and the caveat in `show`/`approximations`; `mcmle` refuses to start from it | Remove or coarsen a term, or pass `init=` to `mcmle` |
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
9. **Honest uncertainty**: For dyad-dependent MPLE fits, use `se=:bootstrap` or refit with `method=:mcmle` before interpreting p-values
