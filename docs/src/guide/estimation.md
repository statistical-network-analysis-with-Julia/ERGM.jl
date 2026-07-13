# Model Estimation

ERGM.jl provides two estimation methods: Maximum Pseudo-Likelihood Estimation (MPLE) for fast approximation and Monte Carlo Maximum Likelihood Estimation (MCMLE) for accurate inference. Both methods return an `ERGMResult` object containing coefficients, standard errors, and fit statistics.

## Overview

The estimation process differs by method:

**MPLE**:
1. Build a design matrix of change statistics for all potential edges
2. Fit logistic regression via LBFGS optimization
3. Compute standard errors from the Fisher information matrix

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

The logistic regression is solved via LBFGS optimization (from Optim.jl), with numerical stability ensured by clamping linear predictors to $[-500, 500]$.

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
result_mcmle = ergm(net, terms; method=:mcmle)
```

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
    burnin = 1000,
    interval = 100,
    max_iter = 20,
    verbose = true
)
```

### How MCMLE Works

The algorithm iterates:

1. **Initialize**: Start with MPLE estimates $\theta^{(0)}$
2. **Sample**: Generate networks from the current model $P_{\theta^{(t)}}$ via Metropolis-Hastings MCMC
3. **Check convergence**: at full Hummel step length, stop when every per-statistic convergence t-ratio $(g_{\text{obs}} - \bar{g})/\text{sd}(g)$ is below `conv_threshold` *and* a Hotelling $T^2$ test of the mean difference is non-significant at `hotelling_alpha`
4. **Update**: partial Newton-Raphson step toward the Hummel pseudo-target: $\theta^{(t+1)} = \theta^{(t)} + \gamma\,\Sigma^{-1}(g(\text{obs}) - \bar{g}(\text{sim}))$, with step length $\gamma$ adapting toward 1 as the sampled statistic cloud covers the observed statistics
5. **Repeat** until convergence or `max_iter` reached

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

The sampler uses burn-in to reach stationarity and thinning to reduce autocorrelation.

### MCMLE Parameters

| Parameter | Description | Default | Guidance |
|-----------|-------------|---------|----------|
| `n_samples` | MCMC samples per iteration | 1000 | More = better approximation |
| `burnin` | Steps before sampling | `20 × n_dyads` | Increase if poor mixing |
| `interval` | Steps between samples | `max(100, n_dyads ÷ 10)` | Increase to reduce autocorrelation |
| `max_iter` | Maximum NR iterations | 20 | Increase for slow convergence |
| `conv_threshold` | Max allowed convergence t-ratio | 0.1 | Smaller = stricter convergence |
| `hotelling_alpha` | Level of the Hotelling T² convergence test | 0.05 | — |
| `rng` | RNG all draws flow from | `Random.default_rng()` | Pass `Xoshiro(seed)` for reproducibility |
| `bridge_rungs` | Path-sampling segments for the log-likelihood | 16 | More = less bias in AIC/BIC |
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
  rest of the network.
- **MCMLE conditions on face values.** The MH sampler never proposes
  toggling a masked dyad, so all simulation happens conditional on the
  stored face values, and the observed sufficient statistics include the
  masked dyads at face value. `mcmle` emits a one-time warning when the
  network has masked dyads. This is an honest, clearly labeled
  approximation — full missing-data maximum likelihood (statnet's approach
  of integrating over the unobserved dyads) is future work.

```julia
set_missing_dyad!(net, 3, 4)          # dyad 3->4 was not measured
fit = ergm(net, terms; method=:mple)  # excluded from the pseudo-likelihood
nobs(fit)                             # one fewer observation
```

## Comparing Methods

```julia
# Quick exploration
result_mple = ergm(net, terms; method=:mple)

# Final analysis
result_mcmle = ergm(net, terms; method=:mcmle, verbose=true)

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
| `aic` | `Float64` | Akaike Information Criterion |
| `bic` | `Float64` | Bayesian Information Criterion |
| `method` | `Symbol` | `:mple` or `:mcmle` |
| `converged` | `Bool` | Convergence status |
| `mcmc_samples` | `Matrix{Float64}` or `nothing` | MCMC samples (MCMLE only) |
| `se_type` | `Symbol` | `:hessian`, `:bootstrap`, or `:mcmc` — how the SEs were obtained |

### Accessor Functions

```julia
coef(result)       # Coefficient vector
stderror(result)   # Standard errors vector
vcov(result)       # Variance-covariance matrix
```

### Displaying Results

```julia
println(result)
```

Output:

```text
ERGM Results
============
Method: mple
Log-likelihood: -45.6789
AIC: 97.36, BIC: 103.12
Converged: true

Coefficients:
------------------------------------------------------------
edges                  -2.3456     0.4321     0.0000 ***
triangle                0.8765     0.3210     0.0063 **
nodematch.gender        0.5432     0.2890     0.0601 .
------------------------------------------------------------
Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

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
z = 1.959964  # standard-normal 97.5% quantile, i.e. quantile(Normal(), 0.975)

lower = coef(result) .- z .* stderror(result)
upper = coef(result) .+ z .* stderror(result)

# Odds ratio confidence intervals
or_lower = exp.(lower)
or_upper = exp.(upper)
```

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

### Checking Convergence

```julia
if !result.converged
    @warn "Model did not converge — results may be unreliable"
end
```

### Common Causes and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Model degeneracy | Very large coefficients, non-convergence | Use GWESP/GWDegree instead of Triangle/Kstar |
| Near-degeneracy | Slow convergence, unstable estimates | Simplify model, use geometrically weighted terms |
| Perfect separation | Infinite coefficients | Remove or combine problematic terms |
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
