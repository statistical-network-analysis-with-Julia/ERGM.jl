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
| Standard errors | May underestimate SEs for dependent models |
| Use case | Exploration, initial estimates, independent-dyad models |

## Monte Carlo Maximum Likelihood Estimation (MCMLE)

MCMLE uses MCMC sampling to approximate the normalizing constant ratio, providing more accurate estimates for models with dependence:

```julia
result = ergm(net, terms;
    method = :mcmle,
    n_samples = 1000,
    burnin = 1000,
    interval = 100,
    max_iter = 20,
    tol = 1e-4,
    verbose = true
)
```

### How MCMLE Works

The algorithm iterates:

1. **Initialize**: Start with MPLE estimates $\theta^{(0)}$
2. **Sample**: Generate networks from the current model $P_{\theta^{(t)}}$ via Metropolis-Hastings MCMC
3. **Check convergence**: If $\max|\bar{g}(\text{sim}) - g(\text{obs})| < \text{tol}$, stop
4. **Update**: Newton-Raphson step: $\theta^{(t+1)} = \theta^{(t)} + \Sigma^{-1}(g(\text{obs}) - \bar{g}(\text{sim}))$
5. **Repeat** until convergence or `max_iter` reached

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
| `burnin` | Steps before sampling | 1000 | Increase if poor mixing |
| `interval` | Steps between samples | 100 | Increase to reduce autocorrelation |
| `max_iter` | Maximum NR iterations | 20 | Increase for slow convergence |
| `tol` | Convergence tolerance | 1e-4 | Smaller = stricter convergence |

### MCMLE Strengths and Limitations

| Aspect | Detail |
|--------|--------|
| Accuracy | Consistent and asymptotically efficient |
| Standard errors | Correctly accounts for dependencies |
| Speed | Slower — requires MCMC at each iteration |
| Initialization | Benefits from good MPLE starting values |
| Use case | Final results, dependent models |

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
using Distributions

alpha = 0.05
z = quantile(Normal(), 1 - alpha/2)

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
8. **Set random seeds**: For reproducibility in MCMLE
