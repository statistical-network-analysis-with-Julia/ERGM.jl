# Estimation API Reference

This page documents the functions for model fitting, simulation, and diagnostics.

## Model Fitting

### ergm / fit_ergm

`ergm` is a `const` alias of `fit_ergm` (`ergm === fit_ergm`): one method
table, one keyword vocabulary.

```@docs
fit_ergm
ergm
```

### mple

```@docs
mple
```

### mcmle

```@docs
mcmle
```

### mcmc_se

```@docs
mcmc_se
```

### mcmc_convergence

The MCMLE convergence tests as one `public` (not exported) function, so a
variant that solves its own moment equations uses the same rule.

```@docs
ERGM.mcmc_convergence
```

### newton_fit / logistic_derivatives

The shared optimizer and logistic kernel are **Networks.jl's** (`public`
there) and re-exported by ERGM.jl, so `using ERGM` provides them unchanged
and `ERGM.newton_fit === Networks.newton_fit`. They are documented on
Networks.jl's
[inference page](https://Statistical-network-analysis-with-Julia.github.io/Networks.jl/dev/api/inference/);
ERGM's MPLE is `newton_fit` applied to the binomial-row form of
`logistic_derivatives`:

```julia
using ERGM
# Edges-only MPLE of the Florentine marriage network in closed form: 120
# dyads, 20 ties, so θ̂ = logit(20/120)
d = logistic_derivatives(ones(1, 1), [120.0], [20.0])
fit = newton_fit(d, [0.0])
fit.θ[1] ≈ log(20 / 100)     # true
fit.converged                # true
```

## Simulation

### mh_toggle!

The Metropolis toggle kernel every ERGM-family sampler is built on: the
accept/reject arithmetic, burn-in and thinning, with the proposal, the change
statistics and the state mutation supplied as callables.

```@docs
mh_toggle!
```

### mh_sample

```@docs
mh_sample
```

### Sampler defaults

Every sampler resolves an omitted `burnin`/`interval` through one dyad-scaled
rule (`public`, not exported):

```@docs
ERGM._mcmc_defaults
```

### simulate_ergm

```@docs
simulate_ergm
```

### sample_networks

```@docs
sample_networks
```

## Utilities

### is_dyad_dependent / has_dyad_dependent

```@docs
is_dyad_dependent
has_dyad_dependent
```

## Result Metadata

ERGM.jl implements the ecosystem's
[result-metadata protocol](https://Statistical-network-analysis-with-Julia.github.io/Networks.jl/dev/api/metadata/),
so what a fit actually did is programmatically inspectable rather than buried in
a `show` method. `Networks.fit_metadata(result)` collects these accessors.

The key one is [`is_exact`](@ref): an MPLE fit of a **dyad-independent** formula
*is* the exact MLE, while the same estimator on a formula containing any
dyad-dependent term (`Triangle`, `GWESP`, `Mutual`, ...) is an approximation
with anticonservative standard errors. That distinction is exactly what a user
needs and cannot otherwise see.

```@docs
objective(::ERGMResult)
is_exact(::ERGMResult)
se_method(::ERGMResult)
```

The missing-data trait a term opts into (see the ecosystem
[missing-data contract](https://Statistical-network-analysis-with-Julia.github.io/Networks.jl/dev/api/contracts/)):

```@docs
supports_missing(::AbstractERGMTerm)
```

## Diagnostics

### gof

```@docs
gof
```

### mcmc_diagnostics

```@docs
mcmc_diagnostics
```
