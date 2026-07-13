# Estimation API Reference

This page documents the functions for model fitting, simulation, and diagnostics.

## Model Fitting

### ergm / fit_ergm

```@docs
fit_ergm
```

### mple

```@docs
mple
```

### mcmle

```@docs
mcmle
```

### newton_fit

```@docs
newton_fit
logistic_derivatives
```

## Simulation

### mh_sample

```@docs
mh_sample
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

### is_dyad_dependent

```@docs
is_dyad_dependent
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
