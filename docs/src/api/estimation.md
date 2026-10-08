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

### resolve_method

The default `method=:auto` follows R: the MPLE (exact) when no term is
dyad-dependent, the MCMLE otherwise. The rule is `public` (not exported) so
the ERGM variants resolve `:auto` the same way.

```@docs
ERGM.resolve_method
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

### The MCMLE iteration's building blocks

`mcmle` is built from drivers that know nothing about networks: the
iteration, R ergm 4's `confidence` stopping rule, the covariance, the
ESS-adaptive sample-size chooser and the bridge integral behind the
log-likelihood. They are part of the [Extension API](@ref extension-api),
so a variant's own MCMC estimator (ERGMEgo's moment matching, TERGM's CMLE,
ERGMRank's and ERGMCount's MCMLEs) runs on the same ones:
[`mcmle_solve`](@ref ERGM.Extension.mcmle_solve),
[`confidence_test`](@ref ERGM.Extension.confidence_test),
[`mcmle_covariance`](@ref ERGM.Extension.mcmle_covariance),
[`ess_sample`](@ref ERGM.Extension.ess_sample),
[`mcmle_sampler`](@ref ERGM.Extension.mcmle_sampler) (`mcmle`'s own sampler,
for an estimator that samples an `ERGMModel`) and
[`bridge_integrate`](@ref ERGM.Extension.bridge_integrate).

### newton_fit / logistic_derivatives

The shared optimizer and logistic kernel are **NetworkCore.jl's** (`public`
there) and re-exported by ERGM.jl, so `using ERGM` provides them unchanged
and `ERGM.newton_fit === NetworkCore.newton_fit`. They are documented on
NetworkCore.jl's
[inference page](https://Statistical-network-analysis-with-Julia.github.io/NetworkCore.jl/dev/api/inference/);
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
rule, [`mcmc_defaults`](@ref ERGM.Extension.mcmc_defaults) (part of the
[Extension API](@ref extension-api)): `burnin = 20 × n_dyads` and
`interval = max(100, n_dyads ÷ 10)`.

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
[result-metadata protocol](https://Statistical-network-analysis-with-Julia.github.io/NetworkCore.jl/dev/api/metadata/),
so what a fit actually did is programmatically inspectable rather than buried in
a `show` method. `NetworkCore.fit_metadata(result)` collects these accessors.

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
[missing-data contract](https://Statistical-network-analysis-with-Julia.github.io/NetworkCore.jl/dev/api/contracts/)):

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

## [Renamed and removed names](@id renames)

0.2.0 is the first public release. Names that existed only in development
versions were renamed or removed outright, without deprecation shims:

| Development name | 0.2.0 | Note |
|:--|:--|:--|
| `using Networks` | `using NetworkCore` | the foundation package was renamed; types and functions keep their names |
| `mcmle(...; max_iter=)` | `mcmle(...; maxiter=)` | the ecosystem's iteration-cap keyword |
| `mcmle(...; tol=)` | removed | it was ignored: convergence is the stopping rule `termination=` (R's confidence test by default) |
| `ERGM._requires_directed`, `ERGM._requires_undirected` | `requires_directed`, `requires_undirected` | the exported term traits |
| `ERGM._vertex_attribute(t)` | `required_vertex_attributes(t)` | a tuple of every attribute the term reads |
| `ERGM._has_dyad_dependent` | `has_dyad_dependent` | exported |
| `ERGM._z_pvalues` | `NetworkCore.z_pvalues` | the one z → p helper |
| `ERGM._hummel_step`, `ERGM._bridge_quadrature`, `ERGM._bridge_logZ`, `ERGM._warn_degenerate_stats` | internal (no longer `public`) | use the drivers [`ERGM.Extension.mcmle_solve`](@ref) and [`ERGM.Extension.bridge_integrate`](@ref) |
| `ERGM._separated`, `ERGM._warn_separated` | removed | NetworkCore's `logistic_separation` / `warn_separation`, or the `verdict` that [`ERGM.Extension.mple_fit_design`](@ref) returns |
| the `public` underscore names of the extension surface (`ERGM._mple_fit_design`, `ERGM._mcmle_solve`, …) | [`ERGM.Extension`](@ref) | no longer `public`; the [name map](@ref extension-name-map) gives each one's replacement |

One default changed meaning rather than name: `fit_ergm`/`ergm` take
`method=:auto` (R's rule, [`ERGM.resolve_method`](@ref)) instead of
`method=:mple`. A dyad-independent formula is fitted exactly as before; a
dyad-dependent one is now fitted by MCMLE, as R's `ergm()` does. Pass
`method=:mple` for the pseudo-likelihood fit.
