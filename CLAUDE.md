# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ERGM.jl is a Julia port of the R `ergm` package (from StatNet) for fitting, simulating, and diagnosing Exponential-Family Random Graph Models. It provides MPLE and MCMLE estimation, MCMC network simulation, and goodness-of-fit diagnostics.

## Development Commands

```bash
# Run tests
julia --project -e 'using Pkg; Pkg.test()'

# Load package in REPL
julia --project -e 'using ERGM'
```

In the Julia REPL:
```julia
using Pkg; Pkg.test()          # run tests
using ERGM                      # load package
```

Docs build with Documenter via `docs/make.jl`. Julia 1.12+ is required (Network.jl cannot load on earlier versions).

## Architecture

**Type hierarchy** (`src/terms/base.jl`): `AbstractERGMTerm` is the root, with subtypes `StructuralTerm`, `NodalTerm`, `DyadicTerm`, and `ConstraintTerm`. Each term must implement `compute(term, net) -> Float64` and `change_stat(term, net, i, j) -> Float64`.

**Core data flow**: Terms are collected into a `TermSet`, wrapped in an `ERGMFormula` (with optional constraints), then combined with a `Network` into an `ERGMModel`. Model construction validates the formula (`src/terms/materialize.jl`): attribute-based terms whose vertex attribute is missing from the network, and intrinsically directed terms (`Mutual`) on undirected networks, throw `ArgumentError`. Attribute-based nodal terms are then *materialized* into typed twins (`MaterializedNodeCov`, ...) that snapshot the attribute into dense vectors, so hot-loop change statistics avoid the untyped attribute Dicts (semantics and names are preserved; verified against the raw terms in tests). Multi-level `NodeFactor` (first sorted level dropped by default — statnet's `base=1`) and multi-cell `NodeMix` (first cell dropped — `levels2=-1`) expand into one materialized statistic per level/cell at model construction; `Degree`/`IDegree`/`ODegree` accept degree vectors/ranges that expand to one term per degree at construction time. Fitting produces an `ERGMResult` containing coefficients, standard errors, p-values, fit statistics, and `se_type` (`:hessian`/`:bootstrap`/`:mcmc`). `is_dyad_dependent(term)` classifies terms (conservative default `true` for unknown types); `show(::ERGMResult)` prints a statnet-style caveat for pseudo-likelihood fits of dyad-dependent models.

**Estimation** (`src/estimation/`):
- `mple.jl` - Builds a compressed dyad-level design matrix from `change_stat` values, then fits logistic regression via `Optim.LBFGS` (`_mple_fit` core). `se=:bootstrap` adds parametric-bootstrap SEs (simulate at the MPLE, refit each, empirical covariance; refits threaded). Dyads masked as missing (`Network.set_missing_dyad!`) are excluded from the design matrix and from `_n_dyads`/`nobs`.
- `mcmle.jl` - Initializes from MPLE, then iterates Hummel-stepped Newton-Raphson updates using MCMC-sampled statistics; convergence via per-statistic t-ratios plus a Hotelling T² test. The reported log-likelihood (AIC/BIC) comes from `_bridge_loglik`: a path-sampling ladder from a dyad-independent reference (exact `_dyad_independent_logZ`) to θ̂, trapezoid-integrated, rungs parallelized with deterministic per-rung seeds. `_mcmc_sample` remains as a thin wrapper over `mh_sample`.
- `newton.jl` - Public `newton_fit(loglik_grad_hess, θ0)`: shared Newton–Raphson maximizer with step halving (factored from `Relevent._newton`) for downstream packages.

**Simulation** (`src/mcmc/simulation.jl`): everything is built on the single MH edge-toggle kernel `_mh_run!`. Masked (missing) dyads are never proposed for toggling — the chain conditions on their face values (`mcmle` warns once about this; full missing-data ML is future work) — and `_random_network` starting states preserve them. `mh_sample(model, θ; ...)` is the public single-chain primitive (returns `(stats, networks)`; used by ERGMEgo.jl). `sample_networks`/`simulate_ergm` split draws over `n_chains` independent chains via `Threads.@spawn`, each seeded deterministically from the caller's `rng` — results are reproducible and thread-count-independent (`n_chains` must never default to `Threads.nthreads()`).

**Diagnostics** (`src/mcmc/diagnostics.jl`): `gof()` compares observed degree/ESP/geodesic distributions against simulated networks (accepts `rng`, `n_chains`, `burnin`, `interval`). `mcmc_diagnostics()` computes autocorrelation and effective sample size from MCMLE samples; it throws `ArgumentError` on fits without MCMC samples (MPLE).

**Entry point**: `ergm(net, terms; method=:mple)` and `fit_ergm(net, terms; ...)` are aliases defined at the bottom of `mple.jl`.

## Key Dependencies

- **Network.jl** (`Network` package) - Network data structure with vertex/edge attributes (`get_vertex_attribute`, `get_edge_attribute`, etc.)
- **Graphs.jl** - Graph algorithms; `Network` wraps a `Graphs` graph internally (accessed via `net.graph` for things like `gdistances`)
- **Optim.jl** - LBFGS optimizer for MPLE logistic regression
- **Distributions.jl** - `Normal()` CDF for p-value computation

## Conventions

- All term statistics return `Float64`.
- `change_stat` returns the state-independent add-direction change statistic `g(y⁺ij) − g(y⁻ij)` (statistic with edge (i,j) present minus absent, rest of the network fixed). It must NOT depend on whether the edge currently exists; the MH sampler negates it for removal proposals.
- Term `name()` functions use R ergm-style lowercase names (e.g., `"edges"`, `"nodematch.gender"`, `"gwesp.fixed.0.5"`).
- Internal/private functions are prefixed with underscore (e.g., `_mcmc_sample`, `_copy_network`).
- All randomness flows through `rng::AbstractRNG` keywords (default `Random.default_rng()`); never call bare `rand()` in sampling/fitting code. Parallel loops draw one seed per task from the caller's rng.
- Tests use `@testset` blocks grouped by feature in a single `test/runtests.jl` file.
- The package depends on the local/unregistered `Network` package (not in General registry), resolved via `[sources]` from `../Network.jl`.
