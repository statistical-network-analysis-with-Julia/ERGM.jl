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

No docs build system is set up yet.

## Architecture

**Type hierarchy** (`src/terms/base.jl`): `AbstractERGMTerm` is the root, with subtypes `StructuralTerm`, `NodalTerm`, `DyadicTerm`, and `ConstraintTerm`. Each term must implement `compute(term, net) -> Float64` and `change_stat(term, net, i, j) -> Float64`.

**Core data flow**: Terms are collected into a `TermSet`, wrapped in an `ERGMFormula` (with optional constraints), then combined with a `Network` into an `ERGMModel`. Fitting produces an `ERGMResult` containing coefficients, standard errors, p-values, and fit statistics.

**Estimation** (`src/estimation/`):
- `mple.jl` - Builds a full dyad-level design matrix from `change_stat` values, then fits logistic regression via `Optim.LBFGS`.
- `mcmle.jl` - Initializes from MPLE, then iterates Newton-Raphson updates using MCMC-sampled network statistics. Uses Metropolis-Hastings edge toggling.

**Simulation** (`src/mcmc/simulation.jl`): `simulate_ergm` and `sample_networks` use the same MH edge-toggle sampler as MCMLE.

**Diagnostics** (`src/mcmc/diagnostics.jl`): `gof()` compares observed degree/ESP/geodesic distributions against simulated networks. `mcmc_diagnostics()` computes autocorrelation and effective sample size from MCMLE samples.

**Entry point**: `ergm(net, terms; method=:mple)` and `fit_ergm(net, terms; ...)` are aliases defined at the bottom of `mple.jl`.

## Key Dependencies

- **Network.jl** (`Network` package) - Network data structure with vertex/edge attributes (`get_vertex_attribute`, `get_edge_attribute`, etc.)
- **Graphs.jl** - Graph algorithms; `Network` wraps a `Graphs` graph internally (accessed via `net.graph` for things like `gdistances`)
- **Optim.jl** - LBFGS optimizer for MPLE logistic regression
- **Distributions.jl** - `Normal()` CDF for p-value computation

## Conventions

- All term statistics return `Float64`.
- `change_stat` returns positive values when adding an edge increases the statistic, negative when removing.
- Term `name()` functions use R ergm-style lowercase names (e.g., `"edges"`, `"nodematch.gender"`, `"gwesp.fixed.0.5"`).
- Internal/private functions are prefixed with underscore (e.g., `_mcmc_sample`, `_copy_network`).
- Tests use `@testset` blocks grouped by feature in a single `test/runtests.jl` file.
- The package depends on a local/unregistered `Network` and `SNA` package (not in General registry).
