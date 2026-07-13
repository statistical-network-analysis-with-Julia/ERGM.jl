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

Docs build with Documenter via `docs/make.jl`. Julia 1.12+ is required (Networks.jl cannot load on earlier versions).

## Architecture

**Type hierarchy** (`src/terms/base.jl`): `AbstractERGMTerm` is the root, with subtypes `StructuralTerm`, `NodalTerm`, `DyadicTerm`, and `ConstraintTerm`. Each term must implement `compute(term, net) -> Float64` and `change_stat(term, net, i, j) -> Float64`.

**The statistic protocol is shared, not ERGM's** (Networks.jl `src/statistics.jl`): `compute`, `name` and `compute_all` are empty generics defined in the foundation and imported by name here (`import Networks: compute, name, compute_all`), exactly like `gof`. ERGM adds the term methods; REM.jl adds relational-event methods with a different signature; they are methods of ONE function, so `ERGM.compute === REM.compute` and `using ERGM, REM` leaves the verbs usable unqualified — before this, two distinct exported functions of the same name left them *undefined* under Julia's conflicting-export rule (REM.jl#3). Never define a local `compute`/`name`; downstream term packages (ERGMCount, ERGMMulti, ERGMRank, ERGMEgo, ERGMUserterms, TERGM) correctly do `import ERGM: name, compute`, which resolves to the shared generics.

**Term traits are a public protocol** (`src/terms/traits.jl`): `required_vertex_attributes(term)` / `required_edge_attributes(term)` (tuples of Symbols, default `()`), `requires_directed(term)` / `requires_undirected(term)` (default `false`), `is_dyad_dependent(term)` (conservative default `true`), and `Networks.supports_missing(term)` (default `false` — every built-in term counts a masked dyad at face value; the principled missing-data treatment lives in the *estimator*, see below). All are exported and documented. Formula validation and materialization read the *traits*, never a term's fields, so a term from any package participates in exactly the same checks as a built-in — that is the point of the protocol (ERGMUserterms.jl#1; `ERGMUserterms.validate_term` exercises each declaration, and `ERGMUserterms.jl/examples/MyTermPackage` is a package template using all four). The traits were internal before v0.2 and TERGM.jl had already reached into them, so the private names survive as **`const` aliases of the same generics** (`_requires_directed`, `_requires_undirected`) — TERGM's shipped `ERGM._requires_directed(::Delrecip) = true` therefore still adds a method to the function validation dispatches on. `_vertex_attribute(term)` remains as a shim returning the first required vertex attribute or `nothing`. Never re-privatize these; never make validation introspect fields.

**Core data flow**: Terms are collected into a `TermSet`, wrapped in an `ERGMFormula` (with optional constraints), then combined with a `Network` into an `ERGMModel`. Model construction validates the formula (`src/terms/materialize.jl`) against the term traits above: terms whose declared vertex/edge attributes are missing from the network, and terms declaring a direction requirement the network does not meet (`Mutual` on an undirected network), throw `ArgumentError`. Attribute-based nodal terms are then *materialized* into typed twins (`MaterializedNodeCov`, ...) that snapshot the attribute into dense vectors, so hot-loop change statistics avoid the untyped attribute Dicts (semantics and names are preserved; verified against the raw terms in tests). Multi-level `NodeFactor` (first sorted level dropped by default — statnet's `base=1`) and multi-cell `NodeMix` (first cell dropped — `levels2=-1`) expand into one materialized statistic per level/cell at model construction; `Degree`/`IDegree`/`ODegree` accept degree vectors/ranges that expand to one term per degree at construction time. Fitting produces an `ERGMResult` containing coefficients, standard errors, p-values, fit statistics, `se_type` (`:hessian`/`:bootstrap`/`:mcmc`), and `missing_method` (`:none`/`:available_case`/`:condition_on_face`; see below). `is_dyad_dependent(term)` classifies terms (conservative default `true` for unknown types); `show(::ERGMResult)` prints a statnet-style caveat for pseudo-likelihood fits of dyad-dependent models, and reports the missing-dyad treatment whenever the network carried a mask.

**Missing dyads** (`src/missing.jl`): ERGM.jl honours the Networks.jl missing-data contract (`supports_missing` / `require_observed`, imported by name because ERGM adds a method to the trait). MPLE excludes masked dyads from the pseudo-likelihood (available-case) — a principled treatment, so `supports_missing(mple) == true` and it needs no keyword. Everything MCMC-based (`mcmle`, `mh_sample`, `sample_networks`, `simulate_ergm`, `gof`) can only *freeze* a masked dyad at its stored face value, which is a different estimand, so each takes `missing::Symbol=:error` and rejects masked networks by default; `missing=:condition_on_face` is the explicit, warned opt-in (`_guard_missing` validates and returns the method; `_warn_condition_on_face` is called by user-facing entry points only, so exactly one warning per user call). The ERGM opt-in symbol is deliberately `:condition_on_face`, not Network's generic `:face`. Full missing-data MCMLE is unimplemented (issue #4).

**Estimation** (`src/estimation/`):
- `mple.jl` - Builds a compressed dyad-level design matrix from `change_stat` values, then fits logistic regression via `Optim.LBFGS` (`_mple_fit` core). `se=:bootstrap` adds parametric-bootstrap SEs (simulate at the MPLE, refit each, empirical covariance; refits threaded — the simulations condition masked dyads on their face value, and say so). Dyads masked as missing (`set_missing_dyad!`) are excluded from the design matrix and from `_n_dyads`/`nobs`.
- `mcmle.jl` - Initializes from MPLE, then iterates Hummel-stepped Newton-Raphson updates using MCMC-sampled statistics; convergence via per-statistic t-ratios plus a Hotelling T² test. The reported log-likelihood (AIC/BIC) comes from `_bridge_loglik`: a path-sampling ladder from a dyad-independent reference (exact `_dyad_independent_logZ`) to θ̂, trapezoid-integrated, rungs parallelized with deterministic per-rung seeds. `_mcmc_sample` remains as a thin wrapper over `mh_sample`.
- `newton.jl` - Public `newton_fit(loglik_grad_hess, θ0)`: shared Newton–Raphson maximizer with step halving (factored from `Relevent._newton`) for downstream packages.
- `newton.jl` also hosts **`logistic_derivatives(X, y; offset)`** — the shared, allocation-free logistic `(ll, grad, hess)` builder (review finding 15). `ERGMMulti.ergm_multi`'s MPLE over the within-layer dyads, `TERGM.cmple`'s over the free dyads of the auxiliary networks, and `ERGMRank.ergm_rank`'s swap MPLE over the (ego, alter-pair) comparisons are **the same logistic likelihood**, and each carried its own copy of the loop with a per-row `x * x'` outer product inside it. One builder now: workspaces allocated once, derivatives formed by gemv/gemm over the whole design (`η = Xβ`, `∇ = X'r`, `−H = X'WX`), and an evaluation allocates only the `p` gradient and `p×p` Hessian it returns — 192-304 bytes, **independent of the number of rows** (was 649 KB on a 3120-row ERGMMulti design), and 4-8x faster. ERGMRank passes `y = trues(...)`: the observed order is always the "success". `offset` is `ergm`'s offset mechanism (a fixed per-row addition to the linear predictor, used by ERGMMulti for held-fixed coefficients). **Never paste the loop back into a package**; add to this one. The `@allocated` regression tests in all four packages exist to make that stick.

**The bootstrap loop is shared, not ERGM's** (Networks.jl `src/bootstrap.jl`): `_mple_bootstrap_cov` supplies only the two callbacks (how to simulate at θ̂, how to refit) to `Networks.bootstrap_cov`, which owns the loop, the threading and the rng discipline. ERGM's parametric bootstrap is the *reference implementation* the downstream MPLEs were rolled out from — `ERGMCount.count_mple`, `ERGMRank.fit_ergm_rank` and `ERGMMulti.ergm_multi` all now take the same `se=:bootstrap` with the same `n_boot`/`boot_burnin`/`boot_interval`/`rng` keywords and the same semantics (point estimates unchanged, covariance replaced), and `REM.fit_rem` uses the same loop to redraw its case-control risk set. It lives in Networks.jl rather than here, unlike `newton_fit`, for one reason: REM.jl does not depend on ERGM.jl. Never copy the loop into a package; add the callbacks.

**Simulation** (`src/mcmc/simulation.jl`): everything is built on the single MH edge-toggle kernel `_mh_run!`. Masked (missing) dyads are never proposed for toggling and `_random_network` starting states preserve them, so the chain conditions on their face values — which is why every sampler entry point requires the explicit `missing=:condition_on_face` opt-in to run on a masked network at all (see **Missing dyads** above). `mh_sample(model, θ; ...)` is the public single-chain primitive (returns `(stats, networks)`; used by ERGMEgo.jl). `sample_networks`/`simulate_ergm` split draws over `n_chains` independent chains via `Threads.@spawn`, each seeded deterministically from the caller's `rng` — results are reproducible and thread-count-independent (`n_chains` must never default to `Threads.nthreads()`).

**Diagnostics** (`src/mcmc/diagnostics.jl`): `gof()` compares observed degree/ESP/geodesic distributions against simulated networks (accepts `rng`, `n_chains`, `burnin`, `interval`). `mcmc_diagnostics()` computes autocorrelation and effective sample size from MCMLE samples; it throws `ArgumentError` on fits without MCMC samples (MPLE).

**Entry point**: `ergm(net, terms; method=:mple)` and `fit_ergm(net, terms; ...)` are aliases defined at the bottom of `mple.jl`.

**Golden fixtures (statnet ergm)** — `test/fixtures/flomarriage_ergm.toml`, generated by `test/fixtures/r/flomarriage_ergm.R` (`Rscript test/fixtures/r/flomarriage_ergm.R > test/fixtures/flomarriage_ergm.toml`) and loaded with Networks.jl's `load_golden`, which **throws on a fixture with no `[provenance]`**. Issue #8: the older "Golden master vs R ergm" testset carries R's numbers as bare literals in comments — right, but unregenerable and with hand-chosen atols.

The fixture deliberately covers **both kinds of ERGM fit**, because they are different kinds of number and one tolerance for both would be dishonest:

- **dyad-independent** (`edges + nodecov("wealth")`) — the likelihood factorizes, so MPLE **is** the exact MLE and both packages solve the same convex logistic regression. Asserted at **1e-6**; ERGM.jl agrees to 6.6e-12 (coefficients) and 4.8e-8 (SEs). A failure here is a bug, not noise. **Do not loosen it.**
- **dyad-dependent** (`edges + gwesp(0.5, fixed=TRUE)`) — MCMLE on both sides. The R script refits under five further seeds and freezes R's own seed-to-seed sd (`mcmle_seed_sd`, 0.0057/0.0059); the tolerance (0.03) is a multiple of that measured width, and the Julia side compares the mean of five seeded fits. Observed gap 0.0016/0.00034 — smaller than R's disagreement with itself.

**Known behavioural difference from statnet, pinned by that testset:** `mcmle` runs its convergence check *before* applying the first Newton update, so when the MPLE already satisfies the moment condition (here: max t-ratio 0.006) it returns the MPLE unchanged and the point estimate has **zero** seed-to-seed variance. statnet always takes at least one MCMLE step. The estimate is defensible (E_θ[g] = g_obs to within MC error *is* the MLE condition, and it lands inside R's noise) but it is not produced the same way — the test asserts this explicitly so the day it changes is visible.

## Key Dependencies

- **Networks.jl** (`Network` package) - Network data structure with vertex/edge attributes (`get_vertex_attribute`, `get_edge_attribute`, etc.)
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
- The package depends on the local/unregistered `Network` package (not in General registry), resolved via `[sources]` from `../Networks.jl`.
