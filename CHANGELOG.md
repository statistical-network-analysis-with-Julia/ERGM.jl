# Changelog

All notable changes to ERGM.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

Release driven by the 2026-07 expert-panel review: a critical
attribute-dropping copy bug is fixed, the MCMLE loop is brought to statnet
standards, two silently R-divergent term semantics are corrected (breaking),
the missing statnet terms are added, and the change-statistic hot path is
rewritten for O(deg) scaling.

### Breaking

- **Directed `GWESP` default changed from union to `:OTP`.** Previously
  directed GWESP counted either-direction shared partners while emitting
  statnet's OTP label `gwesp.fixed.<decay>` — a silently different model.
  `GWESP(decay; type=...)` now implements the four Hunter/Handcock directed
  shared-partner types `:OTP | :ITP | :OSP | :ISP` plus `:union`, defaulting
  to statnet-compatible `:OTP`; the old either-direction statistic is
  relabeled `gwesp.union.fixed.<decay>`. Directed models refit with 0.2
  produce different coefficients; undirected GWESP is unchanged. *Migration:*
  pass `type=:union` to reproduce 0.1 fits.
- **`NodeMatch(attr; diff=true)` no longer counts mismatches.** `diff=true`
  now means R-compatible differential (per-level) homophily — one
  `nodematch.<attr>.<level>` statistic per requested level, and `diff=true`
  without `level` throws rather than guessing. The old mismatch count moved
  to the new `NodeMismatch(attr)` term. *Migration:* replace
  `NodeMatch(:a; diff=true)` with `NodeMismatch(:a)` if you wanted the 0.1
  mismatch statistic.
- **`NodeFactor` drops the first level by default.** Previously
  `NodeFactor(attr)` produced a single all-levels statistic that was
  collinear with `Edges()` by construction; it now expands to one statistic
  per level with the first (sorted) level as the reference, exactly like R's
  `nodefactor`. *Migration:* pass `base=0` to keep all levels (as separate
  per-level statistics); use `levels=`/`level=` for explicit control.
- **`gof` returns a `Network.GOFResult`** instead of a NamedTuple
  `(results::Dict, n_sim)`. P-values are two-sided Monte-Carlo
  `(1+k)/(N+1)` (never exactly zero), and on directed fits the `:degree`
  panel splits into `:idegree`/`:odegree`. *Migration:* access panels via
  the `GOFResult`/`GOFStatistic` fields or just `show` the result.
- **`mcmc_diagnostics` on an MPLE fit now throws an `ArgumentError`**
  instead of returning an `(error=...,)` NamedTuple. *Migration:* call it
  only on `method=:mcmle` fits.
- **`change_stat` contract is now the state-independent add-direction
  convention** `g(y⁺ᵢⱼ) − g(y⁻ᵢⱼ)`: its value must not depend on whether the
  edge currently exists (previously terms returned a toggle-signed delta).
  This breaks externally written custom terms. *Migration:* drop any
  `has_edge`-based sign flips from custom `change_stat` methods — the
  sampler negates the value for removal proposals (see ERGMUserterms.jl).
- **`mcmle`'s `tol` keyword is deprecated and ignored** (a warning is
  emitted); convergence is now assessed by per-statistic t-ratios
  (`conv_threshold`) plus a Hotelling T² test. `burnin`/`interval` defaults
  now scale with dyad count instead of the fixed 1000/100. *Migration:*
  remove `tol=...`; pass explicit `burnin`/`interval` to reproduce old
  sampling budgets.
- **Minimum Julia version raised to 1.12** (was documented as 1.9+); the
  unused `SNA` dependency was dropped. *Migration:* upgrade Julia.
- **Package UUID regenerated** (placeholder replaced). *Migration:*
  re-resolve environments that recorded the old UUID.

### Added

- Missing statnet terms: `Degree(d)`, `IDegree(d)`, `ODegree(d)` (accept
  vectors/ranges, e.g. `Degree(0:2)`, expanding to one term per degree),
  `GWIDegree(decay)`, `GWODegree(decay)`, `GWDSP(decay; type=...)` (same
  directed types as GWESP), `NodeMix(attr)` (mixing-matrix cells, first cell
  dropped as reference), and `NodeMismatch(attr)`.
- Loud failure on user errors, as in R ergm: model construction validates
  attribute-based terms against the network's vertex attributes (typo'd
  attributes throw an `ArgumentError` listing what exists) and rejects
  intrinsically directed terms (`Mutual()`, ...) on undirected networks.
- Parametric-bootstrap MPLE standard errors: `ergm(...; method=:mple,
  se=:bootstrap, n_boot=...)`; `show` prints a statnet-style
  anticonservative-SE caveat for pseudo-likelihood fits of dyad-dependent
  models.
- Missing-data support: dyads masked with `Network.set_missing_dyad!` are
  excluded from the MPLE design matrix (and `nobs`); MCMC never toggles them
  and `mcmle` warns that it conditions on their face values.
- Public building blocks for downstream packages: `mh_sample(model, θ)`
  (single-chain MH sampler returning sampled statistics/networks; replaces
  ERGMEgo's private-API use), `newton_fit(loglik_grad_hess, θ0)` (shared
  Newton–Raphson with step halving, factored out for the ERGM variants), and
  the `is_dyad_dependent(term)` trait.
- `rng::AbstractRNG` keywords on all sampling and fitting functions
  (`ergm`, `mcmle`, `simulate_ergm`, `sample_networks`, `gof`,
  `mh_sample`); same seed ⇒ identical results independent of thread count.
- StatsAPI accessors: `loglikelihood`, `aic`, `bic`, `nobs`, `dof` join
  `coef`/`stderror`/`vcov`; `vcov` now returns the full covariance matrix
  (was a diagonal reconstruction); `ERGMResult` records `se_type`.
- `mcmc_diagnostics` adds Geyer initial-sequence ESS (`ess_geyer`) and
  Geweke convergence diagnostics (`geweke_z`/`geweke_p`) per statistic.
- `using ERGM` now re-exports the Network.jl public API (constructors,
  attribute setters, `load_dataset`, ...), so one import suffices.
- BenchmarkTools suite (`benchmark/`) with allocation regression tests.

### Changed

- MCMLE overhaul to statnet standards: Hummel step-length control on the
  Newton updates, Hotelling T² convergence test (replacing the unattainable
  raw-count tolerance that made every fit report `converged=false`),
  dyad-scaled burnin/interval defaults, and explicit sampler-collapse /
  singular-covariance detection with warnings instead of silent MPLE
  fallback.
- MCMLE log-likelihood (hence AIC/BIC) is now estimated by a path-sampling
  (bridge) ladder from a dyad-independent reference (`bridge_rungs`
  keyword), replacing the high-variance one-jump importance sampler.
- `coef`/`stderror`/`vcov` are now methods of the StatsAPI generics rather
  than package-local functions, so `using ERGM, StatsBase` (or loading two
  model packages) no longer breaks the shared verb API.
- Directed structural statistics now follow statnet definitions: directed
  `Triangle` (ttriple+ctriple), `Kstar` (out-stars), `TwoPath` (excluding
  mutual returns), `GWDegree`; the GWESP change statistic includes the full
  indirect effect. Undirected `NodeFactor`/`NodeCov` counts are no longer
  halved. Fitted statistic values differ accordingly from 0.1.
- Attribute-based terms are materialized at model construction into typed
  twins that snapshot attributes into dense vectors (names and semantics
  preserved).
- `show(::ERGMResult)` prints through the shared `Network.print_coeftable`
  presentation layer (R-style coefficient table with significance codes).

### Fixed

- **Attribute-preserving network copies (critical).** `_copy_network`
  previously copied only vertices and edges, so every MCMLE chain,
  `simulate_ergm`, and `gof` run evaluated attribute-based terms
  (`NodeMatch`, `NodeCov`, `NodeFactor`, `AbsDiff`, ...) against empty
  covariates — statistics were silently exactly zero and MCMLE degraded to
  MPLE coefficients with NaN SEs behind a misleading singular-covariance
  warning. All copies now go through the attribute-preserving
  `Base.copy(::Network)`, and regression tests pin
  `compute(term, copy(net)) == compute(term, net)` for every term plus an
  MCMLE fit with `NodeMatch`.
- P-values no longer underflow to exactly `0.0`: computed as
  `2·ccdf(Normal(), |z|)` and floored at `floatmin` (accurate to |z|≈38).
- GOF geodesic-distance panel no longer reaches into the private
  `net.graph` field.

### Performance

- Change statistics for `Triangle`, `GWESP`, `GWDSP`, and the ESP GOF use
  sorted neighbor-list intersection — O(deg) per toggle instead of O(n)
  vertex scans (directed Triangle `compute` was O(n³)) — lifting clustered
  MCMC models from ~300 to thousands of nodes.
- Tuple-backed `TermSet` and materialized typed attribute vectors give
  statically dispatched, allocation-free MH hot loops.
- MPLE builds a compressed design matrix over unique change-stat rows.
- Sampling, GOF, bridge rungs, and bootstrap refits run on parallel threads
  (`n_chains` keyword) with deterministic per-chain seeding.

## [0.1.0] - 2026-02-09

Initial release: ERGM terms (structural, nodal, dyadic), MPLE and MCMLE
estimation, MCMC simulation, and goodness-of-fit/MCMC diagnostics.
