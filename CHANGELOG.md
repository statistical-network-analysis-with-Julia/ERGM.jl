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
- **`gof` returns a `Networks.GOFResult`** instead of a NamedTuple
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
- **MCMLE, simulation and GOF now reject networks with masked (missing)
  dyads by default.** Holding an unobserved dyad fixed at its stored face
  value — never toggling it, and scoring it as recorded — targets a
  different estimand from both statnet's missing-data MLE and MPLE's
  available-case pseudo-likelihood, so it is no longer the silent default
  (it used to be, behind a warning). `mcmle`, `mh_sample`,
  `sample_networks`, `simulate_ergm`, and `gof` take
  `missing::Symbol=:error` and throw `Networks.require_observed`'s shared
  `ArgumentError` on a masked network. *Migration:* pass
  `missing=:condition_on_face` to opt back in to the 0.1 behaviour
  explicitly (still warned), or use `mple`, which handles the masked dyads
  properly.

### Added

- **Provenanced golden fixture against a real statnet `ergm` fit** (issue #8).
  `test/fixtures/flomarriage_ergm.toml` freezes an ergm 4.12.0 / R 4.6.1 fit of
  the Florentine marriage network, regenerable with
  `Rscript test/fixtures/r/flomarriage_ergm.R > test/fixtures/flomarriage_ergm.toml`
  and loaded through Networks.jl's `load_golden`, which refuses a fixture with no
  provenance. It replaces (does not remove) the R coefficients that lived as bare
  literals in test comments: those cannot be regenerated, carry no record of which
  ergm produced them, and had hand-chosen atols beside them.

  The fixture covers **both kinds of ERGM fit, at different tolerances, because
  they are different kinds of number**:

  - **Dyad-independent** (`edges + nodecov("wealth")`): the likelihood factorizes,
    so MPLE *is* the exact MLE and both packages solve the same convex logistic
    regression. Asserted at **1e-6**. ERGM.jl agrees to **6.6e-12** on the
    coefficients and **4.8e-8** on the standard errors.
  - **Dyad-dependent** (`edges + gwesp(0.5, fixed=TRUE)`): MCMLE on both sides.
    The R script refits under five further seeds and freezes R's own seed-to-seed
    sd (0.0057 / 0.0059); the tolerance (0.03) is ~5x that and ~10% of a fitted
    standard error. ERGM.jl's five-seed mean lands **0.0016 / 0.00034** from R —
    closer than R gets to itself.

  Documented behavioural difference, now pinned by the testset: ERGM.jl's MCMLE
  runs its convergence check *before* the first Newton update, and on this model
  it passes at the MPLE (max t-ratio 0.006), so the returned point estimate **is**
  the MPLE and has zero seed-to-seed variance. statnet always takes at least one
  MCMLE step. The estimate satisfies E_θ[g] = g_obs to within Monte-Carlo error —
  the MLE condition — and lands inside R's noise, so it is defensible; but it is
  not produced the same way, and the test says so.

- **The MPLE parametric bootstrap now runs on the shared
  `Networks.bootstrap_cov`** rather than its own loop. `mple(model;
  se=:bootstrap)` is unchanged in API and semantics — it is the reference
  implementation the count, rank and multilayer MPLEs were rolled out from, and
  factoring its loop into Networks.jl is what let them share it instead of
  copy-pasting it four times (issue #9).

- **The term traits are a public, documented protocol** (`src/terms/traits.jl`;
  ERGMUserterms.jl#1). A term now *declares* what it needs, and formula
  validation acts on the declarations rather than on ERGM's own term types —
  so a term defined in any package participates in exactly the same checks as
  a built-in one:
  - `required_vertex_attributes(term)` / `required_edge_attributes(term)`
    (tuples of `Symbol`, default `()`) — validated against the network at
    `ERGMModel` construction; an absent attribute throws the standard
    `ArgumentError` instead of silently producing an all-zero design column.
    Edge-attribute validation is new.
  - `requires_directed(term)` / `requires_undirected(term)` (default `false`) —
    rejected on an incompatible network, as before.
  - `is_dyad_dependent(term)` — already public, now documented alongside the
    rest of the protocol.
  - `Networks.supports_missing(term)` (default `false`) — a term declares
    `true` iff its statistic honours the missing-dyad mask, i.e. is invariant
    to the face value of a masked dyad. Every built-in term is `false`: they
    count masked dyads at face value, and ERGM's principled treatment lives in
    the estimator (`supports_missing(mple) == true`).

  The private predecessors (`_vertex_attribute`, `_requires_directed`,
  `_requires_undirected`) remain: the two direction traits are now `const`
  aliases of the public generics, so downstream packages that had reached into
  them — TERGM.jl ships `ERGM._requires_directed(::Delrecip) = true` — keep
  working unchanged and keep driving validation. `_vertex_attribute` is a shim
  returning the first required vertex attribute (or `nothing`).

- **Ecosystem missing-data contract honoured** (Networks.jl `supports_missing`
  / `require_observed`). `supports_missing(mple) == true`: MPLE's exclusion
  of masked dyads from the design matrix is the standard available-case
  pseudo-likelihood, a principled treatment. Nothing else in ERGM.jl declares
  support; `ERGMResult` gains a `missing_method::Symbol` field recording what
  actually happened (`:none`, `:available_case`, or `:condition_on_face`),
  and `show` reports the treatment whenever the network carried a mask.
  Full missing-data MCMLE (conditional simulation of the unobserved dyads)
  remains unimplemented — see issue #4.

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
- Missing-data support: dyads masked with `Networks.set_missing_dyad!` are
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
- `using ERGM` now re-exports the Networks.jl public API (constructors,
  attribute setters, `load_dataset`, ...), so one import suffices.
- BenchmarkTools suite (`benchmark/`) with allocation regression tests.

### Changed

- **`compute`, `name` and `compute_all` are now the shared Networks.jl
  generics**, imported by name and extended with ERGM's term methods, rather
  than ERGM's own functions. They are still exported, and `compute(term, net)`
  is unchanged; what changes is identity: `ERGM.compute === REM.compute ===
  Networks.compute`. Previously each model package owned a distinct function of
  the same name, so `using ERGM, REM` left the unqualified verbs *undefined*
  under Julia's conflicting-export rule (REM.jl#3). Downstream term packages
  (ERGMCount, ERGMMulti, ERGMRank, ERGMEgo, ERGMUserterms, TERGM) keep working
  unchanged: `import ERGM: name, compute` resolves to the shared generics.
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
- `show(::ERGMResult)` prints through the shared `Networks.print_coeftable`
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

- **`logistic_derivatives(X, y; offset)` — the shared, allocation-free
  logistic `(ll, grad, hess)` builder** (review finding 15), exported next to
  `newton_fit`. `ERGMMulti`'s MPLE over the within-layer dyads, `TERGM`'s CMPLE
  over the free dyads of the auxiliary networks, and `ERGMRank`'s swap MPLE over
  the (ego, alter-pair) comparisons are all the *same* logistic likelihood, and
  all three carried their own copy of the loop with a per-row `x * x'` outer
  product inside it — a fresh `p×p` matrix on every design row of every Newton
  evaluation. There is now one builder, with the workspaces allocated once and
  the derivatives formed by gemv/gemm over the whole design (`η = Xβ`,
  `∇ = X'r`, `−H = X'WX`). An evaluation allocates only the gradient and Hessian
  it returns: 192-304 bytes, independent of the number of rows (was 649 KB on a
  3120-row ERGMMulti design, 471 KB on a 4200-row TERGM one), and 4-8x faster.
  Pinned by `@allocated` regression tests here and in all three packages.
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
