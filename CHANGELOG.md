# Changelog

All notable changes to ERGM.jl are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
package adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

First public release of ERGM.jl, a Julia port of R's `ergm` (statnet):
exponential-family random graph models fitted by MPLE and MCMLE, MCMC
simulation, goodness of fit and MCMC diagnostics, with term values, labels,
estimates and standard errors checked against R ergm 4.12 by provenanced
fixtures. The changes below are relative to 0.1.0, which was not publicly
released.

**Dependency renamed:** the foundation package is now `NetworkCore` (developed as `Networks`); write `using NetworkCore` where code said `using Networks`. Types and functions keep their names.

### Highlights

- **R's default estimator**: `ergm`/`fit_ergm` take `method=:auto`, which
  fits the MPLE (the exact MLE) when no term is dyad-dependent and the
  MCMLE otherwise, as R's `ergm()` does.
- **statnet's MCMLE**: every iteration takes a Monte-Carlo Newton step
  (Hummel step length), and the default stopping rule is R ergm 4's
  `termination=:confidence` equivalence test, with R's sample-size boost
  when it fails. Faux Mesa High `edges + nodematch + gwesp` (and
  `+ gwdegree`) models converge at the defaults and match R within R's
  seed spread.
- **R ergm 4's sampler design in `mcmle`**: by default `mcmle` proposes
  with R's `SPDyad` (tie/no-tie mixed with a shared-partner-focused
  proposal, exact Hastings ratio; `proposal=:spdyad`) and samples
  ESS-adaptively to R's `MCMLE.effectiveSize = 64` (`effective_size=64`);
  `proposal=:tnt, effective_size=nothing` gives fixed-size tie/no-tie
  sampling. The other samplers use statnet's tie/no-tie proposal
  (`proposal=:tnt`); `proposal=:random` keeps the uniform random-dyad
  toggle.
- **Honest standard errors**: MCMLE standard errors include the
  Monte-Carlo component (R's "MCMC %"). A dyad-dependent MPLE no longer
  reports naive Wald z/p-values by default; `se=:bootstrap` gives
  calibrated ones.
- **Missing-data maximum likelihood**: `mcmle(model; missing=:mle)` fits a
  network with masked dyads as R `ergm()` does for NA ties (Handcock & Gile
  2010); `mple` uses the available-case pseudo-likelihood.
- **R-compatible terms and labels**: statnet's directed GWESP/GWDSP types,
  `nodefactor`/`nodemix`/`degree` expansion, R's coefficient names, and new
  terms (`ostar`/`istar`, `concurrent`, `gwnsp`, `degrange`, `meandeg`,
  `density`, `sender`/`receiver`, `triadcensus`, …).
- **Loud failures instead of silent mis-fits**: two-mode networks,
  self-loops, NA attributes, wrong-direction terms and `constraints=` are
  refused, with messages that name the cause.
- **R's handling of boundary statistics**: a statistic at the boundary of
  its attainable range — a `nodematch(diff=TRUE)` level with no
  within-level tie, a singleton level's `nodemix` cell, `triangle` on a
  network with no two-path — is fixed at `∓Inf` with R's message and the
  rest is estimated, by the MPLE and the MCMLE alike (R's `drop=TRUE`), so
  the statnet tutorial's Goodreau model fits as typed in R.
- **Fast hot paths**: O(degree) change statistics, allocation-free MH
  steps at any number of statistics, multithreaded chains with
  thread-count-independent results.

### Added

- `drop=true` on `mple`, `mcmle` and `fit_ergm`: R's `control.ergm(drop=)`.
  The default fixes a statistic at the boundary of its attainable range at
  `∓Inf`, with R's message, and estimates the rest; `drop=false` refuses
  such a model with an `ArgumentError`.
- `coefnames(fit)` (StatsAPI) returns the coefficient labels.
- A "Coming from R" page in the documentation: R's `ergm` calls, controls
  and terms with their ERGM.jl counterparts.

**Terms**

- `NodeMatch(attr; diff=true)` without `level=` expands over the levels,
  as R's `nodematch(diff=TRUE)` does, with R's `levels=` keyword; the
  statnet tutorial's Goodreau model is typed as in R.
- `Degree`, `IDegree`, `ODegree` (a single degree or a vector/range:
  `Degree(0:2)` is one term that expands to `degree0, degree1, degree2`
  when the model is built), `GWIDegree`, `GWODegree`, `GWDSP`, `NodeMix`
  (first cell dropped as reference) and `NodeMismatch`.
- `OStar(k)` / `IStar(k)` (R's `ostar`/`istar`), directed only.
- `Offset(term, coef)` — statnet's `offset()`: the coefficient is fixed,
  not estimated, in MPLE, MCMLE, the bootstrap, simulation, GOF and the
  log-likelihood; reported as R does (fixed value, SE 0, excluded from
  `dof`/AIC/BIC). Finite offsets and `±Inf` constraints (ties forbidden or
  forced, also on dyad-dependent statistics), including with
  `missing=:mle`; checked against R on faux.mesa.high and flomarriage
  (`test/fixtures/offset_ergm.toml`).
- **Curved terms**: `GWESP(decay; fixed=false)` and
  `GWDegree(decay; fixed=false)` (types `CurvedGWESP`, `CurvedGWDegree`)
  estimate the decay by the curved exponential-family MCMLE; coefficients
  are named as in R (`gwesp`, `gwesp.decay`). Checked against R ergm on
  faux.mesa.high (`test/fixtures/curved_ergm.toml`).
- `TransitiveTies` and `CyclicalTies` (statnet `transitiveties` /
  `cyclicalties`), checked against ergm 4.12 `summary()`.
- `ESP(k; type=)` (statnet `esp(d)` / `desp(d, type=)`): the number of ties
  with exactly `k` shared partners, for undirected networks and the directed
  types `:OTP`, `:ITP`, `:OSP`, `:ISP`; `ESP(0:2)` expands like
  `Degree(0:2)`, with R's labels (`esp1`, `esp.OTP1`). Values, labels and
  an MPLE are checked against ergm 4.12, change statistics against brute
  force; a count of 0 is fixed at `-Inf` (R's `minval`).
- `Concurrent`, `GWNSP`, `DegRange`/`IDegRange`/`ODegRange`, `MeanDeg`,
  `Density`, `Sender`/`Receiver` (one statistic per vertex but the first)
  and `ERGM.TriadCensus` (public but not exported, because Siena.jl
  exports a GOF `TriadCensus`), with R's labels; values are checked
  against ergm 4.12 `summary()` and change statistics against brute force.
- `GWESP(decay; type=)` and `GWDSP(decay; type=)` implement the directed
  shared-partner types `:OTP` (default), `:ITP`, `:OSP`, `:ISP` and
  `:union`, each summed over ordered dyads as R's `dgwesp`/`dgwdsp` do.
- Decay 0 for every geometrically weighted term (`gwesp(0, fixed=TRUE)`,
  statnet's most common specification); a decay may be any non-negative
  `Real`.
- A public term-trait protocol: `required_vertex_attributes`,
  `required_edge_attributes`, `requires_directed`, `requires_undirected`,
  `is_dyad_dependent` and `NetworkCore.supports_missing(term)`. Formula
  validation reads the traits, so a user-defined term gets the same checks
  as a built-in one. `has_dyad_dependent(model)` is exported.

**Estimation**

- `mcmle` — and so the default fit of a dyad-dependent formula — fixes a
  statistic at the boundary of its attainable range at `∓Inf` and estimates
  the rest, with R's message, as R's `ergm()` does; it used to refuse the
  model and ask for `Offset(term, -Inf)`. The fixed statistic is held at its
  bound by the sampler; `show` and `approximations` say which coefficients
  were fixed, and `dof`/AIC/BIC count the estimated ones.
- The internal `ERGM._separated` and the one-argument `_warn_separated`,
  which had no caller left, are removed.

- `mple(model; se=:bootstrap, n_boot=)`: parametric-bootstrap standard
  errors. Replicates without a finite MPLE are excluded with one warning
  (and recorded); every refit is kept in `fit.boot_replicates`.
- `mple` takes `maxiter=100` and `tol=1e-8`.
- `mcmle(model; missing=:mle)`: missing-data MLE with a free and a
  constrained chain per iteration (new `obs_burnin`/`obs_interval`
  keywords, R's `obs.MCMC.*`). Validated against R on flomarriage with four
  NA dyads.
- `mcmle` keywords: `n_chains` (independent chains seeded from `rng`),
  `init` (starting coefficients, so `mcmle(model; init=coef(fit))` resumes
  an unconverged fit), `bridge_rungs=0` (skip the log-likelihood; `loglik`,
  AIC and BIC are `NaN`, all else identical), `termination`,
  `conv_precision`, `conv_confidence`, `max_n_samples` and `proposal`.
- `ERGMResult` fields `vcov_fisher` (inverse Fisher information),
  `mcmc_convergence` (the convergence report on the final sample),
  `termination` (rule, p-value, precision, confidence, sample size),
  `chain_lengths`, `boot_replicates`, `se_type` and `missing_method`
  (`:none`, `:available_case`, `:mle` or `:condition_on_face`); `mcmc_se(fit)`
  returns the Monte-Carlo part of the standard errors.
- Full StatsAPI surface on `ERGMResult`: `coef`, `stderror`, `vcov` (now
  the full covariance matrix), `confint`, `coeftable`, `loglikelihood`,
  `aic`, `bic`, `nobs`, `dof`. `show` prints through `coeftable`, so the two
  always agree.
- Result metadata: `is_exact`, `se_method`, `objective` and
  `NetworkCore.approximations(fit)`. `is_exact` is `false` for an unconverged
  or `±Inf` MPLE fit; every approximation `show` warns about is listed.
- `show(fit)` prints R's "MCMC %" line (`100·(se − se_fisher)/se`, R's
  `summary.ergm` definition) for MCMLE fits, a `Termination:` line under
  `Converged: true`, and the non-convergence advice under
  `Converged: false`.

**Sampling and diagnostics**

- `mh_sample(model, θ)`: a single-chain sampler returning statistics and
  networks, with `toggleable=:free | :all | :masked`.
- `mh_toggle!`: the exported Metropolis–Hastings toggle kernel shared by the
  ERGM variants (proposal, change statistics and state update are
  callables). Its optional `hastings=` keyword takes the log Hastings ratio
  of an asymmetric proposal; existing callers are unaffected.
- `proposal=` keyword on `mh_sample`, `sample_networks`, `simulate_ergm`,
  `gof`, `mcmle` and `mple` (for its bootstrap): `:tnt`, `:spdyad` — R
  ergm's shared-partner-focused proposal (`SPDyad`), about 6× TNT's
  effective sample size per draw on faux.mesa.high — or `:random`.
  `mcmle` defaults to `:spdyad`, the other samplers to `:tnt`.
- `mcmle(...; effective_size=64)`, the default: ESS-adaptive sampling (R's
  `MCMLE.effectiveSize`) — chains continue across iterations and each
  sample is extended until its effective sample size reaches the target.
  With `proposal=:spdyad` it is R ergm 4's default design;
  `proposal=:tnt, effective_size=nothing` gives fixed-size TNT sampling.
- **`ERGM.Extension`, the extension API**: the stable, semver-covered API
  for packages that build estimators or term families on ERGM.jl (the six
  ERGM-family packages are written against it). `using ERGM.Extension`
  brings in 18 underscore-free names: the formula pipeline
  (`collect_terms`, `validate_formula`, `materialize`, `expand_terms`,
  `require_supported_network`, `n_observed_dyads`), the boundary of the
  parameter space (`attainable_range`, `extreme_statistics`,
  `boundary_columns`, `warn_boundary`), the pseudo-likelihood fitter
  `mple_fit_design`, and the Monte-Carlo MLE drivers (`mcmc_defaults`,
  `mcmle_solve`, `confidence_test`, `mcmle_covariance`, `ess_sample`,
  `mcmle_sampler`, `bridge_integrate`). `ERGM` exports none of them; `Extension` is a
  `public` name of `ERGM`. A docs page, "Extension API", states the
  contract and the boundary convention every ERGM-family estimator follows.
- `ERGM.Extension.attainable_range(term, net)` (R ergm's `minval`/`maxval`)
  is a documented generic: a package declares the range of its own term or
  network type by adding a method, and `extreme_statistics` dispatches
  through it. ERGM.jl's own methods take an `AbstractNetwork`, so they never
  claim a bound on another package's network type.
- `ERGM.Extension.mple_fit_design` takes `note=` (a `NamedTuple` with any of
  `boundary`, `not_varying`, `linear_dependence`, `separation`) to replace
  R ergm's closing sentence of each warning, and `noun=` for the rows the
  fit runs on. It returns its separation `verdict` (NetworkCore's
  `SeparationVerdict` on the design actually fitted), the `boundary` and
  `aliased` columns, the `fitted` columns and the `kept_rows`, so a caller
  never runs the separation check a second time.
- `ERGM.Extension.require_supported_network(net)`: the refusals `ERGMModel`
  gives a two-mode network or one with a self-loop, in ERGM.jl's words.
- `ERGM.Extension.mcmle_sampler(model; effective_size=64, proposal=:spdyad,
  …)`: `mcmle`'s own sampler (R ergm 4's SPDyad proposal and ESS-adaptive
  chains continued between iterations, or fixed-size samples with
  `effective_size=nothing`) as the `draw`/`resize` pair `mcmle_solve`
  takes, so an estimator that solves moment equations on an `ERGMModel`
  (ERGMEgo's moment matching) samples as `mcmle` does. `mcmle` and the
  curved MCMLE now run on it; their draws are unchanged.
- `MCMCDiagnostics`, the result type of `mcmc_diagnostics`: per-term lag-1
  autocorrelation, lag-1 and Geyer ESS and the Geweke test, computed within
  each chain and combined.
- GOF thinning is effective-sample-size aware at the default interval: if
  the simulated networks' model statistics have ESS < `n_sim/2` they are
  redrawn at twice their measured autocorrelation time (at most 64× the
  default interval); ESS < `n_sim/4` is warned about.
- `rng::AbstractRNG` keyword on every sampling and fitting function; the
  same seed gives the same result at any thread count.

**API and tooling**

- `using ERGM` re-exports NetworkCore.jl's user-facing API (including
  `Network`, `BipartiteNetwork`, `degree`, `src`/`dst`, `missing_policies`,
  `z_pvalues`, `CoefficientTable`). The golden-fixture harness,
  `bootstrap_cov`, `check_se`, `check_statsapi` and the module name
  `NetworkCore` are not re-exported: qualify them or add `using NetworkCore`.
- `NetworkCore.missing_policies` is declared for `mple`, `mcmle`, the samplers
  and `gof`.
- Clearer call-site errors in `fit_ergm`/`ergm`: a single term needs no
  brackets, swapped arguments, a non-term in the term list (`Edges` instead
  of `Edges()`) and an unknown `method=` are explained, and a `Vector{Any}`
  of terms is accepted.
- `ERGM.mcmc_convergence`, `ERGM.MCMLEConvergence` and
  `ERGM.resolve_method` are public (not exported); the building blocks the
  ERGM variant packages call are the extension API, `ERGM.Extension`.
- Every exported and public name has a docstring with a runnable example,
  executed by the test suite.
- Provenanced fixtures against R 4.6.1 / ergm 4.12.0, regenerable from the
  scripts in `test/fixtures/r/`: flomarriage MPLE and MCMLE fits, term
  values and names on flomarriage and samplike, drop-semantics and
  separation designs, GOF panels, the missing-data fit and the Faux Mesa
  High MCMLE (`test/fixtures/mcmle_ergm.toml`).
- Aqua.jl quality checks; a BenchmarkTools suite with allocation
  regression tests, run in CI.
- README: the ergm 4 citation (Krivitsky, Hunter, Morris & Klumb 2023, JSS
  105(6)) and a request to cite ergm and the methods too.

### Breaking

Relative to the development versions (0.2.0 is the first release):

- **No underscore name of ERGM.jl is `public` any more.** The 16 that were
  declared `public` for the variant packages are replaced by
  `ERGM.Extension` names, with no deprecation aliases: `_collect_terms`,
  `_validate_formula`, `_materialize`, `_expand_terms`, `_mcmc_defaults`,
  `_confidence_test`, `_mcmle_solve`, `_bridge_integrate`, `_ess_sample`,
  `_warn_boundary`, `_mple_fit_design` and `_mcmle_covariance` keep their
  name without the underscore; `_boundary_columns_iterated` is
  `boundary_columns`, `_n_dyads` is `n_observed_dyads` (REM.jl exports an
  unrelated `n_dyads`), and `_refuse_two_mode` and `_refuse_self_loops` are
  merged into `require_supported_network`. The internal `_attainable_range`
  and `_extreme_statistics` are now `attainable_range` and
  `extreme_statistics`. A method a package added to `ERGM._materialize` is
  now a method of `ERGM.Extension.materialize`. The name map is on the
  "Extension API" docs page.
- The dead helpers `_separated` and `_warn_separated` are deleted (no
  caller; use NetworkCore's `logistic_separation`/`warn_separation`, or the
  `verdict` `mple_fit_design` returns).

### Changed

**Estimation**

- `fit_ergm`/`ergm` default to `method=:auto` (R's rule, the `public`
  `ERGM.resolve_method`): the MPLE for a dyad-independent formula, where it
  is the exact MLE, and the MCMLE for a dyad-dependent one. Before, every
  formula was fitted by MPLE unless `method=:mcmle` was passed, so a
  dyad-dependent fit reported pseudo-likelihood point estimates where R
  reports the MLE (Florentine `edges + triangle`: triangle 0.221 against
  R's 0.164). Migration: pass `method=:mple` for the pseudo-likelihood
  fit. A keyword the chosen estimator does not take (e.g. `se=:bootstrap`,
  an MPLE keyword, on a formula `:auto` fits by MCMLE) is an
  `ArgumentError` naming the estimator that takes it.
- `show(::ERGMResult)` says what the estimator is for the formula
  ("maximum pseudo-likelihood, which is the likelihood: the formula is
  dyad-independent", or an approximation under dyadic dependence).
- The non-convergence warning and the caveat in `show`/`approximations`
  quote the stopping rule that decided (`termination`: its p-value and the
  step length), not the classical t-ratios and Hotelling p of
  `fit.mcmc_convergence`, which for a converged fit describe the sample
  drawn before the final Newton step.
- MCMLE was rebuilt: Hummel step-length control, a Monte-Carlo Newton step
  at every iteration including the first, dyad-scaled sampler defaults,
  and warnings (not a silent MPLE fallback) on sampler collapse or a
  singular covariance.
- The default stopping rule is `termination=:confidence`: converged when the
  99 % Monte-Carlo confidence region of the estimating equation lies inside
  a tolerance region of 0.1 of the statistics' variance; on failure the
  sample grows up to `max_n_samples` (default `16 × n_samples`). The
  max |t| < 0.1 + Hotelling T² rule is `termination=:hotelling`.
- `mcmle` samples as R ergm 4 does by default: `proposal=:spdyad` and
  `effective_size=64`. Seeded fits therefore differ from fits made with
  fixed-size TNT sampling by Monte-Carlo error; pass
  `proposal=:tnt, effective_size=nothing` to reproduce those.
- `converged=true` is the verdict of the final sample the standard errors
  come from.
- `mcmle`'s `maxiter` default is 60 (R's `MCMLE.maxit`). The development
  spellings `max_iter` and the ignored `tol` were removed without a
  deprecation period (see the rename table in the docs' Estimation API
  page).
- MCMLE standard errors include the Monte-Carlo error of the sample mean
  (Hunter & Handcock 2006); the inverse Fisher information alone is
  `fit.vcov_fisher`.
- MCMLE log-likelihood (hence AIC/BIC) is estimated by path sampling over
  `bridge_rungs` rungs from a dyad-independent reference, integrated by
  Simpson's rule; an odd `bridge_rungs` is raised to the next even number.
- MPLE of a dyad-dependent formula: with the default `se=nothing` the z
  values and p-values are `NaN` and `confint` throws an `ArgumentError`
  (naive pseudo-likelihood SEs under-cover: 95 % Wald coverage 0.71–0.85 in
  simulation); `show` explains and `approximations` records it.
  Migration: `se=:hessian` restores R's naive Wald table with the caveat;
  `se=:bootstrap` reports calibrated z, p and intervals. Dyad-independent
  fits are unchanged.
- MPLE is Newton–Raphson on the shared `NetworkCore.newton_fit` (re-exported,
  with `logistic_derivatives`); Optim.jl is no longer a dependency. Results
  agree with R to ~1e-12.
- Boundary statistics follow R's `drop=TRUE`: `mple` warns R's sentence,
  fixes the coefficient at `-Inf`/`+Inf` (SE 0, p 0), fits the rest on the
  dyads the term does not touch, and `dof`/AIC/BIC use R's `logLik` df and
  nobs. A design separated by a combination of statistics is warned about
  ("the MPLE does not exist") and returned with `converged=false`. `mcmle`
  refuses both before sampling. Migration: `isfinite.(coef(fit))` separates
  the dropped coefficients.
- Masked (missing) dyads: `mcmle`, `mh_sample`, `sample_networks`,
  `simulate_ergm` and `gof` take `missing=:error` by default and refuse a
  masked network, naming the accepted opt-ins. Migration: `missing=:mle`
  (for `mcmle`) or the warned `missing=:condition_on_face` restores
  estimation; `mple` handles masked dyads without a keyword.
- Sampler `burnin`/`interval` default to `20 × n_dyads` and
  `max(100, n_dyads ÷ 10)` everywhere. Migration: pass
  `burnin=10000, interval=1000` for the old simulation budget.

**Terms and labels**

- `change_stat` follows the state-independent add-direction convention
  `g(y⁺ᵢⱼ) − g(y⁻ᵢⱼ)`. Migration: remove `has_edge`-based sign flips from
  custom terms.
- Directed `GWESP` defaults to `type=:OTP` (statnet). Migration:
  `type=:union` reproduces 0.1's either-direction statistic.
- Directed `GWESP`/`GWDSP` coefficients are labelled
  `gwesp.OTP.fixed.<d>`, as R does; undirected ones `gwesp.fixed.<d>` for
  every type.
- Integer decays print without a decimal (`gwesp.fixed.1`), `GWDegree` is
  labelled `gwdeg.fixed.<d>`, and `AbsDiff(attr; pow)` with `pow ≠ 1` is
  `absdiff<pow>.<attr>`. Migration: index coefficients by R's names.
- `Kstar` and `GWDegree` are undirected-only, as in R. Migration: on a
  directed network use `OStar`/`IStar` and `GWODegree`/`GWIDegree`.
- `NodeMatch(attr; diff=true)` is R's per-level homophily: it expands into
  one statistic per attribute level (every level, as R's default; `levels=`
  selects some, `level=` builds one), under R's labels. Migration: the 0.1
  mismatch count is `NodeMismatch(attr)`.
- `NodeFactor` expands to one statistic per level with the first level
  dropped. Migration: `base=0` keeps all levels.
- Directed `Triangle` (ttriple + ctriple), `TwoPath` and `GWDegree`, and
  undirected `NodeFactor`/`NodeCov` counts (no longer halved), follow
  statnet; fitted values differ from 0.1.
- `compute`, `name` and `compute_all` are the shared NetworkCore.jl generics,
  and `coef`/`stderror`/`vcov` the StatsAPI ones, so `using ERGM, REM` or
  `using ERGM, StatsBase` no longer leaves them undefined.

**Validation and refusals**

- `ERGMModel`/`fit_ergm` refuse, with an `ArgumentError` naming the cause:
  a two-mode network; a network containing a self-loop (Migration:
  `rem_edge!(net, v, v)`); an attribute term whose attribute is missing on
  some vertex, or absent or misspelled (statnet refuses NA attributes too);
  and a non-empty `constraints=` on `fit_ergm`/`ergm` or `ERGMFormula`.
- `summary_stats(net, terms; missing=:error)` refuses a masked network
  (Migration: `missing=:face`), validates the formula as `ERGMModel` does,
  expands multi-level terms and uses the model's labels.
- An `EdgeCov` matrix of the wrong size, an unknown `NodeCov(transform=)`
  and an unknown `gof(stats=)` symbol are `ArgumentError`s raised before
  any work.

**Results and API**

- `gof` returns a `NetworkCore.GOFResult`; Monte-Carlo p-values are
  `(1+k)/(N+1)`, and directed fits get `:idegree`/`:odegree` panels.
- `mcmc_diagnostics` returns an `MCMCDiagnostics` struct (same field names)
  and throws on an MPLE fit. Migration: read fields, not positions.
- `ERGMModel{T,D}`/`ERGMResult{T,D}` carry directedness as a type parameter;
  the `directed` field is gone. Migration: `is_directed(model)`.
- `sample_networks`/`simulate_ergm` return a `Vector{Network{T,D}}`.
- `ergm` is a `const` alias of `fit_ergm`.
- `show(::ERGMModel)` and `show(::ERGMFormula)` print the formula, not a
  struct dump.
- Errors raised inside threaded chains reach the caller as the original
  exception, not wrapped in `TaskFailedException` (through NetworkCore's
  shared `spawn_all`).
- Bootstrap standard-error notes (warning, `show`, `approximations`) say
  that excluding the failed refits biases the standard errors downward.
- Removed private aliases: `_requires_directed`, `_requires_undirected`,
  `_vertex_attribute`, `_has_dyad_dependent`, `_z_pvalues`. Migration:
  `requires_directed`, `requires_undirected`, `required_vertex_attributes`,
  `has_dyad_dependent`, `NetworkCore.z_pvalues`.
- `ERGM._hummel_step`, `ERGM._bridge_quadrature`, `ERGM._bridge_logZ` and
  `ERGM._warn_degenerate_stats` are internal (no variant calls them; the
  drivers `ERGM.Extension.mcmle_solve` and `ERGM.Extension.bridge_integrate`
  are the API).
- Julia 1.12 or later is required. The package UUID was regenerated
  (re-resolve environments). SNA, StatsBase, SparseArrays and Optim are no
  longer dependencies; Printf is.

### Fixed

- A statistic whose change statistics are all zero on the observed network
  — a `NodeMix` cell of a singleton level, `Triangle` on a network with no
  two-path, `Degree(1)` when every degree is at least 3 — made the MPLE
  return 0 for every coefficient (Newton stopped at its start on the
  singular information), and the default MCMLE could report a finite
  coefficient with `Converged: true` where R fixes it at `-Inf`. The fits
  now compare each observed statistic with its term's attainable range, as
  R's `ergm.checkextreme.model` does, and fix it at `∓Inf` when it sits at
  an end. A statistic that does not vary without reaching a bound, or is a
  linear combination of the statistics before it, is reported as `NaN`
  with R's warning (R reports `NA`) and the rest is estimated; `simulate_ergm`
  and `gof` simulate such a dyad-independent statistic at 0. Pinned by brute
  force and by the provenanced `boundary_ergm.toml`.
- The simulation guide computed the clustering coefficient on `net.graph`,
  the storage field (a symmetric digraph for an undirected network), which
  gave 0.039 instead of 0.191 on Florentine marriages; it now computes
  transitivity from the network.

- Separation of the pseudo-likelihood by a combination of statistics is
  decided by the shared `NetworkCore.logistic_separation` verdict, an exact
  linear programme as R's `mple.existence`, instead of two signatures on
  the Newton iteration (which missed a design approached slowly). The fit
  follows the ecosystem's separation policy: the shared warning names the
  separating terms (also in `fit.separated_terms`, `show` and
  `approximations`), `converged == false`, and z values, p-values and
  `confint` are `NaN`; `se=:bootstrap` is refused on it.
- ERGM's samplers silently changed the model of a user term that reads an
  edge attribute from the network at evaluation time: `rem_edge!` deletes
  the tie's attributes, so such a term decayed to its default under MCMC.
  Every sampler entry point now probes the terms defined outside ERGM.jl
  (statistic and change statistics with and without the network's edge
  attributes) and refuses a live reader with an `ArgumentError` naming the
  snapshot remedy.
- `summary_stats` refuses a network with a self-loop, as model
  construction does (the off-diagonal statistics differ from R's on a
  looped network: `kstar(2)` of a looped triangle is 7 in R).
- `gof` on a `Network{Int32}` with an unreachable pair exhausted memory:
  the distance panel compared `Graphs.gdistances`' unreachable marker
  (`typemax(Int32)` there) against `typemax(Int)`, counted every
  unreachable pair as a distance of 2³¹ − 1 and sized the histogram to
  match.
- Network copies dropped vertex attributes, so every MCMLE chain,
  simulation and GOF run saw attribute terms (`NodeMatch`, `NodeCov`, …)
  as zero, and MCMLE degraded to the MPLE with NaN standard errors.
- MCMLE could return the MPLE unchanged (convergence was tested before the
  first step), biasing estimates toward the MPLE.
- MCMLE fits reported `converged=false` on every model (an unattainable
  raw-count tolerance).
- A perfectly separated MPLE "converged" silently to a huge coefficient
  and standard error; it is now dropped or flagged as above.
- The GOF `:distance` panel double-counted undirected dyads and omitted
  unreachable pairs; it now matches R's `obs.dist`, with a final `"Inf"`
  level.
- `mple` with a large finite `Offset` could run to a flat asymptote and
  report "the MPLE does not exist" although a finite maximum exists; the
  Newton start is now reached by continuation in the offset.
- A user-defined term that declares `is_dyad_dependent == false` is probed
  at model construction (its change statistic must not depend on other
  ties) and refused with an `ArgumentError` when the declaration is false;
  before, such a fit was reported as exact with a wrong log-likelihood.
- The exact dyad-independent log-normaliser now includes the empty
  network's statistics, θ'g(∅): MCMLE log-likelihood/AIC/BIC were off by
  that amount for user terms nonzero on the empty network.
- A user term defining `change_stat(::MyTerm, net, i, j)` without type
  annotations raised a `MethodError` inside `fit_ergm`; the fallbacks for an
  undefined `compute`/`change_stat` now throw an `ArgumentError` naming the
  method to define.
- Terms compare structurally (`==`/`hash` from their fields):
  `NodeFactor(:g) == NodeFactor(:g)` was `false`.
- `fit_ergm(...; constraints=...)` raised a bare `MethodError`.
- P-values no longer underflow to exactly 0 (floored at `floatmin`).
- `AbsDiff(attr; pow=2)` threw a `TypeError` (any `Real` is accepted).
- MCMLE and bridge diagnostics printed values such as
  `6.969999999999999e-32` (now `6.97e-32`).

### Performance

- Change statistics for `Triangle`, `GWESP`, `GWDSP` and the GOF ESP panel
  are O(degree) per toggle (directed `Triangle` `compute` was O(n³)), so
  clustered models run on thousands of nodes.
- MH steps and whole-model statistic fills are allocation-free at any
  number of statistics; attribute terms read dense snapshots taken at
  model construction.
- MPLE builds a compressed design over unique change-statistic rows with
  O(unique rows) allocation and runs 2–4× faster on the shared
  allocation-free logistic kernel.
- Sampling, GOF, bridge rungs and bootstrap refits run on threads with
  deterministic per-chain seeding.
- A PrecompileTools workload covers fitting, simulation, GOF and display,
  cutting first-call latency.

### Known limitations

What a user of R `ergm` will not find in 0.2.0, and what happens instead
(also in the README's "Not implemented" section):

- **Absent term families**: `nodeicov`/`nodeocov`/`nodeifactor`/
  `nodeofactor`; `ttriple`/`ctriple`/`asymmetric`/`transitive`/
  `intransitive` (and the `attr=` argument of `transitiveties`/
  `cyclicalties`); the `dsp`/`nsp` count
  terms (and the `RTP` type of `esp`/`desp`); `isolates` (only as `Degree(0)`), `balance`, `cycle`,
  `smalldiff`, `dyadcov`, `hamming`, `concurrentties`, `nodematch(keep=)`,
  `nodemix(levels2=)` and multi-range `degrange(from=c(...))` (use one
  `DegRange` per range). There is no term to call, so nothing is silently
  mis-fit.
- **Two-mode (bipartite) networks and terms** (`b1degree`, `b2degree`, …):
  `ERGMModel`/`fit_ergm` throw an `ArgumentError`.
- **Curved terms beyond `gwesp` and `gwdegree`**: `GWESP(decay; fixed=false)`
  and `GWDegree(decay; fixed=false)` estimate their decay (by `mcmle`); the
  other geometrically weighted terms (`gwdsp`, `gwnsp`, `gwidegree`,
  `gwodegree`) take a fixed decay only, and a curved term cannot be combined
  with an offset or with `missing=:mle`. A decay the data do not identify
  (its term's coefficient near 0) ends unconverged, with a warning — R's
  curved MCMLE is fragile in the same cases.
- **`constraints=`**: a non-empty value throws an `ArgumentError`
  (`ERGM.ConstraintTerm` is an unexported placeholder).
- **Two corners of `offset()`**: an observed network that violates an
  infinite offset is refused with an `ArgumentError` (R fits it), and an
  `mcmle` fit with an infinite offset on a dyad-dependent statistic reports
  no log-likelihood/AIC/BIC.
- **R's sampler defaults outside `mcmle`**: `mcmle` samples as R does
  (`SPDyad`, ESS-adaptive to 64), but `simulate_ergm`, `gof`, `mh_sample`,
  `sample_networks` and the MPLE bootstrap default to plain TNT where R's
  `simulate` uses `SPDyad` (both are exact; pass `proposal=:spdyad`). The
  shared-partner proposal handles outgoing two-paths only (R's default type
  for directed networks); R's adaptive burn-in detection and the
  `BDStratTNT` family of stratified proposals are not implemented.
- **Attribute terms with an NA value**: refused at model construction,
  naming the vertices, as statnet refuses; there is no zero-fill.
- **Self-loops**: a network containing a loop is refused (R warns and
  fits); a `loops=true` network with no loop is modelled as loop-free.
- **`drop=FALSE`**: a boundary statistic is fixed at `∓Inf` by `mple` and
  `mcmle` alike, as R's default `drop=TRUE` does; `drop=false` refuses such
  a model rather than fitting R's `drop=FALSE` model. A curved model
  (`GWESP(d; fixed=false)`) with a boundary statistic is refused. A
  `GWESP`/`GWDSP`/`GWNSP` statistic at 0 is fixed at `-Inf` (R declares no
  bound for them and fits a finite, unidentified coefficient). Under
  `mcmle`, a dyad-dependent statistic that does not vary at the observed
  network is refused unless `init=` is given (R starts it at 0).
- **User terms that read an edge attribute live are refused by the
  samplers** (`mcmle`, `simulate_ergm`, `gof`, `sample_networks`,
  `mh_sample`, the MPLE bootstrap): `rem_edge!` deletes a tie's attributes,
  so the chain would target a different model. Snapshot the attribute in
  `ERGM.Extension.materialize` or pass a matrix (`EdgeCov`).
- **Simulation and GOF under missing data**: only the warned
  `missing=:condition_on_face` is offered (`missing=:mle` is refused with
  an explanation); estimation under missing data is implemented.

## [0.1.0] - 2026-02-09

Initial development version (not publicly released): ERGM terms (structural, nodal, dyadic), MPLE and MCMLE
estimation, MCMC simulation, and goodness-of-fit/MCMC diagnostics.
