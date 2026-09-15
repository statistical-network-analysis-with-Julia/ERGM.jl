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

- **`mcmc_diagnostics` returns an `MCMCDiagnostics` struct, computed chain
  by chain** (panel 2026-09 round 3, minor). It returned a NamedTuple that
  printed as 16-digit floats and treated the `n_chains > 1` sample as one
  chain — lag-1 autocorrelation, Geyer ESS and the Geweke test (first 10 %
  vs last 50 %) ran across the chain seams, so Geweke compared chain 1's
  start with the tail of the last chain and the ESS disagreed with the
  MCMLE's own chain-aware `n_eff`. The struct has the same field names
  (`d.ess_geyer` etc. keep working) plus `chain_lengths`, every quantity
  is computed within each chain and combined (ESSs summed — the Geyer
  column is now exactly `fit.mcmc_convergence.n_eff`'s estimator — lag-1
  AC length-weighted, Geweke reported for the chain of largest |z|), and
  `show` prints one row per term (lag-1 AC, ESS (lag-1), ESS (Geyer),
  Geweke z, Geweke p) under a header naming the draws and chains. Code
  that destructured the NamedTuple by position must use the fields.
- **`ERGMResult` has two new fields, `chain_lengths::Vector{Int}` and
  `boot_replicates::Union{Nothing,Matrix{Float64}}`** (round 3): the
  per-chain row counts of `mcmc_samples` (empty for MPLE) and, for
  `se=:bootstrap`, the `n_boot × p` matrix of refitted coefficients (NaN
  rows for the excluded replicates; `nothing` otherwise). Positional
  construction of an `ERGMResult` must pass them.
- **`is_exact(fit)` is `false` for an unconverged MPLE fit and for one
  with a `±Inf` coefficient** (round 3): a point on an asymptote or an
  extended-value estimate is not the exact MLE.
- **The GOF `:distance` panel counts unordered dyads on undirected
  networks and carries an `"Inf"` level** (round 3, major). It counted
  ordered pairs (`for i, j != i`) — every printed count was twice R's on an
  undirected network (flomarriage: `[40, 70, 64, 30, 6]` for R's `[20, 35,
  32, 15, 3]`) — and dropped the unreachable pairs, the reachability
  comparison the panel exists for. The panel is now R's `obs.dist`: one
  level per finite distance and a final `"Inf"` level (15 on flomarriage,
  the pairs involving the isolated Pucci family), pinned to R at 1e-9 on
  flomarriage and on the directed samplike (ordered pairs, `Inf` = 0) by
  the provenanced fixtures, together with R's `obs.deg`/`obs.esp` panels.
  `labels[end] == "Inf"` and `simulated` has one more column.
- **`dof`, AIC and BIC follow R's `logLik` df/nobs after a drop** (round
  3, minor). `dof` counted every coefficient, including one fixed at
  `-Inf` by a boundary statistic, and BIC used all dyads; R's `logLik.ergm`
  carries df = the finite coefficients and nobs = the dyads the dropped
  term does not touch (`nobs(fit)` itself stays every dyad). On the 30-node
  separated-nodematch network R gives df 1, AIC 102.7665, BIC 106.4703
  (= −2ℓ + log 300); ERGM.jl reported dof 2, AIC 104.77, BIC 112.94 — now
  R's numbers, pinned by the provenanced `sep_*` rows of `ergm_terms.toml`.
- **Directed `GWDSP(type=:OSP)` / `type=:ISP` sum over ordered dyads, as
  R's `dgwdsp` does — the statistic and its change statistic double**
  (panel 2026-09 round 2, major). Both were summed over *unordered* dyads
  and came out at exactly half of statnet (on `samplike`, R 4.12.0:
  `gwdsp.OSP.fixed.0.5 = 316.085`, `gwdsp.ISP.fixed.0.5 = 229.136`;
  ERGM.jl gave 158.04 / 114.57), with a consistently halved change
  statistic, so MPLE/MCMLE coefficients for these two types were off by a
  factor of 2 relative to R and the randomized brute-force test could not
  see it. Every directed type now follows R's `ddsp` C convention (all
  ordered dyads; `:OTP`/`:ITP` were already right, `:union` is unchanged),
  and the provenanced `ergm_terms.toml` pins all of `gwesp`/`gwdsp`
  OTP/ITP/OSP/ISP on `samplike`. *Migration:* a directed fit with
  `GWDSP(d; type=:OSP)` or `:ISP` refit under 0.2 gives half the old
  coefficient on that term (the same model, R's parametrisation).
- **Directed `GWESP`/`GWDSP` coefficients carry R's directed label,
  `gwesp.OTP.fixed.<decay>`** (round 2, minor). R ergm 4.12.0 names the
  default type *with* its type on a directed network
  (`summary(samplike ~ gwesp(0.5, fixed=TRUE))` is `gwesp.OTP.fixed.0.5`;
  the undirected label `gwesp.fixed.0.5` has no type). ERGM.jl printed the
  undirected label on directed models, so by-name comparison with a
  statnet directed fit broke for the most common specification. The label
  is now resolved against the network: a new two-argument method of the
  shared statistic generic, `name(term, net)` (default `name(term)`), is
  what `ERGMModel`/`fit_ergm` (`fit.model.formula.terms.names`),
  `summary_stats` and `TermSet(terms, names)` use — `gwesp.fixed.<d>` on
  an undirected network for every type (the type is ignored there, in the
  label as in the statistic), `gwesp.<type>.fixed.<d>` on a directed one.
  The one-argument `name(GWESP(0.5))`, which cannot see a network, is
  unchanged (`gwesp.fixed.0.5`). Pinned by the fixture's samplike names.
  *Migration:* code indexing a directed fit's coefficients by
  `"gwesp.fixed.0.5"`/`"gwdsp.fixed.0.5"` must use `"gwesp.OTP.fixed.0.5"`
  / `"gwdsp.OTP.fixed.0.5"`; an undirected `GWESP(d; type=:ISP)` model is
  now labelled `gwesp.fixed.<d>` (was `gwesp.ISP.fixed.<d>`).
- **`AbsDiff(attr; pow)` with `pow != 1` is named `absdiff<pow>.<attr>`**
  (round 2, minor), R's label (`absdiff2.wealth`, `absdiff0.5.wealth`),
  so two powers of one attribute in a formula print distinct names; pinned
  by the fixture's `absdiff("wealth", pow=2) = 91570` row.
- **`summary_stats` honours the missing-data contract** (round 2, major).
  `summary_stats(net, terms; missing=:error)` — statnet's `summary()` —
  refuses a network with masked dyads through the shared
  `Networks.require_observed` (the message names `missing=:face`, which
  the routine now takes), and reads face values only under the explicit
  `missing=:face` opt-in; `missing_policies(summary_stats) == (:error,
  :face)`. It used to return `(edges = 20.0, …)` silently on a masked
  network. Its keys are the model labels (`name(term, net)`), so a
  directed network's `GWESP(0.5)` is `gwesp.OTP.fixed.0.5` there too. R's
  `summary()` counts an NA dyad as absent; the masked fixture reproduces
  R under `missing=:face` on a copy with the masked ties removed.
  `compute`/`compute_all` remain protocol-level raw evaluations.
  *Migration:* pass `missing=:face` where a masked network's face-value
  statistics were wanted.
- **A network containing a self-loop is refused at `ERGMModel`** (round 2,
  minor). The term statistics counted a loop, but `nobs`, the
  pseudo-likelihood design, the MH proposal and every simulation range
  over the off-diagonal dyads only, so an `edges`-only MPLE on a 6-vertex
  network with one loop was `logit(3/15)` against an observed statistic of
  4 and GOF compared a loop-inclusive observed statistic with loop-free
  simulations, without a word (R ergm at least warns "This network
  contains loops"). `ERGMModel`/`fit_ergm` now throw an `ArgumentError`
  naming the looped vertices when a loop is *present*; a `loops=true`
  network with no loop is accepted and modelled as the loop-free network
  it is (the diagonal is structurally absent). Listed under Known
  limitations. *Migration:* `rem_edge!(net, v, v)` first.
- **Boundary statistics: R's `drop=TRUE` semantics in `mple`, refusal in
  `mcmle`** (round 2, major — the front-page Quick Start fit a perfectly
  separated `nodematch`, "converged" at −24.4 with a standard error of
  21,392 and said nothing). `_boundary_columns` reads the compressed
  design for a column whose nonzero rows are all non-ties (or all ties)
  with one sign: no finite MPLE exists there. `mple` now warns R's
  sentence ("observed statistic(s) nodematch.group are at their smallest
  attainable values. Their coefficients will be fixed at -Inf"), returns
  the coefficient as `-Inf` (`+Inf` at the largest value) with standard
  error 0, z `∓Inf`, p-value 0, and fits the remaining coefficients on the
  rows the dropped columns do not touch — the exact limit of the
  pseudo-likelihood and exactly R's numbers (30-node docs network: edges
  `-3.178054`, SE `0.2946269`, logLik `-50.38324` = `logit(12/300)`, not the
  edges-only `logit(12/435)`; pinned). `show` prints a note under the
  table and `Networks.approximations(fit)` lists the fixed coefficients.
  `mcmle` runs the same detector before initialising and throws an
  `ArgumentError` quoting the sentence and the two ways out (drop the
  term, or `method=:mple`) — a `-Inf` coefficient has no Newton update and
  the sampler could not honour it. R only knows `nodematch`'s minimum (it
  warns "The MPLE does not exist!" at the maximum); ERGM.jl detects both
  sides. *Migration:* a formula that used to return a huge finite
  coefficient with a huge SE now returns `∓Inf` with SE 0 and a warning;
  `isfinite.(coef(fit))` tells them apart.
- **`_mh_run!` lost its redundant `directed::Bool` argument** (round 2,
  minor): `_mh_run!(rng, net, terms, θ, n_samples, burnin, interval,
  collect_networks, toggleable=:free)`; the directedness is the network's
  type parameter. Private; no variant called it.
- **`Kstar` and `GWDegree` are undirected-only, as in R ergm; directed
  networks get the new `OStar`/`IStar`** (panel 2026-09, item 12b).
  `requires_undirected(::Kstar) == requires_undirected(::GWDegree) == true`,
  so `ERGMModel`/`fit_ergm` refuse them on a directed network with an
  `ArgumentError` naming the directed variants (`OStar(k)`/`IStar(k)`,
  `GWODegree`/`GWIDegree`; the message quotes R's "Term may not be used with
  networks with directed==TRUE"), and the terms' own `compute`/`change_stat`
  refuse a directed network too, so a caller bypassing `ERGMModel` cannot
  get the old statistic either. Before, a directed network silently received
  *out*-stars / the out-degree GW statistic under the undirected label
  `kstar2` / `gwdegree.fixed.0.5` — a different model than the one named.
  *Migration:* on directed networks replace `Kstar(k)` with `OStar(k)` (the
  statistic you were actually getting) and/or `IStar(k)`, and `GWDegree(d)`
  with `GWODegree(d)` and/or `GWIDegree(d)`.
- **Coefficient names carry R's decay labels, and `GWDegree` is labelled
  `gwdeg`** (item 12, panel-missed). An integer-valued fixed decay prints
  without a decimal point, as in statnet: `GWESP(0.0)` is `"gwesp.fixed.0"`
  and `GWESP(1.0)` is `"gwesp.fixed.1"` (was `"gwesp.fixed.1.0"`), same for
  `GWDSP`, `GWIDegree`, `GWODegree`; fractional decays are unchanged
  (`"gwesp.fixed.0.5"`). `GWDegree(d)` is named `"gwdeg.fixed.<d>"` — R's
  label, matching `gwideg`/`gwodeg` — instead of `"gwdegree.fixed.<d>"`.
  By-name comparison with a statnet fit now works for every fixed-decay term;
  the provenanced `ergm_terms.toml` fixture asserts the names R emits.
  *Migration:* code indexing coefficients by these names must use the R
  spellings.
- **A vertex without an attribute value is refused, not zero-filled**
  (item 12c). `ERGMModel` (hence `fit_ergm`/`ergm`) throws an
  `ArgumentError` when a term's declared vertex attribute exists but is not
  set on every vertex — or is set to `missing`/`nothing` — naming the term,
  the attribute, the count and the vertex ids ("term 'nodecov.wealth' needs
  vertex attribute :wealth on every vertex, but 3 of 16 vertices have no
  value (vertices 3, 7, 12). statnet refuses NA attribute values; …").
  Before, `NodeCov`/`AbsDiff` silently read `0.0` for such a vertex and the
  categorical terms treated it as matching nothing — a design column nobody
  asked for. The raw `NodeCov`/`AbsDiff` `compute`/`change_stat` throw the
  same `ArgumentError` on incomplete data instead of zero-filling; the
  code-0 branches of the materialized categorical terms remain as
  defensive code but are unreachable through `ERGMModel`. *Migration:* set
  a value for every vertex (`vertex_attribute_vector`/`default=` at the
  Networks level if a fill is genuinely intended) or drop the term.
- **Curated re-exports (panel 2026-09, item 3).** `using ERGM` re-exports
  Networks.jl through an explicit list mirroring Networks' export blocks,
  replacing the runtime loop over `names(Networks)`. Now re-exported:
  `Network` (the loop skipped it, so `using ERGM; Network(5)` was an
  `UndefVarError`), `BipartiteNetwork`, `degree`/`indegree`/`outdegree`,
  `missing_policies`, `z_pvalues`, `CoefficientTable`, `coeftable`. **No
  longer re-exported** — qualify them: `Networks.load_golden`,
  `GoldenFixture`, `check_golden`, `golden_report`, `golden_tolerance`,
  `bootstrap_cov`, `record_drop!`, `check_se`, `check_statsapi`, and the
  module name `Networks` itself. A drift testset pins the list against
  Networks' frozen inventory. *Migration:* add `using Networks` (or qualify)
  where a test or script used the golden harness or `bootstrap_cov` through
  `using ERGM` alone.
- **`ERGMModel{T,D}` / `ERGMResult{T,D}` — directedness is a type parameter,
  and the `directed` field is gone** (item 19). `model.network` is a concrete
  `Network{T,D}` (it was the UnionAll `Network{T}`, dynamically dispatched at
  every read), `Graphs.is_directed(model)` replaces `model.directed`,
  `sample_networks`/`simulate_ergm` return a concretely typed
  `Vector{Network{T,D}}` (was `Vector{Network{T}}`), and the two positional
  `ERGMResult` constructors from before `se_type`/`missing_method` existed are
  removed (no caller remained). *Migration:* `model.directed` →
  `is_directed(model)`; `ERGMModel{T}` in a signature → `ERGMModel{T,D}` or
  plain `ERGMModel`.
- **`mcmle`'s iteration cap is `maxiter`** (item 16; the ecosystem-wide
  spelling, as in ERGMCount/ERGMMulti/ERGMRank/TERGM/REM/`newton_fit`).
  `max_iter` is accepted as a deprecated shim for one release: it is honoured
  and warns once. `fit_ergm(...; method=:mcmle, max_iter=…)` therefore still
  runs, with the warning. *Migration:* rename the keyword.
- **Two-mode networks are refused instead of silently mis-fit.** `ERGMModel`
  (hence `fit_ergm`/`ergm`) throws an `ArgumentError` on a network with the
  two-mode flag (`network(n; bipartite=k)`) or a `BipartiteNetwork`. Before,
  the model constructed and the MPLE counted the structurally impossible
  within-mode dyads as observations (`nobs == 30` on 8 cross-mode dyads) while
  the sampler toggled them. Bipartite terms (`b1degree`, …) are not
  implemented — see the README's "Not implemented" section.
- **`ERGMFormula(...; constraints=[...])` is refused** with an `ArgumentError`
  for a non-empty vector. The field was stored and never read by any sampler
  or estimator, so a "constrained" model was silently the unconstrained one.
  `ConstraintTerm` stays exported as a reserved abstract type.

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

- **The variants' building blocks are `public`** (reconciliation of the
  2026-09 cross-repo requests from TERGM, ERGMMulti, ERGMRank, ERGMCount and
  ERGMUserterms; panel item 13): `_boundary_columns`,
  `_boundary_columns_iterated`, `_separated`, `_warn_boundary`,
  `_warn_separated`, `_mple_fit_design`, `_collect_terms`, `_expand_terms`,
  `_refuse_two_mode`, `_refuse_self_loops`, `_mcmle_covariance`,
  `_warn_degenerate_stats` and `_bridge_logZ` join the `public` block, each
  with a docstring and a runnable example (the docstring gate walks `public`
  names). `_mple_fit_design` takes `context::AbstractString="mple"` and
  prefixes both R sentences with it (TERGM's CMPLE passes `"cmple"` and
  drops its re-emission of the warnings); `_warn_boundary` takes `note=`
  (the parenthesis after "no finite maximum pseudo-likelihood estimate
  exists" — ergm.multi's counterpart does not drop, so ERGMMulti says so
  there instead of restating the sentence) and `noun=` (the rows the
  remaining coefficients are fitted on: ERGMRank's "swap comparisons").
  The dependants' reach-in testsets, which held these names as
  `@test_broken`/allow-listed, now assert them public.
- **`_expand_terms(terms, net)`** — the *specification* of a term list:
  `_materialize`'s expansion of a multi-level `NodeFactor`, multi-cell
  `NodeMix` and multi-degree `Degree`/`IDegree`/`ODegree` against the
  network, returned as plain single-statistic terms WITHOUT the attribute
  snapshot. A model that evaluates its terms on several networks (TERGM's
  T−1 auxiliary panels, ERGMMulti's layers) stores the specification and
  materializes per network; TERGM's `_specification`, which unwrapped the
  twin's `base` field from outside ERGM, is deleted for it.
- **`src` and `dst` are re-exported** with `edges` (Networks.jl's inventory
  gained them for the term authors who write `for e in edges(net); src(e)
  …`); the drift pin against Networks' exported inventory is unchanged.
- **`MCMCDiagnostics`** (round 3): the exported result type of
  `mcmc_diagnostics`, with a `show` that prints the per-term table (see
  Breaking) and a docstring example.
- **`fit.boot_replicates`** (round 3): every parametric-bootstrap refit,
  so percentile intervals and the excluded replicates are inspectable.
- **Provenanced fixture rows** (round 3, `test/fixtures/r/ergm_terms.R`
  sections (e)–(g) and `flomarriage_ergm.R`): R's drop-semantics fit on
  the 30-node separated network (coefficients, SEs, logLik, df, logLik
  nobs, `nobs()`, AIC, BIC — the testset's bare R literals are gone), the
  8-node mixed-sign `nodecov` design and R's "The MPLE does not exist!"
  warning, a 6-node design separated by a *combination* of two `nodecov`
  columns with the same warning, and `gof()`'s observed degree/esp/distance
  panels on flomarriage and samplike.
- **Missing-data maximum likelihood: `mcmle(model; missing=:mle)`** (panel
  2026-09, item 32 / N5 — the last open July item). A network with dyads
  masked as missing can now be fitted by the estimator R `ergm()` uses for
  NA ties (Handcock & Gile 2010): the likelihood of the observed dyads is
  `Z_obs(θ)/Z(θ)`, the masked dyads integrated out. Each MCMLE iteration
  runs two chains — the *free* chain over every dyad (masked ones included)
  giving `ḡ = Ê[g(Y)]` and `Σ_free`, and a *constrained* chain that toggles
  only the masked dyads with the observed dyads held fixed, giving
  `ĝ_obs = Ê[g(Y) | Y_obs]` and `Σ_obs`. `ĝ_obs` replaces the observed
  statistics as the Newton target (`θ += γ Σ_free⁻¹ (ĝ_obs − ḡ)`), the
  convergence t-ratios and Hotelling test compare the two chains with the
  difference's variance `Σ_free/n_eff + Σ_obs/n_eff_obs`
  (`mcmc_convergence(...; target_samples=, target_chain_lengths=)`), the
  Fisher information is `Σ_free − Σ_obs` (Eq. 9 there; checked for positive
  definiteness — NaN standard errors with a warning otherwise), both chains
  contribute Monte-Carlo error to the standard errors, and the
  log-likelihood is `log Z_obs(θ) − log Z(θ)` from two path-sampling bridges
  (`_bridge_logZ` with `toggleable=:all` from the exact dyad-independent
  normalizer over all dyads, and with `toggleable=:masked` from the exact
  normalizer over the masked dyads with the observed dyads fixed;
  `_dyad_independent_logZ(model, θ; toggleable=)` computes all three). New
  keywords `obs_burnin`/`obs_interval` (R's `obs.MCMC.burnin`/`.interval`)
  control the constrained chain, defaulting to the dyad-scaled rule applied
  to the number of masked dyads. The fit records `missing_method = :mle`,
  `show` prints "missing-data maximum likelihood (masked dyads integrated
  out by a constrained chain)", `approximations(fit)` lists the MC error of
  the target, `nobs` is the number of observed dyads, and
  `supports_missing(mcmle) == true` (the default `missing=:error` still
  refuses; `missing_policies(mcmle) == (:error, :condition_on_face, :mle)`).
  Under dyad independence the estimate coincides with the available-case
  MPLE (the exact MLE of the observed dyads; pinned to 1e-6 and to the exact
  log-likelihood). **Validated against R** by the new provenanced fixture
  `test/fixtures/flomarriage_missing_ergm.toml` (`Rscript
  test/fixtures/r/flomarriage_missing_ergm.R`, R 4.6.1 / ergm 4.12.0):
  flomarriage with dyads (3,4), (1,9), (7,16), (2,11) set to NA — two ties,
  two non-ties. The mean of five seeded `missing=:mle` fits of
  `edges + gwesp(0.5, fixed=TRUE)` at `n_samples=4096` lands 0.0063/0.0075
  from R's frozen fit and 0.002/0.0002 from R's own five-seed mean (R's
  seed-to-seed sd 0.0028/0.0053; asserted at 0.03, ≥ 5× that sd), the
  standard errors 0.0001/0.0009 from R's, and R's "MCMC %" column (0, 0) is
  reproduced. The same fixture is the first R pin of `supports_missing(mple)`:
  R's dyad-independent fit with NA dyads is the logistic regression on the
  116 observed dyads, which `mple` reproduces to 1e-6 (observed ~1e-11), and
  records that statnet's `summary()` counts NA dyads as absent (18 edges,
  not 20), asserted at 1e-9 on a copy with the masked ties removed. The
  `:condition_on_face` and `:mle` estimands differ by 0.11 on both
  coefficients there — twenty R seed sds — pinned so the distinction stays
  visible. Every "not yet implemented / issue #4" sentence about missing-data
  ML is gone from the code, README, docs and CLAUDE.md.
- **`mh_sample(...; toggleable=:free | :all | :masked)`** — the dyad set a
  chain may toggle, implemented purely in `_mh_run!`'s proposal closure:
  `:free` (every unmasked dyad; the default and bit-identical to before),
  `:all` (every dyad, the unconditional model) and `:masked` (only the
  masked dyads, drawn uniformly from their list, observed dyads fixed —
  requires a mask). The two missing-data chains read no unobserved value as
  an observation, so they take no `missing=` opt-in (one is refused as
  meaningless). Dyad-scaled defaults scale with the toggleable set
  (`_resolve_mcmc_controls(...; toggleable)`, `_n_toggleable`), and
  `_random_network(net; randomize_masked=true)` gives a `:all` chain a start
  with the masked dyads randomized too (mask kept).
- **Every export has a docstring with a runnable example, and the suite runs
  them** (grade-A criterion 5). The testset "Every exported docstring carries
  a runnable example" walks `names(ERGM)` (exported and `public`), requires an
  ERGM-owned docstring with a fenced ```julia block for every ERGM binding
  (re-exported Networks/Graphs/StatsAPI names need only be documented
  somewhere), and evaluates every block in a fresh module after `using ERGM`
  — 82 blocks. Examples were added to 25 exported names (`Edges`, `Mutual`,
  `Triangle`, `TwoPath`, `EdgeCov`, `NodeFactor`, `NodeMatch`, `NodeMismatch`,
  `NodeMix`, `AbstractERGMTerm`, `TermSet`, `compute`, `compute_all`,
  `change_stat`, `change_stat_all`, `name`, `coef`, `stderror`, `vcov`,
  `objective`, `is_exact`, `se_method`, `mcmc_diagnostics`, the four trait
  generics) and docstrings to the seven `public` helpers that had none
  (`_requires_directed`, `_requires_undirected`, `_vertex_attribute`,
  `_validate_formula`, `_materialize`, `_has_dyad_dependent`, `_z_pvalues`).
  Sketches that are deliberately not standalone programs (the `mh_toggle!`
  TERGM adoption sketch, the `Degree(0:2)`/`NodeMatch(diff=true)` idioms)
  use a ```jl fence, which highlights the same and is not executed.
- **`mh_toggle!` — the Metropolis toggle kernel, exported** (panel 2026-09,
  item 28). `mh_toggle!(rng, θ, delta, propose, change!, apply!, on_sample;
  burnin, interval, n_samples)` owns only the accept/reject arithmetic,
  burn-in and thinning; the proposal, the add-direction change statistics
  (with a `removal::Bool` return that flips the log-ratio) and the state
  mutation are callables, so the same loop samples binary networks (ERGM's
  `_mh_run!` is now a thin adapter), TERGM's constrained formation/
  dissolution networks, ERGMMulti's layers and ERGMRank's rankings (adoption
  sketches in the docstring; ERGMCount's Gibbs sweep is not a toggle). The
  refactor is bit-identical: frozen pre-refactor `mh_sample` /
  `sample_networks` outputs are pinned, and the kernel adds 0 bytes per step
  (pinned).
- **`mcmle(...; n_chains=1)`** (item 24c): each iteration's `n_samples` and
  the final sample are split over `n_chains` independent chains seeded from
  `rng` exactly as `sample_networks` does (per-chain counts, one seed per
  chain, `Threads.@spawn`, all from the observed network), concatenated in
  chain order; the Hotelling test's effective sample size is the sum of the
  per-chain Geyer ESSs. Bit-identical at any thread count (pinned by a
  fresh-process test at a different `--threads`), `n_chains=1` is exactly the
  old single-chain sampler, and `n_chains` never defaults to
  `Threads.nthreads()`.
- **`mcmc_se(fit)`, `fit.vcov_fisher`, `fit.mcmc_convergence`** (items 24a,
  24d): the Monte-Carlo component of the MCMLE standard errors (see
  Changed), the pure inverse-Fisher covariance, and the convergence report
  `MCMLEConvergence = (iterations, step_length, t_ratios, hotelling_p,
  n_eff)` recomputed on the final sample. MPLE fits carry `vcov_fisher ==
  vcov`, `mcmc_se == zeros`, `mcmc_convergence === nothing`.
- **`mcmle(...; bridge_rungs=0)`** (item 24b) skips the path-sampling
  log-likelihood: `loglik`, `aic`, `bic` are `NaN`, `show` prints
  "Log-likelihood: not estimated (bridge_rungs=0)", `approximations(fit)`
  records it, and — because the bridge runs last and consumes randomness only
  after the final sample — coefficients, standard errors and `mcmc_samples`
  are bit-identical to the `bridge_rungs=16` fit with the same rng (pinned).
  The panel measured the bridge at ~70 % of a single-threaded `mcmle`.
- **`mcmle(...; init=)`** — starting coefficients (statnet's
  `control.ergm(init=)`), so the non-convergence warning's "refit from these
  coefficients" is one call: `mcmle(model; init=coef(fit))`.
- **`ERGM.mcmc_convergence(samples, targets; conv_threshold,
  hotelling_alpha, chain_lengths)`** (`public`): the per-statistic t-ratios,
  chain-aware Geyer ESS, Hotelling T² p-value and verdict as one function,
  used by `mcmle` itself and offered to ERGMEgo's moment matching in place
  of its 1 % relative-change rule.
- **`ERGM._mcmc_defaults(model)`** (`public`): THE dyad-scaled sampler rule
  `(burnin = 20 × n_dyads, interval = max(100, n_dyads ÷ 10))` — see
  Changed.
- **`OStar(k)` / `IStar(k)`** — statnet's `ostar`/`istar` (names
  `"ostar<k>"`/`"istar<k>"`, `requires_directed`, O(1) change statistics
  from the endpoint's out-/in-degree, 0 B per `change_stat`; pinned by the
  brute-force, randomized and benchmark regression lists and by the samplike
  rows of the provenanced term fixture) (item 12b).
- **Decay 0 for every geometrically weighted term** (item 12a): `GWESP`,
  `GWDSP`, `GWDegree`, `GWIDegree`, `GWODegree` accept `decay >= 0` (was
  `> 0`, refusing statnet's most common specification
  `gwesp(0, fixed=TRUE)`). The weight formulas are exact at α = 0 (`0^0 == 1`):
  `GWESP(0.0)` is the number of edges with ≥ 1 shared partner, `GWDSP(0.0)`
  the number of dyads with ≥ 1 shared partner, `GWDegree(0.0)` the number of
  non-isolates, `GWIDegree(0.0)`/`GWODegree(0.0)` the vertices with in-/out-
  degree ≥ 1 — asserted against brute force and against R. A negative or
  non-finite decay throws "decay must be non-negative". Decays may be given
  as any `Real` (`GWESP(0)`).
- **Provenanced term-parity fixture `test/fixtures/ergm_terms.toml`**,
  generated by `test/fixtures/r/ergm_terms.R` from R 4.6.1 / ergm 4.12.0 and
  loaded with `Networks.load_golden`: on `flomarriage` the summary statistics
  and R's coefficient names for `edges + gwesp(0, fixed=TRUE) +
  gwdsp(0, fixed=TRUE) + gwdegree(0, fixed=TRUE) + gwdegree(1, fixed=TRUE) +
  degree(0:2)` (1e-9) and the `estimate="MPLE"` coefficients/SEs of
  `edges + degree(0:2)` (dyad-dependent, so pseudo-likelihood against
  pseudo-likelihood — exact at 1e-6, observed ~1e-13); on `samplike`
  (Sampson's monastery, exported as an edge list in the fixture and rebuilt
  by a `samplike()` test helper) the statistics of `edges + mutual + ostar(2)
  + istar(2) + gwidegree(0.5, fixed=TRUE) + gwodegree(0.5, fixed=TRUE) +
  triangle` (1e-9); and the error strings R raises for `kstar(2)` /
  `gwdegree(0.5, fixed=TRUE)` on the directed network and for `nodecov` with
  an NA, so the fixture documents that R refuses too and the testset asserts
  ERGM.jl errors wherever R errors.
- **`has_dyad_dependent(model)`** is exported and documented (item 13): THE
  predicate behind the `show` caveat and `is_exact`. `_has_dyad_dependent` is
  a deprecated `const` alias; the variants should add methods to the public
  name rather than define same-named privates.
- **`confint(fit; level=0.95)` and `coeftable(fit)`** complete the ecosystem
  StatsAPI surface on `ERGMResult` (item 15): normal-theory Wald limits (one
  row per coefficient), and a `Networks.CoefficientTable` built from the same
  vectors `show(fit)` prints — `show` now renders through `coeftable`, so
  the printed and the inspected table cannot disagree.
  `Networks.check_statsapi(fit; strict=true)` passes for MPLE, MCMLE and
  bootstrap fits, pinned by the "StatsAPI surface is complete" testset.
- **`Networks.missing_policies` declared** for every MCMC entry point (item 5):
  `missing_policies(mh_sample) == missing_policies(sample_networks) ==
  missing_policies(simulate_ergm) == (:error, :condition_on_face)` and
  `missing_policies(mcmle) == (:error, :condition_on_face, :mle)`; `gof` is
  the shared generic, so ERGM declares the per-result-type form
  `missing_policies(gof, ::Type{<:ERGMResult})`.
  `missing_policies(mple) == (:error,)` — it handles the mask and needs no
  keyword. Tooling prints these instead of assuming `:face`.
- **Actionable `fit_ergm`/`ergm` call-site errors** (item 31): a single term
  needs no brackets (`ergm(net, Edges())`); swapped positional arguments
  (`fit_ergm(terms, net)`) throw "arguments are swapped: call fit_ergm(net,
  terms)"; a `Vector{Any}` of terms is accepted and a non-term element is named
  with its position and type (a term *type* such as `Edges` gets "did you mean
  `Edges()`?"); an unknown `method=` names the two that exist; and
  `[Edges(), Degree(0:2)]` — the spelling the README always showed — now
  splices the expanded degree terms in (`_collect_terms`, also behind
  `ERGMFormula`, `TermSet` and `summary_stats`) instead of a raw `MethodError`.
- `public` declarations (Julia ≥ 1.11) for the underscore names other packages
  reach into — `_requires_directed`, `_requires_undirected`,
  `_vertex_attribute`, `_validate_formula`, `_materialize`, `_copy_network`,
  `_n_dyads`, `_has_dyad_dependent`, `_z_pvalues` — with their `===`
  identities pinned by a testset.
- Non-convergence is loud for both estimators: `mple` warns when the Newton
  iteration exhausts `maxiter` and `mcmle` warns when its convergence tests
  never pass; both record `converged = false` and add an entry to
  `Networks.approximations(fit)` (item 24, the loud-non-convergence part).
- `mple` takes `maxiter=100` and `tol=1e-8` (forwarded to `newton_fit`).

- **Provenanced golden fixture against a real statnet `ergm` fit** (issue #8).
  `test/fixtures/flomarriage_ergm.toml` freezes an ergm 4.12.0 / R 4.6.1 fit of
  the Florentine marriage network, regenerable with
  `Rscript test/fixtures/r/flomarriage_ergm.R > test/fixtures/flomarriage_ergm.toml`
  and loaded through Networks.jl's `load_golden`, which refuses a fixture with no
  provenance. It replaces — and, as of the 2026-09 sweep, has absorbed — the R
  coefficients that lived as bare literals in test comments ("Golden master vs
  R ergm", "Golden master: MCMLE matches R statnet"): those could not be
  regenerated, carried no record of which ergm produced them, and had
  hand-chosen atols beside them. The fixture now also freezes the Bernoulli
  model `flo ~ edges` (`eo_coefficients`/`eo_std_errors`, 1e-6; the MCMLE of
  that dyad-independent model is asserted against it too) and the triangle
  count, and the literal testsets are deleted.

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
  pseudo-likelihood, a principled treatment, and `supports_missing(mcmle) ==
  true` since `missing=:mle` (above). `ERGMResult` gains a
  `missing_method::Symbol` field recording what actually happened (`:none`,
  `:available_case`, `:mle` or `:condition_on_face`), and `show` reports the
  treatment whenever the network carried a mask.

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
  and `mcmle` refuses a masked network unless `missing=:condition_on_face`
  is passed (see Breaking), in which case it warns that it conditions on
  their face values.
- Public building blocks for downstream packages: `mh_sample(model, θ)`
  (single-chain MH sampler returning sampled statistics/networks; replaces
  ERGMEgo's private-API use), `newton_fit(loglik_grad_hess, θ0)` (shared
  Newton–Raphson with step halving — since moved to Networks.jl and
  re-exported, see Changed), and the `is_dyad_dependent(term)` trait.
- `rng::AbstractRNG` keywords on all sampling and fitting functions
  (`ergm`, `mcmle`, `simulate_ergm`, `sample_networks`, `gof`,
  `mh_sample`); same seed ⇒ identical results independent of thread count.
- StatsAPI accessors: `loglikelihood`, `aic`, `bic`, `nobs`, `dof` join
  `coef`/`stderror`/`vcov`; `vcov` now returns the full covariance matrix
  (was a diagonal reconstruction); `ERGMResult` records `se_type`.
- `mcmc_diagnostics` adds Geyer initial-sequence ESS (`ess_geyer`) and
  Geweke convergence diagnostics (`geweke_z`/`geweke_p`) per statistic.
- `using ERGM` now re-exports the Networks.jl user-facing API (constructors
  including `Network`, attribute setters, `load_dataset`, ...) through the
  curated list described under Breaking, so one import suffices.
- BenchmarkTools suite (`benchmark/`) with allocation regression tests.

### Changed

- Documentation uses the default Documenter themes, with a new package-specific
  SVG icon and browser favicon in the official Julia logo colors.
- **The masked-network refusal is ONE message** (panel item 5 follow-up):
  `_guard_missing` passes ERGM's opt-in bullets (`missing=:mle`,
  `missing=:condition_on_face`) through `Networks.require_observed`'s new
  `hint=` keyword instead of catching the shared `ArgumentError` and
  rethrowing `e.msg * hint`. Same bullets, same order, one refusal.
- **`_fmt3` prints `%.3g`** (ERGMCount round 3): `round(x; sigdigits=3)`
  yields a Float64 that is not exactly representable, so the MCMLE and
  bridge diagnostics printed `6.969999999999999e-32` and
  `1.6699999999999998e33`; they now print `6.97e-32` and `1.67e+33`
  (`Printf` is a new stdlib dependency).
- **`newton_fit`'s convergence verdict is scale-free** (Networks.jl,
  reconciliation round): the shared optimizer `mple` runs on now declares
  convergence on the Newton decrement `½ ∇ℓᵀ(−H)⁻¹∇ℓ < tol` beside the
  gradient norm, accepts a step whose objective decreases by rounding noise
  while the gradient shrinks, and takes the last full step at a stall whose
  predicted gain is below `tol`. On a design with a large-valued
  dyad-dependent column (TERGM's `Triangle` on Y⁺) the MPLE used to report
  `converged=false` one ulp short of the maximum where R's glm converged;
  it no longer does. Coefficients can move at the 1e-9 level and
  `iterations` by one; every golden fixture still agrees at its tolerance.
- **`_boundary_columns` is R's attainable-range test, iterated** (round 3,
  major — see Fixed). A column is at its minimum iff every positive-change
  dyad is a non-tie and every negative-change dyad is a tie (mirror for the
  maximum), whatever the sign pattern; dropping a column restricts the
  pseudo-likelihood to the dyads it does not touch, and the test is
  repeated on that reduced design (`_boundary_columns_iterated`) until
  nothing more is at a boundary — the exact limit of the pseudo-likelihood
  (on the mixed-sign fixture design: `nodecov.x → -Inf`, then `edges →
  +Inf` on the three untouched, all-tied dyads; log-likelihood 0, dof 0).
- **`mcmle` refuses a boundary statistic before computing its MPLE start,
  and builds the pseudo-likelihood design once** (round 3, minor). The
  refusal sat after `mple(model)`, so a separated model first printed the
  `mple: … fixed at -Inf` warning and then threw, and `_mple_data` (the
  O(n²) sweep) ran twice. The design is now built once, checked, and handed
  to `_mple_fit_design` for the start; the boundary testset asserts the
  refusal emits no log line. A start whose MPLE does not exist (perfect
  separation) is refused too, pointing at `init=`.
- **`AbsDiff(attr; pow::Real)`** (round 3, minor): `pow` was typed
  `Float64`, so R's spelling `absdiff("wealth", pow=2)` → `AbsDiff(:wealth;
  pow=2)` threw a raw `TypeError`; any `Real` is accepted and stored as
  `Float64` (`name(AbsDiff(:wealth; pow=2)) == "absdiff2.wealth"`).
- **`NodeCov(attr; transform=)` validates the transform** (round 3,
  minor): anything but `:none`/`:log`/`:sqrt` used to fall back silently
  to no transform (`NodeCov(:wealth; transform=:exp)` computed 2168.0); it
  is now an `ArgumentError` naming the three options.
- **"MCMC %" is R's definition** (round 2, minor). `show(::ERGMResult)`
  printed `round(100·mcmc_se²/se²)` — the share of the standard-error
  *variance* — under R's column name; `ergm:::summary.ergm` computes
  `round(100 * (tot.se - mod.se) / tot.se)`, the share of the standard
  error itself (on the Florentine `edges + gwesp` model at `n_samples=150`
  the two give `[1, 1]` vs `[2, 1]`; both round to 0 at 4096, which is why
  the fixture never noticed). `_mcmc_percent` is now
  `round(100·(se − sqrt(diag(vcov_fisher)))/se)`, NaN-safe, the line reads
  `MCMC % of the standard error (100·(se − se_fisher)/se): …`, and the
  `mcmc_se`/`mcmle`/`se_method` docstrings, README, estimation and
  diagnostics guides say which definition it is. `mcmc_se(fit)` itself is
  unchanged; the variance share is available as
  `100 .* (mcmc_se(fit) ./ stderror(fit)) .^ 2`. Pinned against the hand
  formula.
- **StatsBase and SparseArrays are no longer dependencies** (round 2,
  minor, next to the Optim removal): nothing in `src` used a name from
  either module (`mean`/`cov`/`quantile` are `Statistics`); both `using`
  lines and both `[deps]`/`[compat]` entries are gone. StatsBase stays a
  test extra for the `using ERGM, StatsBase` co-loading check.
- **`gof` validates `stats` before simulating** (round 2, minor):
  `gof(fit; stats=[:degre])` used to run every simulation and then fail
  with "GOFResult needs at least one GOFStatistic", and
  `stats=[:degre, :esp]` silently returned the esp panel alone. An
  unknown symbol (or an empty vector) is now an `ArgumentError` naming the
  typo and the valid symbols (`:degree, :idegree, :odegree, :esp,
  :distance`), raised before a single network is drawn.
- **`EdgeCov` with a wrongly sized matrix is an `ArgumentError`, not a
  `BoundsError`** (round 2, minor): `_validate_formula`
  (`_check_covariate_size`), `compute` and `change_stat` name the matrix
  size and the vertex count ("term 'edgecov' has a 5×5 covariate matrix
  but the network has 16 vertices; EdgeCov needs an n×n matrix in vertex
  order").
- **`show(::ERGMModel)` / `show(::ERGMFormula)` print the formula** (round
  2, minor): `ERGMModel{Int64,false}: 16 vertices, 20 edges (undirected);
  terms: edges + nodecov.wealth` (plus `, k masked dyads` and a non-default
  reference) and `ERGMFormula: edges + nodecov.wealth`, instead of the
  struct dump with every materialized attribute vector.
- **Docs corrected against the code** (round 2, computational-social-
  science lens): the docs home page Quick Start no longer fits a perfectly
  separated `nodematch` with a raw `Triangle()` (groups A/B/C are
  contiguous vertex ranges and closure is `GWESP(0.5)`; the simulated edge
  count, ≈ 15 against 12, is stated); the getting-started Complete Example
  uses `GWESP(0.5)` and shows the `Triangle()` fit as an explicit
  degeneracy demonstration (≈ 67 simulated edges against 15 observed) with a
  pointer to Handling Degeneracy; Step 4's "Output" is the real
  `println(result)` of the MCMLE block that precedes it; the estimation
  guide's Displaying Results shows the current table format (z-value
  column, floored p-values, the "MCMC %" line, the `Converged: false`
  caveat) and its "Perfect separation" row describes the drop behaviour
  above.

- **The missing-dyad refusal and the `:condition_on_face` warning point at
  `missing=:mle`.** `mcmle`'s `require_observed` refusal now ends with two
  bullets (`missing=:mle` first, then `missing=:condition_on_face`); the
  samplers' refusals still name only `:condition_on_face`, and `missing=:mle`
  on `simulate_ergm`/`sample_networks`/`mh_sample`/`gof` throws an
  `ArgumentError` explaining that simulation under missing data is not
  defined (a simulated network has a value at every dyad) and how to
  proceed. The `:condition_on_face` warning no longer says missing-data ML is
  "not yet implemented"; it names `mcmle(model; missing=:mle)`.
  `_guard_missing` takes `policies=` (the estimator passes
  `_MCMLE_MISSING_POLICIES`) and returns the opted-in policy.
- **`_bridge_loglik` is `dot(θ, obs_stats) − _bridge_logZ(...)`**: the
  bridge is now a log-normalizer estimator over a `toggleable` dyad set, so
  the missing-data log-likelihood is the difference of two of them; the
  fully observed path is unchanged (same seeds, same numbers).
- **MCMLE standard errors include the Monte-Carlo component** (item 24d,
  Hunter & Handcock 2006 §3.3). `vcov(fit)` is `V + V·Σ_mc·V` with `V =
  inv(cov(final sample))` — the inverse Fisher information that used to be
  the whole answer, now `fit.vcov_fisher` — and `Σ_mc` the covariance of the
  sampled statistics' mean: the Geyer initial-sequence asymptotic variance
  per statistic on the diagonal, the lag-0 correlations off-diagonal,
  combined over chains as `Σ_c L_c Σ_c / n²`. `se_method(fit)` stays
  `:fisher` (its docstring says the MC term is added). `show` prints R's
  `summary.ergm` "MCMC %" column after the coefficient table — since round
  2 under R's own definition, `round(100·(se − se_fisher)/se)`, the share
  of the *standard error* (not of its variance), as the line `MCMC % of
  the standard error (100·(se − se_fisher)/se): …`; see the Changed entry
  "MCMC % is R's definition" for the numbers. The provenanced
  `dd_std_errors` comparison passes unchanged at 0.03.
- **Non-convergence is loud and specific** (item 24a). The warning is now
  `MCMLE did not converge in maxiter=N iterations (last max t-ratio X,
  Hotelling p Y, step length γ Z): the estimates are the last iterate and the
  standard errors are unreliable; increase maxiter/n_samples/burnin, check
  the model for degeneracy (mcmc_diagnostics), or refit from these
  coefficients (mcmle(model; init=coef(fit)))`; the same sentence (from one
  helper, so they cannot disagree) is printed by `show` directly under
  `Converged: false` and listed by `approximations(fit)` (was a generic
  "did not converge" line in `approximations` only). The unconverged MPLE
  caveat is printed under `Converged: false` too.
- **Every sampler's `burnin`/`interval` default to the dyad-scaled rule**
  (item 24e): `mh_sample`, `simulate_ergm`, `sample_networks` and `gof` take
  `burnin=nothing`/`interval=nothing`, resolved — like `mcmle` and the MPLE
  bootstrap already were — by ONE helper, `_mcmc_defaults(model)`
  (`20 × n_dyads`, `max(100, n_dyads ÷ 10)`), instead of the fixed
  `10000`/`1000` that was far too small for a 500-node network and wasteful
  on a 16-node one (on Florentine data: 2400/100, so a default `gof` is
  ~9× cheaper; on 500 nodes: 2.5 M / 12 475). The keyword names are
  unchanged; an explicit value is honoured as before (pinned). *Migration:*
  pass `burnin=10000, interval=1000` to reproduce a 0.1 simulation budget.
- **`ERGMResult` gained three trailing fields** (`vcov_fisher`, `mcmc_se`,
  `mcmc_convergence`); code constructing an `ERGMResult` positionally (none
  in the ecosystem) must supply them.
- **`Degree(0:2)` is ONE expanding term** (item 31). `Degree`, `IDegree` and
  `ODegree` hold `degrees::Vector{Int}`; `Degree(d)` is `Degree([d])`, and a
  vector or range of degrees is a single term that expands into one
  statistic per degree when the model is built (`_materialize`, exactly like
  a multi-level `NodeFactor`) — and in `summary_stats`. So
  `[Edges(), Degree(0:2)]` is a `Vector{<:AbstractERGMTerm}` (it used to be
  a `Vector{Any}` holding a `Vector{Degree}`, spliced by `_collect_terms`
  since WP1 and a raw `MethodError` before that) and fits with coefficient
  names `edges, degree0, degree1, degree2`, identical to three explicit
  `Degree(d)`. The unexpanded multi-degree term is a specification, not a
  statistic: `compute`/`change_stat` on it throw an `ArgumentError` ("…
  expands to degree0, degree1, degree2 when the model is built. Expand via
  ERGMModel …") and its `name` is the placeholder `"degree(0,1,2)"`.
  `Degree(0:2) == Degree([0, 1, 2])` (`==`/`hash` compare degrees). Code
  that indexed the old vector (`Degree(0:2)[1]`) must use `Degree(0)`.
- **MPLE is Newton–Raphson on the shared kernel, and Optim is gone** (item
  14). `_mple_fit` builds the binomial-row logistic likelihood with
  `Networks.logistic_derivatives(X, n_tot, n_one)` and maximizes it with
  `Networks.newton_fit` (step halving, converges to `tol=1e-8` on the
  objective; NaN standard errors with a warning when the negative Hessian is
  not positive definite, replacing a `try inv(H) catch NaN`). The private
  LBFGS objective and the hand-built information matrix are deleted along
  with the Optim.jl dependency. Agreement with R `ergm` on the provenanced
  flomarriage fixture (`edges + nodecov("wealth")`, asserted at 1e-6) is now
  **8.9e-13 on the coefficients** (was 6.6e-12 under L-BFGS) and 4.8e-8 on the
  standard errors (unchanged — that gap is R's own IRLS termination, not
  ours); the fixture, the `edges`-only analytic check and the MPLE bootstrap
  all pass unchanged.
- **`newton_fit` and `logistic_derivatives` are Networks.jl's.**
  `src/estimation/newton.jl` is deleted; ERGM does `import Networks:
  newton_fit, logistic_derivatives` and re-exports them, so `using ERGM` is
  unchanged and `ERGM.newton_fit === Networks.newton_fit` — one definition for
  ERGM, its variants, REM and Relevent. The numerics are pinned in Networks'
  testsuite; ERGM keeps the identity assertions, one smoke test each, and the
  `@allocated` bound on `logistic_derivatives`.
- **`_z_pvalues` is gone** as a definition (item 13): every Pr(>|z|) column
  (`mple`, `mcmle`, `mcmc_diagnostics`' Geweke p-values) goes through
  `Networks.z_pvalues`, the ONE erfc-based, floored, NaN-aware z → p helper of
  the ecosystem. `ERGM._z_pvalues` survives only as a deprecated `const` alias
  of it while TERGM and ERGMMulti still reach in.
- `mple` validates `se=` with the shared `Networks.check_se` (item 28), so the
  message has the ecosystem's one shape.
- `ergm` is a **`const` alias** of `fit_ergm` (`ergm === fit_ergm`; item 16),
  with a docstring — `?ergm` used to print "No documentation found" — and the
  `fit_ergm` docstring now lists every forwarded keyword and the `method`
  vocabulary.
- `simulate_ergm`, `sample_networks`, `mh_sample`, `mcmle`, `mple`, `ergm`,
  `ERGMModel`, `ERGMResult`, `summary_stats`, `has_dyad_dependent`,
  `confint`, `coeftable` docstrings carry runnable examples.

- `_random_network` (the MCMC starting network behind `mcmle`, `simulate_ergm`,
  `sample_networks` and `gof`) now reads its default starting density with
  `network_density(net; missing=:face)`. Networks.jl's `network_density` refuses
  a masked network by default (panel 2026-09, item 4); a starting density read
  at face value is defensible here because the sampler already conditions on
  the masked dyads' face values under `missing=:condition_on_face`, and the
  masked-network tests would otherwise throw from inside `_random_network`.

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

- **MPLE "converged" silently on a perfectly separated mixed-sign
  statistic that R refuses** (round 3, major). `_boundary_columns` skipped
  any column with entries of both signs, so `ergm(net, [Edges(),
  NodeCov(:x)])` on the 8-node fixture design (x = −4..3, tie iff x_i + x_j
  ≤ 0; nodecov.x = −50 is its smallest attainable value) returned coef
  `[20.8, −41.8]`, SE `[19177, 28256]`, p = 0.999, `converged == true`,
  `is_exact == true` and an empty `approximations` — no warning at all —
  where R 4.6.1/ergm 4.12.0 warns "The MPLE does not exist!". The
  attainable-range test (see Changed) now drops the column with R's
  sentence, and a second guard, `_separated`, catches separation by a
  *combination* of columns that no single-column test can see (R decides
  it with an LP; ERGM.jl reads the two signatures an asymptote leaves on
  Newton — a perfectly predicted dyad at a fitted probability within 1e-8
  of 0/1 while the next Newton step is still > 1e-3 of ‖θ‖ — both required,
  so an extreme but well-determined dyad on a large network is not
  flagged): `mple` warns "the MPLE does not exist (perfect separation) …
  R ergm warns \"The MPLE does not exist!\"", returns `converged ==
  false`, `is_exact == false`, the caveat under `Converged: false` and in
  `approximations`; `mcmle` refuses to start from it. Both designs are
  provenanced fixture rows carrying R's warning text.
- **Parametric-bootstrap standard errors were NaN for every documented
  dyad-dependent model** (round 3, major — a regression of the round-2 drop
  semantics). `refit` called `_mple_fit`, which fixes a boundary statistic
  at `-Inf`; a simulated 16-node replicate with no triangle / no
  shared-partner edge is routine, its `-Inf` row entered
  `Networks.bootstrap_cov`'s `cov`, and `ergm(flo, [Edges(), Triangle()];
  se=:bootstrap)` — the README's and the estimation guide's "honest
  uncertainty" recipe — returned `stderror = [0.37, NaN]`, an all-NaN
  `vcov`, NaN `confint`, with only a misleading `mple: observed
  statistic(s) triangle …` warning per replicate (about a simulated
  network) and nothing in `approximations`. The refits now run silently
  (`_mple_fit(...; warn=false)`), a replicate without a finite MPLE
  (boundary or separated) returns a NaN row, those rows are excluded from
  the covariance (≥ 2 finite refits required, `ArgumentError` otherwise),
  ONE warning names how many of `n_boot` were excluded and says it is
  about the simulated replicates, `show` and `approximations` carry the
  same sentence, and `fit.boot_replicates` holds every row. `se=:bootstrap`
  on a point estimate that itself has a `±Inf` coefficient is refused (no
  network can be simulated at −Inf). The bootstrap testset now covers
  `[Edges(), Triangle()]` and the guide's `[Edges(), GWESP(0.5),
  NodeMatch(:gender)]`, asserting finite SEs, the single warning and the
  record; the guide shows the real output.
- **Terms guide overstated coverage** (round 3, minor): "ERGM.jl covers
  statnet's one-mode, fixed-decay term surface" was false — the sender/
  receiver attribute terms, the directed triadic terms, the esp/dsp/nsp
  count terms and several others have no counterpart. The guide now says
  "implements the terms listed in this guide", and the absent families are
  one bullet in the terms-guide admonition, the README "Not implemented"
  list and the Known limitations below.
- **Terms guide `summary_stats` comment** (round 3, minor) showed
  `(edges = 12.0, gwesp.fixed.0.5 = 4.3, …)` for the page's directed
  3-node example, whose real output is `(edges = 1.0,
  var"gwesp.OTP.fixed.0.5" = 0.0, …)`; the comment is the real output.
- **CHANGELOG Added entry for item 24c** still described "MCMC %" as the
  share of the standard-error *variance* (round 3, minor); it now points
  at the round-2 Changed entry with R's definition.
- **CLAUDE.md quoted 23 regression pins** (round 3, minor); the benchmark
  gate has three testsets (undirected/directed `change_stat`, and
  `change_stat_all!`/`_change_stat_tuple`/`_mple_rows!` at 36 statistics),
  named instead of counted.
- **Docstring examples that did not run**: `fit_ergm`'s used `Random.Xoshiro`
  without `using Random`; `supports_missing(::AbstractERGMTerm)`'s referred to
  an undefined `ObservedEdges` (now defined in the example); the
  `Degree(0:2)` and `NodeMatch(diff=true)` idiom sketches referred to an
  undefined `net` (now ```jl sketches, with runnable examples beside them).
  The `_materialize` generic fallback was defined once, documented; a stray
  second definition is gone.
- **The missing-dyad refusal names the policy ERGM accepts** (panel 2026-09,
  item 5). `mcmle`, `simulate_ergm`, `gof`, `mh_sample` and `sample_networks`
  used to reject a masked network with the shared message's "pass
  `missing=:face`" bullet — and then throw `invalid missing-dyad policy :face`
  when the user did. `_guard_missing` now calls `require_observed(...;
  face_ok=false)` and appends a bullet naming `missing=:condition_on_face`;
  a testset asserts every refusal contains `:condition_on_face` and not
  `missing=:face`. The README and the estimation guide, whose masked-network
  examples the July guard had broken (`simulate_ergm`/`mcmle` on the masked
  page-wide `net`), mask a `copy(net)` and show the opt-in once.
- The "until v0.5" note on the private trait aliases in `traits.jl` said
  v0.2.

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

- **The whole-model fills are allocation-free at any number of statistics**
  (round 2, major). `compute_all`, `change_stat_all`, `change_stat_all!`
  and the MPLE row key built the statistic tuple with `Base.map` over the
  term tuple, which is unrolled only below 32 elements and from 32 on
  takes the `Any32` path — a `Vector{Any}`, one box per Float64, a splat:
  the graders measured 0 B per MH step at 31 statistics and 28,720 B at 32
  (35,200 B at 36; 201 MB for the MPLE sweep of a 120-node network at
  p = 32). A `NodeMix` on an 8-level attribute alone is 35 cells, so the
  "0 B per step" claim did not cover realistic models. The fills now go
  through the `@generated` `_compute_tuple`/`_change_stat_tuple` — one
  statically dispatched call per term for ANY term count — and
  `_mple_rows!` keys on the same stack tuple. Pinned at 36 statistics
  (`edges + NodeMix(8 levels)`) in "MH kernel is allocation-free per step",
  "MPLE design build allocates O(unique rows)" and
  `benchmark/regression_tests.jl`.

- **Allocation pins run in `Pkg.test()` and in CI** (item 7). The
  `change_stat` 0-byte pins for `Edges, Triangle, GWESP(0.5), GWESP(0.0),
  GWDSP, Kstar, TwoPath, GWDegree, Degree, Mutual, OStar, IStar, GWIDegree,
  GWODegree, IDegree, ODegree` on 500-node Erdős–Rényi networks ("Hot paths
  are allocation-free") and the `mh_toggle!` per-step pin (0 bytes between
  `burnin=10_000` and `20_000` on a toy state; the binary-network adapter's
  residual — Graphs.jl growing adjacency vectors inside `add_edge!` on a
  freshly copied network, ~1 B/step and decaying, identical under the old
  loop — is bounded at 4 B/step) are in `test/runtests.jl`;
  `benchmark/Project.toml` instantiates from scratch (23/23 regression pins,
  SCALING ratios 0.98/0.66/1.11 < 3) and `.github/workflows/CI.yml` runs the
  tests with `JULIA_NUM_THREADS=4` on the `'1'`/ubuntu cell and then
  instantiates the benchmark environment and runs
  `benchmark/regression_tests.jl` there.
- **PrecompileTools workload rebuilt on inline networks** (item 18): a
  7-vertex undirected and a 6-vertex directed network with a `:grp`
  attribute (no dataset I/O), both directedness type parameters, MPLE +
  `coeftable`/`confint`/`show`, an `edges + gwesp` MCMLE (plus a
  `bridge_rungs=0, n_chains=2` one), `mcmc_diagnostics`, `gof`,
  `simulate_ergm`, under a `ConsoleLogger(devnull)`. Measured 2026-09-09
  (fresh process, warm depot, one machine; before = the WP1 workload on the
  Florentine data): `using ERGM` 1.07 → 1.2 s; first MPLE 0.40 → 0.01 s on the
  workload's formula and 0.46 → 0.35–0.6 s on a formula with a new
  combination of term types (per-formula `TermSet` specialization); first
  `mcmle` 1.06 → 1.5 s (`--trace-compile` shows only Base logging/Printf
  methods on the warning path, no ERGM method); first `gof` 0.08 → 0.2 s;
  first `simulate_ergm` 0.01 s; precompilation 17 → 22 s. The panel's
  pre-WP1 baseline was 8.0 s / 6.7 s for `using ERGM` / first `ergm`.
- **MPLE is 2–4x faster** on the Newton kernel: Florentine marriage
  (`edges + nodecov`) 0.065 ms → 0.017 ms per fit; a 200-node Erdős–Rényi
  network at density 0.05 with `edges + gwesp(0.5) + nodecov` 21.2 ms →
  11.3 ms (both minimum of repeated warm runs, single thread).
- **`_mple_data` no longer allocates per dyad** (item 26): the row table is
  keyed on an `NTuple{p,Float64}` built on the stack instead of a fresh
  `Vector{Float64}` per dyad, so the sweep allocates nothing and the build
  allocates O(unique rows). Measured on the 200-node network above:
  `edges + gwesp + nodecov` (19 900 unique rows) 4.57 MB → 4.38 MB — what
  remains is the row table's growth and the returned `X`/`n_tot`/`n_one`,
  ≈220 B per unique row; `edges + nodematch` (2 unique rows over 19 900 dyads)
  1.59 MB → 1.07 KB. Pinned by the "MPLE design build allocates O(unique
  rows)" testset (`0` bytes for the sweep with a pre-sized table; `< 256` B
  per unique row overall; `< 4 KB` for the compressed design).
- **PrecompileTools workload** (item 18): a seeded, silenced `mple`, `mcmle`,
  `simulate_ergm`, `gof`, `coeftable`/`confint`/`show` on the Florentine data
  runs at precompile time, so the estimation, sampling and GOF paths are in
  the package image. Measured on one machine (single thread, warm depot):
  `using ERGM` 0.6 s → 1.0 s (larger image), first `mple` 4.9 s → 1.5 s,
  first `mcmle` 1.2 s → 0.7 s, first `simulate_ergm` 0.4 s → 0.5 s — the
  first fit end-to-end 5.6 s → 2.5 s. The residual is per-formula
  specialization of the tuple-backed `TermSet` (a formula with a new
  combination of term types compiles its own change-statistic sweep).
  Precompilation itself takes ~13 s instead of ~1 s.

- **`logistic_derivatives(X, y; offset)` — the shared, allocation-free
  logistic `(ll, grad, hess)` builder** (review finding 15), exported next to
  `newton_fit` (both now hosted in Networks.jl and re-exported, see Changed). `ERGMMulti`'s MPLE over the within-layer dyads, `TERGM`'s CMPLE
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

### Known limitations

What a user of R `ergm` will not find in 0.2.0, and the exact behaviour they
get instead (also in the README's "Not implemented" section and the terms
guide):

- **Curved terms** — `gwesp(fixed=FALSE)`, `gwdegree(fixed=FALSE)`,
  `gwdsp(fixed=FALSE)` with the decay estimated: not implemented. Every
  geometrically weighted term takes a fixed decay ≥ 0; there is no
  curved-family (Hunter–Handcock) MCMLE.
- **Bipartite terms and two-mode networks** — `b1degree`, `b2degree`,
  `b1factor`, `b2factor`, `b1nodematch`, `b1star`, …: not implemented.
  `ERGMModel`/`fit_ergm` refuse a two-mode network (`bipartite=k` flag or
  `BipartiteNetwork`) with an `ArgumentError` naming the gap.
- **`constraints=`** — `edges`, `degrees`, `blockdiag`, `observed`, …: not
  implemented; a non-empty `constraints=` on `ERGMFormula` throws an
  `ArgumentError`.
- **`offset()` terms** — no counterpart.
- **Attribute terms with NA values** — refused at model construction with an
  `ArgumentError` naming the vertices without a value (statnet refuses too).
- **Self-loops** — a network *containing* a loop is refused at model
  construction with an `ArgumentError` naming the vertices (R warns and
  fits regardless); a `loops=true` network with no loop is modelled as
  loop-free. The statistics would count a loop that `nobs`, the
  pseudo-likelihood, the proposal and the simulations never see.
- **Absent term families** — sender/receiver attribute terms `nodeicov`/`nodeocov`/`nodeifactor`/`nodeofactor`; the directed triadic terms `ttriple`/`ctriple`/`transitiveties`/`cyclicalties`/`asymmetric`; the `esp`/`dsp`/`nsp` count terms; `isolates` (only as `Degree(0)`), `balance`, `cycle`, `smalldiff`, `nodematch(keep=)`
  have no counterpart (nothing is silently mis-fit; there is no term to
  call). Use `IStar`/`OStar`, `Triangle` (= `ttriple + ctriple` on a
  directed network), `GWESP`/`GWDSP` and `NodeFactor`/`NodeCov`.
- **`drop=FALSE`** — a statistic at the boundary of its attainable range
  has no finite estimate; `mple` does R's default `drop=TRUE` (`∓Inf`, SE
  0, the rest fit on the untouched dyads, warned; `dof`/AIC/BIC count the
  finite coefficients and the untouched dyads, R's `logLik` df/nobs) and
  `mcmle` refuses with the same sentence. There is no fit of such a model
  with a finite coefficient, and no R-style `-Inf` MCMLE.
- **R's exact `mple.existence` test** — R decides whether the MPLE exists
  with a linear program; ERGM.jl has no LP solver and detects separation
  by a *combination* of statistics from the two signatures an asymptote
  leaves on the Newton iteration (a perfectly predicted dyad at a fitted
  probability within 1e-8 of 0/1 while the next Newton step is still
  > 1e-3 of ‖θ‖). A design whose asymptote is approached more slowly than
  that is not flagged; one that is flagged is returned with `converged ==
  false` and R's sentence, never silently.
- **Simulation and GOF under missing data** — a simulated network has a
  value at every dyad, so `simulate_ergm`, `sample_networks`, `mh_sample`
  and `gof` on a masked network offer only the warned
  `missing=:condition_on_face` (masked dyads frozen at their stored face
  value); `missing=:mle` is refused with an explanation. Estimation under
  missing data is implemented — `mcmle(...; missing=:mle)` (R's
  missing-data MLE) and the available-case MPLE — and `gof` of an `:mle`
  fit therefore compares face-value observed panels against face-value
  simulations, which must be asked for in writing.
- **MCMLE returns the MPLE unchanged when the MPLE already passes the
  convergence tests** (N7, documented and pinned): the check runs before
  the first Newton update, so on the unmasked flomarriage `edges + gwesp`
  the point estimate has zero seed-to-seed variance; statnet always takes at
  least one step. Defensible (the moment condition holds to within MC
  error) and inside R's noise, but not produced the same way.

## [0.1.0] - 2026-02-09

Initial release: ERGM terms (structural, nodal, dyadic), MPLE and MCMLE
estimation, MCMC simulation, and goodness-of-fit/MCMC diagnostics.
