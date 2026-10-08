# ERGM.jl


[![Network Analysis](https://img.shields.io/badge/Network-Analysis-orange.svg)](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl)
[![Build Status](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGM.jl/dev/)
[![Julia](https://img.shields.io/badge/Julia-1.12+-purple.svg)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="docs/src/assets/logo.svg" alt="ERGM.jl icon" width="160">
</p>

Exponential Random Graph Models for Julia.

## Overview

ERGM.jl provides tools for fitting, simulating, and diagnosing Exponential-Family Random Graph Models (ERGMs). ERGMs are statistical models for network structure that express the probability of observing a network as a function of network statistics.

This package is a Julia port of the R `ergm` package from the StatNet collection.

## Installation

Requires Julia 1.12 or newer. The packages are not yet registered.

**Recommended: the ecosystem workspace.** It clones every package side by
side, develops them together in one environment, and adds the packages the
examples also use (CSV, DataFrames, Distributions, Graphs, StatsAPI,
StatsBase):

```bash
mkdir network-analysis && cd network-analysis
git clone https://github.com/statistical-network-analysis-with-Julia/statistical-network-analysis-with-Julia.github.io
julia statistical-network-analysis-with-Julia.github.io/tools/prepare_workspace.jl "$PWD" --clone
julia --project=.snippet-env
```

**Only this package, in your own environment.** Add its dependency first,
in this order:

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/NetworkCore.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGM.jl")
```

The examples below load only `NetworkCore`, `ERGM` and the `Random`
standard library.

## Features

- **Model terms**: Structural, nodal, and dyadic covariate terms
- **Estimation**: MPLE (fast) and MCMLE (full likelihood)
- **Simulation**: MCMC network simulation
- **Diagnostics**: Goodness-of-fit testing, MCMC diagnostics

## Quick Start

```julia
using NetworkCore
using ERGM

# Observed network: Padgett's Florentine marriage ties (bundled dataset)
net = load_dataset(:florentine_marriage)

# Define model terms. Attribute-based terms are validated at model
# construction: a typo'd attribute name throws an ArgumentError listing
# the attributes that do exist.
terms = [
    Edges(),
    NodeCov(:wealth)
]

# Fit the model: dyad-independent, so the default method=:auto fits the
# MPLE, which is the exact MLE (a dyad-dependent formula gets the MCMLE)
result = ergm(net, terms)
println(result)

# Simulate from fitted model
sim_nets = simulate_ergm(result; n_sim=100)
```

## Model Terms

### Structural Terms
<!-- skip-check -->
```julia
Edges()              # Edge count (density)
Mutual()             # Reciprocated edges (directed only)
Triangle()           # Triangle count
Kstar(k)             # k-star count (undirected only, as in R)
OStar(k)             # out-k-stars / in-k-stars (directed only;
IStar(k)             #   statnet ostar/istar)
TwoPath()            # Two-path count
Degree(d)            # Vertices with degree exactly d (undirected only);
IDegree(d)           #   in-degree / out-degree variants for directed
ODegree(d)           #   networks. Degree(0:2) is ONE term that expands
                     #   to degree0, degree1, degree2 when the model is built
GWESP(decay)         # Geometrically weighted ESP, decay ≥ 0
                     # (directed: type=:OTP|:ITP|:OSP|:ISP|:union,
                     #  statnet dgwesp semantics, default :OTP; the directed
                     #  coefficient is R's "gwesp.OTP.fixed.<decay>")
GWESP(decay; fixed=false)     # CURVED: the decay is estimated (statnet
GWDegree(decay; fixed=false)  #   gwesp/gwdegree(fixed=FALSE); mcmle only;
                              #   coefficients "gwesp", "gwesp.decay")
GWDSP(decay)         # Geometrically weighted dyadwise shared partners
                     # (all dyads, tied or not; same directed types, every
                     #  one summed over ordered dyads as in R's dgwdsp)
GWDegree(decay)      # Geometrically weighted degree (undirected only,
                     #   as in R; coefficient "gwdeg.fixed.<decay>")
GWIDegree(decay)     # Geometrically weighted in-degree (directed only)
GWODegree(decay)     # Geometrically weighted out-degree (directed only)
GWNSP(decay)         # Geometrically weighted non-edgewise shared partners
                     #   (statnet gwnsp/dgwnsp; = GWDSP − GWESP, same types)
ESP(k)               # Ties with exactly k shared partners (statnet esp/desp;
                     #   ESP(0:2) expands to esp0, esp1, esp2; directed types
                     #   :OTP|:ITP|:OSP|:ISP, labelled "esp.OTP1")
Concurrent()         # Vertices with degree ≥ 2 (undirected only, statnet concurrent)
DegRange(from, to)   # Vertices with from ≤ degree < to (undirected; to=Inf
IDegRange(from, to)  #   by default; "deg2+", "deg1to3"); in-/out-degree
ODegRange(from, to)  #   variants for directed networks
MeanDeg()            # Mean degree (statnet meandeg; dyad-independent)
Density()            # Density, edges / dyads (statnet density)
TransitiveTies()     # Ties closing a transitive two-path i→k→j (statnet
CyclicalTies()       #   transitiveties) / a cyclical one j→k→i (cyclicalties)
Sender(); Receiver() # p1/p2 sender and receiver effects (directed only):
                     #   one statistic per vertex but the first, "sender2", …
ERGM.TriadCensus()   # Triad census (statnet triadcensus; 15 directed MAN types
                     #   or 3 undirected types, type 0 dropped). Public but not
                     #   exported: Siena.jl exports a GOF TriadCensus
Offset(term, coef)   # statnet offset(term) with offset.coef: the coefficient is
                     #   fixed, not estimated ("offset(edges)"); coef = -Inf is
                     #   a structural zero (those ties are impossible), +Inf
                     #   forces them
```

Every geometrically weighted term takes a **fixed decay ≥ 0** by default —
`GWESP(0.0)` is statnet's `gwesp(0, fixed=TRUE)`, and note that R's own
default is `fixed=FALSE`: write `GWESP(0.5; fixed=false)` for R's
`gwesp(0.5)` — and its coefficient
carries **R's label**: an integer-valued decay prints without a decimal
point (`gwesp.fixed.0`, `gwesp.fixed.1`, `gwdeg.fixed.1`), so a fit can be
compared with a statnet fit by coefficient name — on a *directed* network
R names the default shared-partner terms `gwesp.OTP.fixed.<decay>` /
`gwdsp.OTP.fixed.<decay>`, and so does every model, `summary_stats` and
`name(term, net)` here (`absdiff(pow=2)` is `absdiff2.<attr>`, likewise
R's). `Kstar`/`GWDegree`/`Degree` on a directed network are refused with an
`ArgumentError` naming the directed variant, as R ergm refuses
`kstar`/`gwdegree`/`degree` there.

> **Changed in 0.2**: on *directed* networks, `GWESP(decay)` (and `GWDSP`)
> previously counted either-direction ("union") shared partners while
> emitting statnet's OTP label `gwesp.fixed.<decay>` — a silently different
> model. The default is now statnet-compatible `:OTP`; the old statistic is
> available as `GWESP(decay; type=:union)` under the distinct label
> `gwesp.union.fixed.<decay>`. Directed models fit with 0.1 produce
> different coefficients when refit; undirected GWESP is unchanged. Also
> changed: `Kstar`/`GWDegree` on a directed network used to compute
> *out*-stars / out-degree weights silently under the undirected label (now
> refused; use `OStar`/`IStar`/`GWODegree`/`GWIDegree`), `GWDegree` was
> labelled `gwdegree.fixed.<decay>` (now R's `gwdeg.fixed.<decay>`), and a
> vertex without an attribute value was zero-filled (now refused, as in R).
> See [CHANGELOG.md](CHANGELOG.md).

### Nodal Terms
```julia
NodeFactor(:attr)           # Categorical main effect: one statistic per level,
                            # first (sorted) level dropped as the reference
                            # (statnet nodefactor; base=0 keeps all levels)
NodeCov(:attr)              # Continuous node attribute (set on EVERY vertex —
                            # a vertex without a value is refused, as R refuses NA)
NodeMatch(:attr)            # Uniform homophily on attribute
NodeMatch(:attr; diff=true)  # Differential (per-level) homophily, as in
                             # R nodematch(diff=TRUE): one statistic per
                             # level (`levels=` selects, `level=` one)
NodeMismatch(:attr)         # Mismatched-edges count (heterophily)
NodeMix(:attr)              # Mixing-matrix cell counts (statnet nodemix;
                            # first cell dropped as the reference)
AbsDiff(:attr)              # Absolute difference effect
```

> **Changed in 0.2**: `NodeFactor(attr)` previously produced a single
> statistic counting endpoint appearances across *all* levels — collinear
> with `Edges()` by construction. It now matches statnet: one statistic
> per level with the first (sorted) level as the reference. Pass `base=0`
> for the old all-levels behavior (as separate per-level statistics).

> **Changed in 0.2**: `NodeMatch(attr; diff=true)` previously counted
> *mismatching* dyads — the opposite of R, where `nodematch(diff=TRUE)`
> means differential (per-level) homophily. `diff=true` is now
> R-compatible per-level homophily (`nodematch.<attr>.<level>` naming,
> expanded over every level as R does); the old mismatch statistic
> moved to the new `NodeMismatch(attr)` term. See
> [CHANGELOG.md](CHANGELOG.md) for the full 0.2 migration notes.

### Dyadic Terms
<!-- skip-check -->
```julia
EdgeCov(matrix)      # Edge covariate
```

## Model Fitting

`ergm` and `fit_ergm` are one function (`ergm === fit_ergm`): the statnet
name and the ecosystem's `fit_<model>` name.

**The default estimator is R's.** `ergm(net, terms)` takes `method=:auto`:
a formula with no dyad-dependent term is fitted by MPLE, which is then the
exact MLE (a logistic regression), and a formula with any dyad-dependent
term (`Triangle`, `GWESP`, `Mutual`, `Degree`, …) by MCMLE — what R's
`ergm()` does. `method=:mple` and `method=:mcmle` choose explicitly; the
MPLE of a dyad-dependent formula is the fast approximation below.

```julia
# The default: here dyad-independent, so the MPLE, which is the exact MLE
result = ergm(net, terms)

# Maximum Pseudo-Likelihood (fast; exact for dyad-independent models)
result = ergm(net, terms; method=:mple)

# Monte Carlo MLE (slower; the full likelihood). Every Monte-Carlo draw
# flows from `rng`; `n_chains` splits each iteration's sample over
# independent chains (parallel with threads, identical at any thread count);
# `bridge_rungs=0` skips the path-sampled log-likelihood (loglik/AIC/BIC
# become NaN, coefficients and SEs are bit-identical, ~70 % faster single-threaded)
using Random
result = ergm(net, terms; method=:mcmle, n_samples=1000, n_chains=2, rng=Xoshiro(1))

# The full StatsAPI surface
coef(result)         # Coefficients
stderror(result)     # Standard errors (MCMLE: Fisher + Monte-Carlo component)
vcov(result)         # Covariance matrix
confint(result)      # Normal-theory 95% limits (level= to change)
coeftable(result)    # The R-style table `show` prints, as a CoefficientTable
loglikelihood(result), aic(result), bic(result), nobs(result), dof(result)
mcmc_se(result)      # The Monte-Carlo part of each SE; `show` prints R's "MCMC %"
                     #   (round(100·(se − se_fisher)/se), R's own definition); zeros for MPLE
```

The MCMLE follows statnet's algorithm and defaults:

- **The sampler** is R ergm 4's by default. `mcmle` proposes with R's
  `SPDyad` (what its `MCMC.prop = ~sparse + .triadic` selects for every
  one-mode network): statnet's **tie/no-tie (TNT)** proposal — half the
  proposals remove an existing tie, half toggle a random dyad — mixed with a
  shared-partner-focused proposal, with the exact Hastings correction
  (`proposal=:spdyad`). On faux.mesa.high's gwesp model it gives about 6×
  TNT's effective sample size per draw at about twice the cost per toggle.
  `proposal=:tnt` is plain TNT — the default of the other samplers
  (`simulate_ergm`, `gof`, `mh_sample`, `sample_networks`, the MPLE
  bootstrap), where R uses SPDyad too; both are exact samplers of the same
  model — and `proposal=:random` the uniformly random toggle of ERGM.jl 0.1.
- **ESS-adaptive sampling** (`effective_size=64`, R's
  `MCMLE.effectiveSize`, the `mcmle` default): instead of a fixed
  `n_samples` draws per iteration, the chain is continued from the previous
  iteration and extended, its thinning interval doubling, until the
  effective sample size of the sampled statistics reaches the target.
  Fixed-size TNT sampling remains one keyword pair away:

  ```julia
  result = ergm(net, terms; method=:mcmle, proposal=:tnt, effective_size=nothing)
  ```
- **Every iteration takes a Monte-Carlo Newton step** (Hummel step length),
  the first included, so a fit never returns the MPLE unchanged.
- **The stopping rule is R's `confidence` equivalence test**
  (`termination=:confidence`): the fit converges when the 99 %
  Monte-Carlo confidence region of the estimating equation lies inside a
  tolerance region of 0.1 of the statistics' variance; when it does not,
  the next iteration's sample is enlarged (up to `max_n_samples`, 16 ×
  `n_samples` by default), as R does. `maxiter` defaults to R's 60.
  `termination=:hotelling` restores the pre-0.2 rule (max |t| < 0.1 and a
  non-significant Hotelling test), which could not be met under Monte-Carlo
  noise. On faux.mesa.high, `edges + nodematch("Grade") + nodematch("Race")
  + gwesp(0.25, fixed=TRUE)` (and the same with `gwdegree(0.5)`) converges
  at the defaults and matches R ergm within R's own seed spread (pinned by a
  provenanced fixture).

**Curved models.** `GWESP(decay; fixed=false)` and
`GWDegree(decay; fixed=false)` estimate the decay together with the
coefficients — the curved exponential-family MCMLE of Hunter & Handcock
(2006). The fit reports R's two coefficients per curved term (`gwesp`,
`gwesp.decay`), and the result simulates and is assessed like any other fit:

```julia
fmh = load_dataset(:faux_mesa_high)
curved = ergm(fmh, [Edges(), NodeMatch(:Grade), NodeMatch(:Race), GWESP(0.25; fixed=false)];
              method=:mcmle, rng=Xoshiro(1))
coef(curved)       # ≈ [-6.36, 1.97, 0.27, 1.35, 0.38]: R ergm's estimates
```

An MCMLE fit that exhausts `maxiter` is loud: it warns with the stopping
rule's verdict (the equivalence test's p-value) and the step length,
prints the caveat under `Converged: false`, lists it in
`NetworkCore.approximations(result)` and keeps the numbers in
`result.mcmc_convergence` and `result.termination`; continue it with
`ergm(net, terms; method=:mcmle, init=coef(result))`.

Model construction fails loudly on user errors: attribute-based terms whose
vertex attribute does not exist on the network, and intrinsically directed
terms (e.g. `Mutual()`) on undirected networks, both throw an
`ArgumentError` (as in R ergm) instead of silently fitting a wrong model.
So do the slips a migrant makes at the call site — `fit_ergm(terms, net)`
(swapped arguments), `[Edges, Triangle()]` (a term *type* in the list) — each
with a message saying what to write instead. A single term needs no brackets
(`ergm(net, Edges())`), and `[Edges(), Degree(0:2)]` splices the expanded
degree terms in, as statnet's `edges + degree(0:2)`.

### Honest MPLE uncertainty

For models with dyad-**dependent** terms (`Triangle()`, `GWESP(...)`, ...)
the inverse-Hessian MPLE standard errors are anticonservative — the
pseudo-likelihood treats dependent dyads as independent observations; in
simulation their 95 % Wald intervals covered 0.71–0.85 of the time. So,
**by default, an MPLE fit of a dyad-dependent formula reports its point
estimates and naive standard errors but no z values, p-values or
intervals**: those columns are `NaN`, `show` says why, and `confint`
refuses with an `ArgumentError`. For calibrated inference use the
parametric bootstrap (coverage 0.945–0.970 in the same simulation) or
refit with `method=:mcmle`:

```julia
result = ergm(net, terms; method=:mple, se=:bootstrap, n_boot=100)
```

`se=:hessian`, passed explicitly, is the written opt-in to R's naive Wald
table (printed with the pseudo-likelihood caveat).

A simulated replicate on which the MPLE does not exist (no triangle under
`Triangle()`, no shared-partner edge under `GWESP`, …) is excluded from the
bootstrap covariance — one warning says how many of `n_boot` were, the
result records it (`approximations(result)`, `result.boot_replicates`) —
so the standard errors are finite. They are then conditional on a finite
refit: the excluded replicates are the extreme ones, so the standard errors
are biased downward. `se=:bootstrap` refuses a point estimate that itself
has a `±Inf` coefficient.

For dyad-independent models the pseudo-likelihood is the true likelihood
and the default SEs, z values, p-values and intervals are exact (no caveat
is printed).

MCMLE's reported log-likelihood (hence AIC/BIC) is estimated by an
ergm-style path-sampling (bridge) ladder from a dyad-independent reference
to the fitted coefficients (`bridge_rungs` keyword, Simpson's rule), which
is far more accurate than one-jump importance sampling; the exact reference
normaliser includes `θ'g(∅)`, so a user term whose value on the empty
network is not zero gets the right log-likelihood.

### Missing (unobserved) dyads

If dyads of the network are masked as missing with NetworkCore.jl's
`set_missing_dyad!` (tie status unobserved — distinct from "no tie"):

- **MPLE** excludes masked dyads from the design matrix: they are not
  observed responses, so they contribute no logistic-regression row and
  `nobs` decreases accordingly. Their face values still condition the
  change statistics of the observed dyads. This is the available-case
  pseudo-likelihood, a principled treatment: `supports_missing(mple)` is
  `true` and `mple` needs no keyword.
- **MCMLE refuses a masked network by default and offers two policies.**
  `missing=:mle` is **missing-data maximum likelihood** — what R `ergm()`
  does with NA ties (Handcock & Gile 2010): the masked dyads are integrated
  out of the likelihood by a second, *constrained* MH chain that toggles
  only the masked dyads and estimates `E[g(Y) | Y_obs]`, which replaces the
  observed statistics as the MCMLE target; the Fisher information is
  `Var[g] − Var[g | Y_obs]`, both chains contribute Monte-Carlo error to the
  standard errors, and the log-likelihood is two path-sampling bridges.
  `missing=:condition_on_face` instead holds each masked dyad fixed at its
  stored face value (never toggled, scored as recorded) — the model
  *conditional on the face values*, a different estimand, right only when
  the stored values are true by construction; it is warned. The default
  `missing=:error` throws the shared ecosystem `ArgumentError`, whose last
  bullets name the two opt-ins: `missing_policies(mcmle) == (:error,
  :condition_on_face, :mle)`. The generic `:face` is not accepted. The
  provenanced `test/fixtures/flomarriage_missing_ergm.toml` pins `:mle`
  against R `ergm` on a flomarriage with four NA dyads (agreement within
  R's own seed-to-seed spread) and the available-case MPLE against R's
  dyad-independent fit with NA dyads (1e-6).
- **Simulation and GOF refuse a masked network by default, and refuse
  `:mle`.** A simulated network has a value at every dyad, so "simulation
  under missing data" is not defined: `simulate_ergm`, `sample_networks`,
  `mh_sample` and `gof` accept only `:condition_on_face` (warned) and reject
  `missing=:mle` with a message saying so. `mh_sample` exposes the two
  MCMLE chains as `toggleable=:all` / `toggleable=:masked`.

```julia
net_na = copy(net)
set_missing_dyad!(net_na, 3, 4)            # 3–4 was not measured
fit_na = ergm(net_na, terms)               # MPLE: 3–4 excluded from the pseudo-likelihood
nobs(fit_na)                               # one fewer observation
fit_na.missing_method                      # :available_case

# Missing-data maximum likelihood (R's treatment of NA ties)
fit_mle = ergm(net_na, [Edges(), GWESP(0.5)]; method=:mcmle, missing=:mle,
               rng=Xoshiro(1))
fit_mle.missing_method                     # :mle

# Simulation from a masked fit must be asked for in writing (and is warned):
sims_na = simulate_ergm(fit_na; n_sim=10, missing=:condition_on_face)
```

## Simulation

```julia
# Simulate networks from fitted model
sim_nets = simulate_ergm(result; n_sim=100)

# Simulate from parameters directly
model = ERGMModel(ERGMFormula(terms), net)
sim_nets = sample_networks(model, coef(result); n_sim=100)

# Low-level single-chain sampler: sampled sufficient statistics
# (and optionally networks) from one parameterized MH chain
using Random
out = mh_sample(model, coef(result); n_samples=500, rng=Xoshiro(1))
out.stats            # n_samples × p matrix of sampled statistics
```

Burn-in and thinning default to ONE dyad-scaled rule shared by every
sampler and by `mcmle` (`burnin = 20 × n_dyads`, `interval = max(100,
n_dyads ÷ 10)`; `ERGM.Extension.mcmc_defaults(model)` shows the numbers); pass
`burnin=`/`interval=` to override. The samplers propose with statnet's
tie/no-tie proposal (`proposal=:tnt`, exact Hastings correction) unless
`proposal=:spdyad` (TNT mixed with a shared-partner-focused proposal, R's
default and `mcmle`'s) or `proposal=:random` is passed.

All sampling and fitting functions accept an `rng::AbstractRNG` keyword;
runs with the same seed are exactly reproducible. `sample_networks`,
`simulate_ergm`, `gof` and `mcmle` split their draws over independent chains
run on separate threads (`n_chains` keyword), seeded deterministically from
the caller's `rng`, so results are also independent of the thread count.
An error raised inside a chain reaches the caller as itself (an
`ArgumentError` stays an `ArgumentError`), not wrapped in a
`TaskFailedException`.

Every sampler is built on the exported Metropolis kernel `mh_toggle!(rng, θ,
delta, propose, change!, apply!, on_sample; burnin, interval, n_samples)`:
it owns the accept/reject arithmetic, burn-in and thinning, and takes the
proposal, the change statistics and the state mutation as callables — so the
ERGM variants (TERGM, ERGMMulti, ERGMRank) sample their own state types with
the same, allocation-free, bit-reproducible loop.

## Goodness-of-Fit

`gof` is a method of the ecosystem-wide `NetworkCore.gof` generic — the same
verb works on every fitted model in the ecosystem. For directed networks
the degree comparison is split into in- and out-degree distributions
(`:idegree`/`:odegree`), as in R ergm:

With the default `interval`, GOF thinning is effective-sample-size aware:
when the simulated networks' statistics carry fewer than `n_sim/2`
effective draws, the networks are redrawn at an interval of twice the
measured autocorrelation time of the model's statistics (at most 64× the
default; on faux.mesa.high the lag-1 autocorrelation of the GOF draws falls
from 0.66 to about 0), and fewer than `n_sim/4` is warned about.

```julia
# GOF diagnostics (burn-in/interval default to the dyad-scaled rule above)
gof_result = gof(result; stats=[:degree, :esp, :distance])
# on a directed fit, :degree yields the :idegree and :odegree panels

# MCMC diagnostics (requires an MCMLE fit; throws an
# ArgumentError for MPLE fits, which have no MCMC samples)
mcmle_result = ergm(net, terms; method=:mcmle)
mcmc_diagnostics(mcmle_result)
```

## Shared optimization utility

`newton_fit(loglik_grad_hess, θ0)` — a Newton–Raphson maximizer with step
halving — and `logistic_derivatives(X, y)` / `logistic_derivatives(X, n_tot,
n_one)` — the allocation-free logistic `(ll, grad, hess)` kernel — are
**NetworkCore.jl's** (`public` there), re-exported by ERGM.jl so that
`ERGM.newton_fit === NetworkCore.newton_fit`. ERGM's own MPLE is exactly this
pair: the compressed change-statistic design in binomial-row form, maximized
to `tol=1e-8`. Every ERGM variant's pseudo-likelihood and REM's conditional
logit run on the same two functions; nothing is re-implemented locally.

## Not implemented

What a user of R `ergm` will not find here yet, and what happens instead:

- **Absent term families**: sender/receiver attribute terms `nodeicov`/`nodeocov`/`nodeifactor`/`nodeofactor`; the directed triadic terms `ttriple`/`ctriple`/`asymmetric`/`transitive`/`intransitive` (and the `attr=` argument of `transitiveties`/`cyclicalties`); the `dsp`/`nsp` count terms (and the `RTP` type of `esp`/`desp`); `isolates` (only as `Degree(0)`), `balance`, `cycle`, `smalldiff`, `dyadcov`, `hamming`, `concurrentties`, `nodematch(keep=)`, `nodemix(levels2=)`, multi-range `degrange(from=c(...))` (one `DegRange` per range instead)
  have no counterpart (there is no term to call, so nothing is silently
  mis-fit). Added in 0.2: `concurrent`, `gwnsp`, `degrange`/`idegrange`/`odegrange`, `meandeg`, `density`, `triadcensus`, `sender`/`receiver`, `transitiveties`, `cyclicalties`. Use `IStar`/`OStar` for directed stars, `Triangle` (=
  `ttriple + ctriple` on a directed network) for triadic closure,
  `GWESP`/`GWDSP` for shared-partner effects, and `NodeFactor`/`NodeCov`
  for a one-mode attribute main effect.
- **Two-mode (bipartite) networks and terms** (`b1degree`, `b2degree`,
  `b1factor`, `b1nodematch`, …): `ERGMModel`/`fit_ergm` throw an
  `ArgumentError` on a network created with `bipartite=k` or a
  `BipartiteNetwork`. One-mode terms on a two-mode network would count the
  structurally impossible within-mode dyads as observations, so the model is
  refused rather than silently mis-fit.
- **Curved terms beyond `gwesp` and `gwdegree`**: `GWESP(decay; fixed=false)`
  and `GWDegree(decay; fixed=false)` estimate their decay (by `mcmle`); the
  other geometrically weighted terms (`gwdsp`, `gwnsp`, `gwidegree`,
  `gwodegree`) take a fixed decay only, and a curved term cannot be combined
  with an offset or with `missing=:mle`. A decay the data do not identify
  (its term's coefficient near 0) ends unconverged, with a warning — R's
  curved MCMLE is fragile in the same cases.
- **`constraints=`** (`edges`, `degrees`, `blockdiag`, `observed`, …): a
  non-empty `constraints=` on `fit_ergm`/`ergm` or `ERGMFormula` throws an
  `ArgumentError` (the abstract type `ERGM.ConstraintTerm` is an unexported
  placeholder).
- **Two corners of `offset()`**: `Offset(term, coef)` fixes a coefficient at
  a finite value or at `±Inf` (a constraint: ties forbidden or forced), as
  R does, in every estimator and sampler. Two things differ from R, both in
  words: an observed network that violates an infinite offset (a tie on a
  forbidden dyad, no tie on a forced one) has probability 0 under the model
  and is refused with an `ArgumentError` (R fits it regardless); and an
  `mcmle` fit with an infinite offset on a dyad-*dependent* statistic
  (`Offset(Triangle(), -Inf)`) reports no log-likelihood/AIC/BIC (`NaN`,
  said in `show` — there is no dyad-independent reference to bridge from).
- **R's sampler defaults outside `mcmle`**: `mcmle` samples as R does
  (`SPDyad`, ESS-adaptive to 64), but `simulate_ergm`, `gof`, `mh_sample`,
  `sample_networks` and the MPLE bootstrap default to plain TNT where R's
  `simulate` uses `SPDyad` (both are exact; pass `proposal=:spdyad`). The
  shared-partner proposal handles outgoing two-paths only (R's default type
  for directed networks); R's adaptive burn-in detection and the
  `BDStratTNT` family of stratified proposals are not implemented.
- **Attribute terms with an NA value** (`nodecov`, `nodefactor`,
  `nodematch`, `nodemix`, `absdiff` on an attribute not set on every vertex):
  refused at model construction with an `ArgumentError` naming the vertices
  without a value, exactly as statnet refuses ("Attribute has missing data").
  There is no zero-fill; set a value for every vertex or drop the term.
- **Self-loops**: a network that contains a loop (`loops=true` and
  `has_edge(net, v, v)`) is refused at model construction with an
  `ArgumentError` naming the vertices — the statistics would count the
  loop, but the pseudo-likelihood, `nobs`, the proposal and every simulation
  range over the off-diagonal dyads only (R ergm warns "This network
  contains loops" and fits regardless). A `loops=true` network with no loop
  is accepted and modelled as the loop-free network it is.
- **`drop=FALSE`**: a statistic at the boundary of its attainable range (a
  `NodeMatch` level with no within-level tie, a `NodeMix` cell of a
  singleton level, a `Triangle` on a network with no two-path, …) has no
  finite estimate. `mple` and `mcmle` — and so every default fit — do what
  R's default `drop=TRUE` does: warn "observed statistic(s) … are at their
  smallest attainable values. Their coefficients will be fixed at -Inf",
  return `-Inf` (or `+Inf`) with standard error 0, and estimate the other
  coefficients (the MPLE on the dyads the dropped term does not touch, R's
  `edges = logit(12/300)` on the old Quick Start network; the MCMLE with the
  statistic held at its bound). `dof`, AIC and BIC count the finite
  coefficients and the dyads they were estimated on, R's `logLik` df/nobs.
  What is not implemented is R's `drop=FALSE` fit, which keeps such a term
  in a model whose "MLE is poorly defined": `drop=false` refuses the model
  with an `ArgumentError` instead. Two smaller differences: a curved model
  (`GWESP(d; fixed=false)`) with a boundary statistic is refused (fix the
  decay), and a `GWESP`/`GWDSP`/`GWNSP` statistic that is 0 is fixed at
  `-Inf` (its smallest value) where R, which declares no bound for these
  terms, fits a finite, unidentified coefficient. A statistic that does not
  vary at all, or is a linear combination of the others, is reported as
  `NaN` where R reports `NA`; under `mcmle` a dyad-dependent one is refused
  (its MPLE start does not exist) unless `init=` is given.
- **User terms that read an edge attribute live, under MCMC**: the
  samplers toggle ties with `rem_edge!`/`add_edge!`, which delete a tie's
  edge attributes, so a term that reads one from the network at evaluation
  time would decay to its default. Every sampler entry point (`mcmle`,
  `simulate_ergm`, `gof`, `sample_networks`, `mh_sample`, the MPLE
  bootstrap) probes the terms from outside ERGM.jl on a network with edge
  attributes and refuses such a term with an `ArgumentError`; snapshot the
  attribute when the model is built (`ERGM.Extension.materialize`, as
  ERGMUserterms' `WeightedEdges` does) or pass a matrix (`EdgeCov`). The
  MPLE is unaffected.
- **Simulation and GOF under missing data**: a simulated network has a
  value at every dyad, so `simulate_ergm`/`gof` on a masked network can only
  freeze the masked dyads at their face value (`missing=:condition_on_face`,
  warned) — there is no `missing=:mle` for them (refused with an
  explanation). Estimation under missing data *is* implemented:
  `mcmle(...; missing=:mle)` and the available-case MPLE.

## Change Statistics

For efficient MCMC, each term implements `change_stat()`, the add-direction
change statistic `g(y⁺ᵢⱼ) − g(y⁻ᵢⱼ)` — the statistic with edge (i,j) present
minus the statistic with it absent. Its value does not depend on whether the
edge currently exists:

<!-- skip-check -->
```julia
# Change in statistic from adding edge (i,j), given the rest of the network
delta = change_stat(term, net, i, j)
```

## Custom Terms and Extensions

See ERGMUserterms.jl for templates and utilities for developing custom terms.

A package that builds estimators or term families on ERGM.jl uses its
extension API, the submodule `ERGM.Extension` (`using ERGM.Extension`): the
formula pipeline, the attainable range a term declares (R's `minval` /
`maxval`, which drives R's `drop=TRUE`), the pseudo-likelihood fitter and the
Monte-Carlo MLE drivers. Its names are covered by semantic versioning; no
underscore name of ERGM.jl is public. TERGM, ERGMCount, ERGMRank, ERGMEgo,
ERGMMulti and ERGMUserterms are written against it. See the
[Extension API](https://statistical-network-analysis-with-Julia.github.io/ERGM.jl/dev/api/extension/)
page.

## Mathematical Background

An ERGM has the form:

```
P(Y = y) = exp(θ'g(y)) / c(θ)
```

Where:
- `Y` is the random network
- `y` is an observed network
- `θ` is the parameter vector
- `g(y)` is the vector of sufficient statistics
- `c(θ)` is the normalizing constant

## Documentation

For more detailed documentation, see:

- [Documentation](https://statistical-network-analysis-with-Julia.github.io/ERGM.jl/dev/)
- Coming from R: the [R concordance](https://statistical-network-analysis-with-Julia.github.io/ERGM.jl/dev/guide/r_concordance/)
  and the ecosystem's [migration guide](https://statistical-network-analysis-with-julia.github.io/migration/)

## References

1. Hunter, D. R., & Handcock, M. S. (2006). Inference in curved exponential family models for networks. *Journal of Computational and Graphical Statistics*, 15(3), 565-583.

2. Robins, G., Pattison, P., Kalish, Y., & Lusher, D. (2007). An introduction to exponential random graph (p*) models for social networks. *Social Networks*, 29(2), 173-191.

3. Hunter, D. R., Handcock, M. S., Butts, C. T., Goodreau, S. M., & Morris, M. (2008). ergm: A Package to Fit, Simulate and Diagnose Exponential-Family Models for Networks. *Journal of Statistical Software*, 24(3), 1-29.

4. Krivitsky, P. N., Hunter, D. R., Morris, M., & Klumb, C. (2023). ergm 4: New Features for Analyzing Exponential-Family Random Graph Models. *Journal of Statistical Software*, 105(6), 1-44. doi:10.18637/jss.v105.i06

5. Handcock, M. S., & Gile, K. J. (2010). Modeling social networks from sampled data. *The Annals of Applied Statistics*, 4(1), 5-25.

6. Morris, M., Handcock, M. S., & Hunter, D. R. (2008). Specification of exponential-family random graph models: terms and computational aspects. *Journal of Statistical Software*, 24(4), 1-24.

## Citation

ERGM.jl is a port of R's `ergm`: its estimators, terms and defaults follow
`ergm` 4. **Please also cite the R package and the methods it implements** —
Krivitsky, Hunter, Morris & Klumb (2023) for ergm 4 and Hunter et al.
(2008) for ergm (references 3 and 4 above), and the methods papers your
analysis relies on (Hunter & Handcock 2006 for MCMLE, Handcock & Gile 2010
for missing data) — as the statnet authors ask:

```biblatex
@article{ergm4,
  author  = {Krivitsky, Pavel N. and Hunter, David R. and Morris, Martina and Klumb, Chad},
  title   = {{ergm 4}: New Features for Analyzing Exponential-Family Random Graph Models},
  journal = {Journal of Statistical Software},
  year    = {2023},
  volume  = {105},
  number  = {6},
  pages   = {1--44},
  doi     = {10.18637/jss.v105.i06}
}
```

To cite ERGM.jl itself, use the entry in [`CITATION.bib`](CITATION.bib):

```biblatex
@misc{SNWJERGMJL,
  author = {Santoni, Simone},
  title = {ERGM.jl: Exponential Random Graph Models for Julia},
  year = {2026},
  url = {https://github.com/statistical-network-analysis-with-Julia/ERGM.jl},
  note = {Homepage: https://statistical-network-analysis-with-Julia.github.io/ERGM.jl; GitHub: https://github.com/statistical-network-analysis-with-Julia}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
