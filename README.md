# ERGM.jl


[![Network Analysis](https://img.shields.io/badge/Network-Analysis-orange.svg)](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl)
[![Build Status](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/statistical-network-analysis-with-Julia/ERGM.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://statistical-network-analysis-with-Julia.github.io/ERGM.jl/stable/)
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

Requires Julia 1.12+. ERGM.jl depends on the unregistered
[Networks.jl](https://github.com/statistical-network-analysis-with-Julia/Networks.jl) package, which must be added first:

```julia
using Pkg
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/Networks.jl")
Pkg.add(url="https://github.com/statistical-network-analysis-with-Julia/ERGM.jl")
```

For development, you can instead clone all ecosystem repositories side by
side (the monorepo layout) and start Julia with the root workspace project
(`julia --project=.` in the clone root): the `[sources]` path dependencies
then wire the packages together with no ordered installs needed.

## Features

- **Model terms**: Structural, nodal, and dyadic covariate terms
- **Estimation**: MPLE (fast) and MCMLE (full likelihood)
- **Simulation**: MCMC network simulation
- **Diagnostics**: Goodness-of-fit testing, MCMC diagnostics

## Quick Start

```julia
using Networks
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

# Fit model using MPLE
result = ergm(net, terms; method=:mple)
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
GWDSP(decay)         # Geometrically weighted dyadwise shared partners
                     # (all dyads, tied or not; same directed types, every
                     #  one summed over ordered dyads as in R's dgwdsp)
GWDegree(decay)      # Geometrically weighted degree (undirected only,
                     #   as in R; coefficient "gwdeg.fixed.<decay>")
GWIDegree(decay)     # Geometrically weighted in-degree (directed only)
GWODegree(decay)     # Geometrically weighted out-degree (directed only)
```

Every geometrically weighted term takes a **fixed decay ≥ 0** —
`GWESP(0.0)` is statnet's `gwesp(0, fixed=TRUE)` — and its coefficient
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
NodeMatch(:attr; diff=true, level="A")  # Differential (per-level) homophily,
                                        # as in R nodematch(diff=TRUE):
                                        # one term per attribute level
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
> R-compatible per-level homophily (`nodematch.<attr>.<level>` naming, and
> it requires `level=` rather than guessing); the old mismatch statistic
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

```julia
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

An MCMLE fit that exhausts `maxiter` is loud: it warns with the last max
t-ratio, Hotelling p-value and step length, prints the caveat under
`Converged: false`, lists it in `Networks.approximations(result)` and keeps
the numbers in `result.mcmc_convergence`; continue it with
`ergm(net, terms; method=:mcmle, init=coef(result), maxiter=40)`.

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
the default inverse-Hessian MPLE standard errors are anticonservative —
the pseudo-likelihood treats dependent dyads as independent observations —
and `show(result)` prints a caveat (as statnet does). For honest standard
errors, use the parametric bootstrap or refit with `method=:mcmle`:

```julia
result = ergm(net, terms; method=:mple, se=:bootstrap, n_boot=100)
```

A simulated replicate on which the MPLE does not exist (no triangle under
`Triangle()`, no shared-partner edge under `GWESP`, …) is excluded from the
bootstrap covariance — one warning says how many of `n_boot` were, the
result records it (`approximations(result)`, `result.boot_replicates`) —
so the standard errors are finite; `se=:bootstrap` refuses a point estimate
that itself has a `±Inf` coefficient.

For dyad-independent models the pseudo-likelihood is the true likelihood
and the default SEs are correct (no caveat is printed).

MCMLE's reported log-likelihood (hence AIC/BIC) is estimated by an
ergm-style path-sampling (bridge) ladder from a dyad-independent reference
to the fitted coefficients (`bridge_rungs` keyword), which is far more
accurate than one-jump importance sampling.

### Missing (unobserved) dyads

If dyads of the network are masked as missing with Networks.jl's
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
n_dyads ÷ 10)`; `ERGM._mcmc_defaults(model)` shows the numbers); pass
`burnin=`/`interval=` to override.

All sampling and fitting functions accept an `rng::AbstractRNG` keyword;
runs with the same seed are exactly reproducible. `sample_networks`,
`simulate_ergm`, `gof` and `mcmle` split their draws over independent chains
run on separate threads (`n_chains` keyword), seeded deterministically from
the caller's `rng`, so results are also independent of the thread count.

Every sampler is built on the exported Metropolis kernel `mh_toggle!(rng, θ,
delta, propose, change!, apply!, on_sample; burnin, interval, n_samples)`:
it owns the accept/reject arithmetic, burn-in and thinning, and takes the
proposal, the change statistics and the state mutation as callables — so the
ERGM variants (TERGM, ERGMMulti, ERGMRank) sample their own state types with
the same, allocation-free, bit-reproducible loop.

## Goodness-of-Fit

`gof` is a method of the ecosystem-wide `Networks.gof` generic — the same
verb works on every fitted model in the ecosystem. For directed networks
the degree comparison is split into in- and out-degree distributions
(`:idegree`/`:odegree`), as in R ergm:

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
**Networks.jl's** (`public` there), re-exported by ERGM.jl so that
`ERGM.newton_fit === Networks.newton_fit`. ERGM's own MPLE is exactly this
pair: the compressed change-statistic design in binomial-row form, maximized
to `tol=1e-8`. Every ERGM variant's pseudo-likelihood and REM's conditional
logit run on the same two functions; nothing is re-implemented locally.

## Not implemented

What a user of R `ergm` will not find here yet, and what happens instead:

- **Absent term families**: sender/receiver attribute terms `nodeicov`/`nodeocov`/`nodeifactor`/`nodeofactor`; the directed triadic terms `ttriple`/`ctriple`/`transitiveties`/`cyclicalties`/`asymmetric`; the `esp`/`dsp`/`nsp` count terms; `isolates` (only as `Degree(0)`), `balance`, `cycle`, `smalldiff`, `nodematch(keep=)`
  have no counterpart (there is no term to call, so nothing is silently
  mis-fit). Use `IStar`/`OStar` for directed stars, `Triangle` (=
  `ttriple + ctriple` on a directed network) for triadic closure,
  `GWESP`/`GWDSP` for shared-partner effects, and `NodeFactor`/`NodeCov`
  for a one-mode attribute main effect.
- **Two-mode (bipartite) networks and terms** (`b1degree`, `b2degree`,
  `b1factor`, `b1nodematch`, …): `ERGMModel`/`fit_ergm` throw an
  `ArgumentError` on a network created with `bipartite=k` or a
  `BipartiteNetwork`. One-mode terms on a two-mode network would count the
  structurally impossible within-mode dyads as observations, so the model is
  refused rather than silently mis-fit.
- **Curved terms** (`gwesp(fixed=FALSE)`, `gwdegree(fixed=FALSE)`,
  `gwdsp(fixed=FALSE)` with the decay estimated): every geometrically
  weighted term takes a fixed decay (≥ 0); there is no curved-family
  estimation.
- **`constraints=`** (`edges`, `degrees`, `blockdiag`, `observed`, …): a
  non-empty `constraints=` vector on `ERGMFormula` throws an `ArgumentError`;
  `ConstraintTerm` is exported as a reserved type only.
- **`offset()` terms**: no counterpart; every term in the formula is
  estimated.
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
  `NodeMatch` with no within-group tie, …) has no finite estimate. `mple`
  does what R's default `drop=TRUE` does — warns "observed statistic(s) …
  are at their smallest attainable values. Their coefficients will be fixed
  at -Inf", returns `-Inf` (or `+Inf`) with standard error 0, and fits the
  other coefficients on the dyads the dropped term does not touch (R's
  numbers, e.g. `edges = logit(12/300)` on the old Quick Start network).
  `mcmle` refuses such a model with the same sentence and the two ways out
  (drop the term, or `method=:mple`); there is no R-style `drop=FALSE` fit
  of a boundary statistic. `dof`, AIC and BIC count the finite coefficients
  and the dyads they were estimated on, R's `logLik` df/nobs.
- **R's exact `mple.existence` test**: R decides whether the MPLE exists
  with a linear program. ERGM.jl has no LP solver; a design separated by a
  *combination* of statistics (no single one at its boundary) is detected
  from the two signatures an asymptote leaves on the Newton iteration (a
  perfectly predicted dyad at a fitted probability within 1e-8 of 0/1 while
  the Newton step is still large relative to θ) — `mple` then warns "the
  MPLE does not exist (perfect separation)" and returns `converged ==
  false`; `mcmle` refuses to start from it. A design whose asymptote is
  approached more slowly than that is not flagged.
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

## Custom Terms

See ERGMUserterms.jl for templates and utilities for developing custom terms.

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

- [Stable Documentation](https://statistical-network-analysis-with-Julia.github.io/ERGM.jl/stable/)
- [Development Documentation](https://statistical-network-analysis-with-Julia.github.io/ERGM.jl/dev/)

## References

1. Hunter, D. R., & Handcock, M. S. (2006). Inference in curved exponential family models for networks. *Journal of Computational and Graphical Statistics*, 15(3), 565-583.

2. Robins, G., Pattison, P., Kalish, Y., & Lusher, D. (2007). An introduction to exponential random graph (p*) models for social networks. *Social Networks*, 29(2), 173-191.

3. Hunter, D. R., Handcock, M. S., Butts, C. T., Goodreau, S. M., & Morris, M. (2008). ergm: A Package to Fit, Simulate and Diagnose Exponential-Family Models for Networks. *Journal of Statistical Software*, 24(3), 1-29.

## Citation

If you use ERGM.jl in your work, please cite it using the entry in
[`CITATION.bib`](CITATION.bib):

```biblatex
@misc{SNWJERGMJL,
  author = {{Statistical Network Analysis with Julia}},
  title = {ERGM.jl: Exponential Random Graph Models for Julia},
  year = {2026},
  url = {https://github.com/statistical-network-analysis-with-Julia/ERGM.jl},
  note = {Homepage: https://statistical-network-analysis-with-Julia.github.io/ERGM.jl; GitHub: https://github.com/statistical-network-analysis-with-Julia}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
