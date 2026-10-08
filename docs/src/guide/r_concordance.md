# Coming from R

ERGM.jl follows R's `ergm` package: the same models, the same default
estimator rule, the same coefficient labels and R's handling of statistics at
the boundary of their attainable range. This page maps the R calls you know
to their ERGM.jl counterparts. The ecosystem's
[migration guide](https://statistical-network-analysis-with-julia.github.io/migration/)
covers every package of the suite (network, sna, tergm, networkDynamic, …).

## Fitting

| R (`ergm` 4) | ERGM.jl |
|:--|:--|
| `ergm(net ~ edges + triangle)` | `ergm(net, [Edges(), Triangle()])` (`ergm === fit_ergm`) |
| `estimate = "MLE"` (the default) | `method = :auto` (the default): the MPLE when the formula is dyad-independent, where it is the exact MLE, and the MCMLE otherwise, as in R |
| `estimate = "MPLE"` | `method = :mple` |
| `estimate = "CD"` | not implemented |
| `control.ergm(seed = s)` | `rng = Xoshiro(s)` (every random draw flows from `rng`) |
| `control.ergm(init = θ)` | `init = θ` |
| `control.ergm(MCMLE.maxit = k)` | `maxiter = k` |
| `control.ergm(MCMC.burnin, MCMC.interval)` | `burnin`, `interval` (default: R's dyad-scaled rule) |
| `control.ergm(MCMLE.effectiveSize = 64)` | `effective_size = 64` (the default) |
| `control.ergm(MCMLE.MCMC.precision, MCMLE.confidence)` | `conv_precision`, `conv_confidence` |
| `control.ergm(MCMC.prop = ~sparse + .triadic)` | `proposal = :spdyad` (the `mcmle` default) |
| `control.ergm(drop = TRUE)` (the default) | `drop = true` (the default) |
| `offset(term)`, `offset.coef = c` | `Offset(term, c)` |
| `constraints = ~…` | not implemented (refused with an `ArgumentError`) |
| NA ties in the network | `set_missing_dyad!`, then `missing = :mle` (R's missing-data MLE) |

**Boundary statistics.** When an observed statistic sits at the boundary of
its attainable range — a level of `nodematch(diff=TRUE)` with no
within-level tie, a `nodemix` cell of a singleton level, `triangle` on a
network with no two-path — R prints "Observed statistic(s) … are at their
smallest attainable values. Their coefficients will be fixed at -Inf." and
estimates the rest. ERGM.jl does the same, with the same sentence, under both
`method = :mple` and `method = :mcmle`; `drop = false` refuses such a model
instead. A statistic that does not vary at all is reported as `NaN` where R
reports `NA`.

## Terms

R's term names map to Julia types with R's coefficient labels:

| R | ERGM.jl | R | ERGM.jl |
|:--|:--|:--|:--|
| `edges` | `Edges()` | `nodecov("x")` | `NodeCov(:x)` |
| `mutual` | `Mutual()` | `nodefactor("x")` | `NodeFactor(:x)` |
| `triangle` | `Triangle()` | `nodematch("x")` | `NodeMatch(:x)` |
| `kstar(k)` | `Kstar(k)` | `nodematch("x", diff=TRUE)` | `NodeMatch(:x; diff=true)` |
| `ostar(k)`, `istar(k)` | `OStar(k)`, `IStar(k)` | `nodemix("x")` | `NodeMix(:x)` |
| `twopath` | `TwoPath()` | `absdiff("x")` | `AbsDiff(:x)` |
| `degree(0:2)` | `Degree(0:2)` | `edgecov(W)` | `EdgeCov(W)` |
| `idegree(d)`, `odegree(d)` | `IDegree(d)`, `ODegree(d)` | `concurrent` | `Concurrent()` |
| `gwesp(0.5, fixed=TRUE)` | `GWESP(0.5)` | `degrange(a, b)` | `DegRange(a, b)` |
| `gwesp(fixed=FALSE)` | `GWESP(0.5; fixed=false)` | `meandeg`, `density` | `MeanDeg()`, `Density()` |
| `gwdsp`, `gwnsp` | `GWDSP(d)`, `GWNSP(d)` | `sender`, `receiver` | `Sender()`, `Receiver()` |
| `gwdegree(0.5, fixed=TRUE)` | `GWDegree(0.5)` | `transitiveties`, `cyclicalties` | `TransitiveTies()`, `CyclicalTies()` |
| `gwidegree`, `gwodegree` | `GWIDegree(d)`, `GWODegree(d)` | `triadcensus` | `ERGM.TriadCensus()` |
| `esp(0:2)`, `desp(d, type=)` | `ESP(0:2)`, `ESP(d; type=)` | | |

The terms guide lists every term and the R terms that are not implemented.

## Working with a fit

| R | ERGM.jl |
|:--|:--|
| `summary(net ~ terms)` | `summary_stats(net, terms)` |
| `coef(fit)`, `names(coef(fit))` | `coef(fit)`, `coefnames(fit)` |
| `summary(fit)$coefficients` | `coeftable(fit)` |
| `vcov(fit)`, `confint(fit)` | `vcov(fit)`, `confint(fit)` |
| `logLik(fit)`, `AIC(fit)`, `BIC(fit)`, `nobs(fit)` | `loglikelihood(fit)`, `aic(fit)`, `bic(fit)`, `nobs(fit)` |
| `simulate(fit, nsim = k)` | `simulate_ergm(fit; n_sim = k)` |
| `gof(fit)` | `gof(fit)` |
| `mcmc.diagnostics(fit)` | `mcmc_diagnostics(fit)` |

```julia
using ERGM, Random
fmh = load_dataset(:faux_mesa_high)
# R: ergm(faux.mesa.high ~ edges + nodematch("Race", diff=TRUE), estimate="MPLE")
fit = ergm(fmh, [Edges(), NodeMatch(:Race; diff=true)]; method=:mple)
coefnames(fit)    # R's labels: "edges", "nodematch.Race.Black", …
coef(fit)[2]      # -Inf: no tie between two Black students, as in R
```
