# [Extension API](@id extension-api)

`ERGM.Extension` is the stable API for packages that build estimators or
term families on ERGM.jl. TERGM.jl, ERGMCount.jl, ERGMRank.jl, ERGMEgo.jl,
ERGMMulti.jl and ERGMUserterms.jl are written against it, and so can a
third-party package.

```julia
using ERGM
using ERGM.Extension          # the names below, unqualified
# or, qualified:
using ERGM: Extension
Extension.n_observed_dyads(load_dataset(:florentine_marriage))   # 120
```

## The contract

- **Stability.** The names of `ERGM.Extension`, their documented signatures,
  keywords and return values are covered by semantic versioning from ERGM.jl
  0.2.0, like the exported API. A breaking change to one of them is a
  breaking release.
- **Nothing else is.** No underscore name of ERGM.jl is `public`. Everything
  with a leading underscore is internal and may change in any release, so a
  package built on ERGM.jl must not call one. If it needs a building block
  that is not here, the block belongs here: ask for it.
- **Every name has a consumer.** A name is in the extension API because one
  of the six ERGM-family packages calls it, or because it is an extension
  point a package adds methods to. The test suite pins the inventory and
  checks that each name has a consumer.
- **Extension points.** [`attainable_range`](@ref ERGM.Extension.attainable_range)
  and [`materialize`](@ref ERGM.Extension.materialize) are generic functions
  owned by `ERGM.Extension`. A package adds methods for its own term or
  network types to them (`Extension.attainable_range(t::MyTerm, net) = …`);
  ERGM.jl's own estimators then see those methods.
- **`ERGM` exports nothing from here.** `Extension` itself is a `public`
  (not exported) name of `ERGM`, so `using ERGM` leaves a user's namespace
  unchanged.

```@docs
ERGM.Extension
```

## Formula pipeline

What `ERGMModel` does to a term list, as separate steps, so that a model
evaluated on several networks (TERGM's panels, ERGMMulti's layers) runs the
same checks and expansion on each.

```@docs
ERGM.Extension.collect_terms
ERGM.Extension.validate_formula
ERGM.Extension.materialize
ERGM.Extension.expand_terms
ERGM.Extension.require_supported_network
ERGM.Extension.n_observed_dyads
```

## [The boundary convention](@id boundary-convention)

A statistic whose observed value sits at an end of its attainable range has
no finite maximum-likelihood or pseudo-likelihood estimate. R ergm's default
(`control.ergm(drop=TRUE)`) fixes its coefficient at `-Inf` (at the smallest
value) or `+Inf` (at the largest) and estimates the rest. Every ERGM-family
estimator follows the same convention:

1. **Detection.** The boundary is
   [`extreme_statistics`](@ref ERGM.Extension.extreme_statistics)`(terms, net)`
   (R's `ergm.checkextreme.model`: the observed statistic equals an end of its
   [`attainable_range`](@ref ERGM.Extension.attainable_range)) together with
   [`boundary_columns`](@ref ERGM.Extension.boundary_columns)`(X, n_tot, n_one; fixed=extreme)`
   (the design test, iterated). The first catches statistics whose change
   statistics are all zero, which the design cannot show; the second catches
   statistics the range does not bound, such as a perfectly separated
   `nodematch`.
2. **Pseudo-likelihood.** A pseudo-likelihood fitter calls
   [`mple_fit_design`](@ref ERGM.Extension.mple_fit_design)`(X, n_tot, n_one, names; context, extreme)`.
   That one function does the drop (∓Inf, standard error 0, p-value 0, the
   other coefficients fitted on the rows the dropped columns do not touch,
   `n_kept` for the BIC), the aliasing (a statistic that does not vary, or a
   linear combination of the ones before it, is `NaN`, R's `NA`), Newton and
   NetworkCore's separation verdict. It returns the verdict, so the caller
   never runs it again. Its `note` keyword replaces R ergm's closing sentences
   when the package's own R counterpart behaves differently.
3. **Monte-Carlo MLE.** An MCMC estimator applies the drop instead of
   refusing:
   - the dropped columns become fixed ∓Inf coordinates, excluded from the
     step, the stopping rule and the covariance, and held at ±1e300 in the
     sampler, so the statistic never leaves its bound;
   - it warns R ergm's sentence ([`warn_boundary`](@ref ERGM.Extension.warn_boundary));
   - `drop=false` is the strict mode, an `ArgumentError` raised before any
     start;
   - the log-likelihood is `NaN` when a dropped statistic is dyad-dependent,
     because such a constraint has no dyad-independent reference to bridge
     from;
   - a dyad-independent aliased column is held at 0 (any value gives the same
     model) and reported `NaN`; a dyad-dependent one is refused unless
     `init=` is given.
4. **Reporting.** Fixed coefficients are ∓Inf with standard error 0 and
   p-value 0; aliased ones are `NaN`. `dof` counts the finite coefficients.
   `show` and `approximations` carry a note naming them, and the parametric
   bootstraps refuse a point estimate that is not finite.
5. **Term types of other packages.** A package declares the range of its own
   term types by adding [`attainable_range`](@ref ERGM.Extension.attainable_range)
   methods, never by keeping a separate range table. ERGMMulti declares its
   layer terms' ranges on a `MultilayerNetwork`, ERGMRank its rank terms' on
   a `RankNetwork`, and ERGMEgo maps each ego term to the ERGM term whose
   statistic it estimates.

A package may depart from point 3 where its sampler cannot hold a statistic
at a bound, and must say why. ERGMRank's MCMLE refuses a boundary statistic,
because single swaps do not connect the rankings that share an extreme
value; its swap MPLE follows the convention.

```@docs
ERGM.Extension.attainable_range
ERGM.Extension.extreme_statistics
ERGM.Extension.boundary_columns
ERGM.Extension.warn_boundary
```

### Declaring the range of a new term

```julia
using ERGM
using ERGM: Extension

# A count of isolated vertices: between 0 and the number of vertices
struct Isolates <: AbstractERGMTerm end
ERGM.name(::Isolates) = "isolates"
ERGM.compute(::Isolates, net) = Float64(count(v -> degree(net, v) == 0, vertices(net)))
Extension.attainable_range(::Isolates, net::AbstractNetwork) = (0.0, Float64(nv(net)))

net = network(6; directed=false)
add_edge!(net, 1, 2); add_edge!(net, 3, 4); add_edge!(net, 5, 6)
Extension.extreme_statistics([Edges(), Isolates()], net)   # [(2, :min)]: no isolate
```

## Pseudo-likelihood

```@docs
ERGM.Extension.mple_fit_design
```

## Monte-Carlo MLE

The drivers behind `mcmle`, each written against a closure so that it knows
nothing about networks. `mcmle` itself is
[`mcmle_solve`](@ref ERGM.Extension.mcmle_solve) plus a sampler, and that
sampler is [`mcmle_sampler`](@ref ERGM.Extension.mcmle_sampler): an
estimator that solves moment equations on an `ERGMModel` (ERGMEgo's moment
matching, the curved MCMLE) samples exactly as `mcmle` does — R ergm 4's
SPDyad proposal and ESS-adaptive, continued chains — by passing its `draw`
and `resize` to `mcmle_solve` or to its own iteration.

```@docs
ERGM.Extension.mcmc_defaults
ERGM.Extension.mcmle_solve
ERGM.Extension.confidence_test
ERGM.Extension.mcmle_covariance
ERGM.Extension.ess_sample
ERGM.Extension.mcmle_sampler
ERGM.Extension.bridge_integrate
```

The convergence tests a moment-matching estimator reports,
[`ERGM.mcmc_convergence`](@ref) and its result type `ERGM.MCMLEConvergence`,
and R's default-estimator rule [`ERGM.resolve_method`](@ref) are `public`
names of `ERGM` itself.

## [Name map](@id extension-name-map)

Development versions declared these building blocks `public` under
underscore names. They are no longer `public`; each is replaced by an
`ERGM.Extension` name, with no deprecation alias (0.2.0 is the first
release).

| Development name | `ERGM.Extension` name | Note |
|:--|:--|:--|
| `ERGM._collect_terms` | `collect_terms` | |
| `ERGM._validate_formula` | `validate_formula` | |
| `ERGM._materialize` | `materialize` | an extension point: add methods to `Extension.materialize` |
| `ERGM._expand_terms` | `expand_terms` | |
| `ERGM._refuse_two_mode`, `ERGM._refuse_self_loops` | `require_supported_network` | merged: the one check `ERGMModel` runs on a network; the two refusals stay internal |
| `ERGM._n_dyads` | `n_observed_dyads` | renamed: REM.jl exports an unrelated `n_dyads` |
| `ERGM._attainable_range` (internal) | `attainable_range` | now a documented extension point |
| `ERGM._extreme_statistics` (internal) | `extreme_statistics` | also takes a vector of terms |
| `ERGM._boundary_columns_iterated` | `boundary_columns` | the one-pass test is internal |
| `ERGM._warn_boundary` | `warn_boundary` | |
| `ERGM._mple_fit_design` | `mple_fit_design` | adds `note=` and `noun=`; returns `boundary`, `aliased`, `fitted`, `kept_rows` and the separation `verdict` |
| `ERGM._mcmc_defaults` | `mcmc_defaults` | |
| `ERGM._mcmle_solve` | `mcmle_solve` | |
| `ERGM._confidence_test` | `confidence_test` | |
| `ERGM._mcmle_covariance` | `mcmle_covariance` | |
| `ERGM._ess_sample` | `ess_sample` | |
| `ERGM._bridge_integrate` | `bridge_integrate` | |
| `ERGM._separated`, `ERGM._warn_separated` | removed | had no caller; NetworkCore's `logistic_separation` and `warn_separation` |
