"""
    ERGM.Extension

The stable API for packages that build estimators or term families on
ERGM.jl. TERGM.jl, ERGMCount.jl, ERGMRank.jl, ERGMEgo.jl, ERGMMulti.jl and
ERGMUserterms.jl are written against it, and a third-party package can be
too. The names are covered by semantic versioning from ERGM.jl 0.2.0. They
are not exported by `ERGM` itself: bring them in with `using ERGM.Extension`,
or call them qualified (`Extension.mple_fit_design(...)` after
`using ERGM: Extension`).

| Building block | Names |
|:--|:--|
| Formula pipeline | [`collect_terms`](@ref ERGM.Extension.collect_terms), [`validate_formula`](@ref ERGM.Extension.validate_formula), [`materialize`](@ref ERGM.Extension.materialize), [`expand_terms`](@ref ERGM.Extension.expand_terms), [`require_supported_network`](@ref ERGM.Extension.require_supported_network), [`n_observed_dyads`](@ref ERGM.Extension.n_observed_dyads) |
| Boundary of the parameter space | [`attainable_range`](@ref ERGM.Extension.attainable_range), [`extreme_statistics`](@ref ERGM.Extension.extreme_statistics), [`boundary_columns`](@ref ERGM.Extension.boundary_columns), [`warn_boundary`](@ref ERGM.Extension.warn_boundary) |
| Pseudo-likelihood | [`mple_fit_design`](@ref ERGM.Extension.mple_fit_design) |
| Monte-Carlo MLE | [`mcmc_defaults`](@ref ERGM.Extension.mcmc_defaults), [`mcmle_solve`](@ref ERGM.Extension.mcmle_solve), [`confidence_test`](@ref ERGM.Extension.confidence_test), [`mcmle_covariance`](@ref ERGM.Extension.mcmle_covariance), [`ess_sample`](@ref ERGM.Extension.ess_sample), [`mcmle_sampler`](@ref ERGM.Extension.mcmle_sampler), [`bridge_integrate`](@ref ERGM.Extension.bridge_integrate) |

Two of them are extension points that a package adds methods to for its own
types: [`attainable_range`](@ref ERGM.Extension.attainable_range) (the
`minval`/`maxval` of R ergm's terms, which drives R's `drop=TRUE`) and
[`materialize`](@ref ERGM.Extension.materialize) (the snapshot a term takes
of the network's attributes when a model is built). The boundary convention
every ERGM-family estimator follows is described on the documentation page
"Extension API".

# Example
```julia
using ERGM
using ERGM.Extension
net = load_dataset(:florentine_marriage)
ts = TermSet([Edges(), Triangle()])
validate_formula(ts, net) === nothing                  # true
extreme_statistics(materialize(ts, net), net)          # Tuple{Int, Symbol}[]: nothing at a bound
attainable_range(Edges(), net)                         # (0.0, 120.0)
```
"""
module Extension

export collect_terms, validate_formula, materialize, expand_terms,
       require_supported_network, n_observed_dyads
export attainable_range, extreme_statistics, boundary_columns, warn_boundary
export mple_fit_design
export mcmc_defaults, mcmle_solve, confidence_test, mcmle_covariance, ess_sample,
       mcmle_sampler, bridge_integrate

# The generic functions. ERGM.jl imports each by name and defines its methods
# (and their docstrings) next to the code they belong to; a package built on
# ERGM.jl adds methods for its own term and network types in the same way.
function collect_terms end
function validate_formula end
function materialize end
function expand_terms end
function require_supported_network end
function n_observed_dyads end
function attainable_range end
function extreme_statistics end
function boundary_columns end
function warn_boundary end
function mple_fit_design end
function mcmc_defaults end
function mcmle_solve end
function confidence_test end
function mcmle_covariance end
function ess_sample end
function mcmle_sampler end
function bridge_integrate end

end # module Extension
