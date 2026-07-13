"""
    ERGM.jl - Exponential Random Graph Models for Julia

A Julia package for fitting, simulating, and diagnosing Exponential-Family
Random Graph Models (ERGMs).

Port of the R ergm package from the StatNet collection.
"""
module ERGM

using Distributions
using Graphs
using LinearAlgebra
using Networks
using Optim
using Random
using SparseArrays
using Statistics
using StatsBase

import StatsAPI
import StatsAPI: coef, stderror, vcov, loglikelihood, aic, bic, nobs, dof

# `gof` extends the ONE shared Networks.jl generic (every model package adds
# methods for its own result types), so `gof(fit)` works uniformly across the
# ecosystem and loading several model packages never collides on the name.
import Networks: gof

# The statistic protocol (`compute`/`name`/`compute_all`) is likewise ONE set of
# shared Networks.jl generics that every model package extends for its own
# statistic types. ERGM's methods take terms (`compute(term, net)`), REM's take
# relational-event statistics (`compute(stat, state, sender, receiver)`); they
# are methods of the same function, so `using ERGM, REM` leaves the verbs usable
# unqualified instead of undefined by Julia's conflicting-export rule.
import Networks: compute, name, compute_all

# The ecosystem missing-data contract (Networks.jl `src/missing.jl`): the
# `supports_missing` trait is extended with a method for `mple` (which drops
# masked dyads from the pseudo-likelihood), and `require_observed` is the
# shared guard the MCMC routines call. Imported by name because we add a
# method to the trait.
import Networks: supports_missing, require_observed

# The shared result-metadata protocol (Networks.jl `src/results.jl`): seven
# generic accessors that say what a fit actually did (which estimand, which
# objective, whether that objective is exact FOR THIS FIT, how the standard
# errors were computed, how masked dyads and tied events were treated, plus
# free-text caveats). Imported by name because ERGM adds methods for
# `ERGMResult`; `fit_metadata(fit)` then collects them.
import Networks: estimand, objective, is_exact, se_method, missing_method,
                 tie_method, approximations

# Re-export the Networks.jl public API so that `using ERGM` alone provides the
# network constructors and accessors, mirroring R's library(ergm) attaching the
# network package. The `Network` name itself is skipped: inside this module it
# is bound to the struct, and exporting it would collide with the package
# module binding in downstream namespaces (a plain @reexport fails for the
# same reason).
for _network_export in names(parentmodule(Network))
    _network_export === :Network && continue
    Core.eval(@__MODULE__, Expr(:export, _network_export))
end

# Core types
export AbstractERGMTerm, ERGMFormula, ERGMModel, ERGMResult
export TermSet
export compute, compute_all, name

# Built-in terms
export Edges, Mutual, Triangle, Kstar, TwoPath
export Degree, IDegree, ODegree
export NodeFactor, NodeCov, NodeMatch, NodeMismatch, NodeMix, AbsDiff
export EdgeCov, GWESP, GWDSP, GWDegree, GWIDegree, GWODegree

# Model fitting
export ergm, fit_ergm
export mple, mcmle
export newton_fit, logistic_derivatives

# Simulation
export simulate_ergm, sample_networks, mh_sample

# Diagnostics (`gof` is Networks.jl's shared generic, extended with a method
# for ERGMResult; the explicit export keeps `using ERGM: gof` working)
export gof, mcmc_diagnostics

# Utilities
export change_stat, change_stat_all, summary_stats

# The public term-trait protocol (`src/terms/traits.jl`): what a term declares
# about the data it needs and the models it belongs in. Third-party terms add
# methods to these and thereby take part in the same formula validation as the
# built-ins. (`supports_missing`, the missing-data half, is a Networks.jl
# generic already re-exported above.)
export is_dyad_dependent
export required_vertex_attributes, required_edge_attributes
export requires_directed, requires_undirected

# StatsAPI methods (re-exported so `coef(fit)` etc. work with just `using ERGM`)
export coef, stderror, vcov, loglikelihood, aic, bic, nobs, dof

# Include source files
include("missing.jl")
include("terms/base.jl")
include("terms/structural.jl")
include("terms/nodal.jl")
include("terms/dyadic.jl")
include("terms/traits.jl")
include("terms/materialize.jl")
include("estimation/mple.jl")
include("estimation/mcmle.jl")
include("estimation/newton.jl")
include("mcmc/simulation.jl")
include("mcmc/diagnostics.jl")

end # module
