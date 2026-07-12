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
using Network
using Optim
using Random
using SparseArrays
using Statistics
using StatsBase

import StatsAPI
import StatsAPI: coef, stderror, vcov, loglikelihood, aic, bic, nobs, dof

# `gof` extends the ONE shared Network.jl generic (every model package adds
# methods for its own result types), so `gof(fit)` works uniformly across the
# ecosystem and loading several model packages never collides on the name.
import Network: gof

# Re-export the Network.jl public API so that `using ERGM` alone provides the
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
export newton_fit

# Simulation
export simulate_ergm, sample_networks, mh_sample

# Diagnostics (`gof` is Network.jl's shared generic, extended with a method
# for ERGMResult; the explicit export keeps `using ERGM: gof` working)
export gof, mcmc_diagnostics

# Utilities
export change_stat, change_stat_all, summary_stats
export is_dyad_dependent

# StatsAPI methods (re-exported so `coef(fit)` etc. work with just `using ERGM`)
export coef, stderror, vcov, loglikelihood, aic, bic, nobs, dof

# Include source files
include("terms/base.jl")
include("terms/structural.jl")
include("terms/nodal.jl")
include("terms/dyadic.jl")
include("terms/materialize.jl")
include("estimation/mple.jl")
include("estimation/mcmle.jl")
include("estimation/newton.jl")
include("mcmc/simulation.jl")
include("mcmc/diagnostics.jl")

end # module
