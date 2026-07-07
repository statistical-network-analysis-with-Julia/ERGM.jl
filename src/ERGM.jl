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

# Core types
export AbstractERGMTerm, ERGMFormula, ERGMModel, ERGMResult
export TermSet
export compute, compute_all, name

# Built-in terms
export Edges, Mutual, Triangle, Kstar, TwoPath
export NodeFactor, NodeCov, NodeMatch, AbsDiff
export EdgeCov, GWESP, GWDegree

# Model fitting
export ergm, fit_ergm
export mple, mcmle

# Simulation
export simulate_ergm, sample_networks

# Diagnostics
export gof, mcmc_diagnostics

# Utilities
export change_stat, change_stat_all, summary_stats
export coef, stderror, vcov

# Include source files
include("terms/base.jl")
include("terms/structural.jl")
include("terms/nodal.jl")
include("terms/dyadic.jl")
include("estimation/mple.jl")
include("estimation/mcmle.jl")
include("mcmc/simulation.jl")
include("mcmc/diagnostics.jl")

end # module
