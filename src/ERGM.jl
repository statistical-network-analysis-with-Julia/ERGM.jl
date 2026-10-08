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
using NetworkCore
using Random
using Statistics
using Printf: @sprintf
using PrecompileTools: @setup_workload, @compile_workload

import StatsAPI
# The ecosystem StatsAPI surface (NetworkCore.jl `src/statsapi.jl`): every verb
# is defined on `ERGMResult` and re-exported below. `coeftable` is the SAME
# binding `using NetworkCore` already provides (`NetworkCore.coeftable ===
# StatsAPI.coeftable`), so it is exported exactly once.
import StatsAPI: coef, stderror, vcov, confint, loglikelihood, aic, bic, nobs,
                 dof, coeftable, coefnames

# `gof` extends the ONE shared NetworkCore.jl generic (every model package adds
# methods for its own result types), so `gof(fit)` works uniformly across the
# ecosystem and loading several model packages never collides on the name.
import NetworkCore: gof

# The statistic protocol (`compute`/`name`/`compute_all`) is likewise ONE set of
# shared NetworkCore.jl generics that every model package extends for its own
# statistic types. ERGM's methods take terms (`compute(term, net)`), REM's take
# relational-event statistics (`compute(stat, state, sender, receiver)`); they
# are methods of the same function, so `using ERGM, REM` leaves the verbs usable
# unqualified instead of undefined by Julia's conflicting-export rule.
import NetworkCore: compute, name, compute_all

# The ecosystem missing-data contract (NetworkCore.jl `src/missing.jl`): the
# `supports_missing` trait is extended with a method for `mple` (which drops
# masked dyads from the pseudo-likelihood), `missing_policies` is extended for
# every MCMC entry point (they take `:condition_on_face`, NOT the generic
# `:face`), and `require_observed` is the shared guard the MCMC routines call.
# Imported by name because we add methods to the traits.
import NetworkCore: supports_missing, require_observed, missing_policies

# The shared result-metadata protocol (NetworkCore.jl `src/results.jl`): seven
# generic accessors that say what a fit actually did (which estimand, which
# objective, whether that objective is exact FOR THIS FIT, how the standard
# errors were computed, how masked dyads and tied events were treated, plus
# free-text caveats). Imported by name because ERGM adds methods for
# `ERGMResult`; `fit_metadata(fit)` then collects them.
import NetworkCore: estimand, objective, is_exact, se_method, missing_method,
                 tie_method, approximations

# Shared numerics, hosted in NetworkCore.jl and imported by name: the ONE Newton–Raphson optimizer and logistic-likelihood
# kernel behind the MPLE (`newton_fit`, `logistic_derivatives` — `public` in
# NetworkCore, re-exported here so `using ERGM` is unchanged and
# `ERGM.newton_fit === NetworkCore.newton_fit`), the ONE z → p helper behind every
# Pr(>|z|) column (`z_pvalues`), and the ONE `se=` validator (`check_se`).
import NetworkCore: newton_fit, logistic_derivatives, z_pvalues, check_se
# ... and the one threaded-loop helper (every task waited for, the first
# failure rethrown unwrapped)
import NetworkCore: spawn_all
# ... and the shared separation verdict and its policy (warn, converged =
# false, flag the separated terms, withhold inference)
import NetworkCore: SeparationVerdict, logistic_separation, warn_separation,
                    separation_caveat

# ----------------------------------------------------------------------------
# Curated re-export of the NetworkCore.jl user-facing API, so that `using ERGM`
# alone provides the network constructors and accessors — mirroring R's
# `library(ergm)` attaching the `network` package.
#
# This is an EXPLICIT list mirroring NetworkCore.jl's own export blocks, not a
# runtime loop over `names(NetworkCore)`: the loop used to skip `Network` itself
# (on a comment that stopped being true when the module was renamed from
# `Network` to `NetworkCore`, so `using ERGM; Network(5)` was an UndefVarError)
# and leaked developer tooling into every user namespace. Deliberately NOT
# re-exported: the golden-fixture harness (`GoldenFixture`, `load_golden`,
# `check_golden`, `golden_report`, `golden_tolerance`), the shared numerics
# a user never types (`bootstrap_cov`, `check_se`, `check_statsapi`), the
# conversion-report mutator `record_drop!`, and the module name `NetworkCore`.
# The "Re-exports" testset pins this list against NetworkCore's frozen inventory
# so it cannot silently fall behind.
# ----------------------------------------------------------------------------

# Core types
export AbstractNetwork, Network, BipartiteNetwork

# Graph interface (Graphs.jl bindings re-exported by NetworkCore.jl)
export nv, ne, vertices, edges, has_vertex, has_edge
export neighbors, inneighbors, outneighbors
export degree, indegree, outdegree
export is_directed
export src, dst      # the endpoint accessors of what `edges(net)` yields

# Network construction
export network, network_initialize
export add_vertex!, add_vertices!, rem_vertex!
export add_edge!, add_edges!, rem_edge!

# Missing-dyad (unobserved tie) mask and the missing-data contract
export set_missing_dyad!, is_missing_dyad, delete_missing_dyad!
export clear_missing_dyads!, missing_dyads, n_missing_dyads
export supports_missing, require_observed, MISSING_POLICIES, missing_policies

# Attribute handling
export get_vertex_attribute, set_vertex_attribute!, delete_vertex_attribute!
export vertex_attribute_vector
export get_edge_attribute, set_edge_attribute!, delete_edge_attribute!
export get_network_attribute, set_network_attribute!, delete_network_attribute!
export list_vertex_attributes, list_edge_attributes, list_network_attributes

# Coercion and conversion (the report type and its read-only accessors; the
# mutator `record_drop!` is adapter-author tooling and stays qualified)
export as_matrix, as_adjacency_matrix, as_edgelist, as_dataframe
export network_from_matrix, network_from_edgelist
export ConversionReport, is_lossless, dropped_fields

# I/O and bundled teaching datasets
export read_pajek, write_pajek, write_graphml, write_edgelist_csv
export load_dataset
export network_from_dataframe

# Shared result presentation and GOF containers
export print_coeftable, format_pvalue, signif_code, SIGNIF_LEGEND, z_pvalues
export GOFStatistic, GOFResult, n_simulations, mc_pvalue
export CoefficientTable

# Shared result-metadata protocol and the tied-event vocabulary
export ResultMetadata, fit_metadata
export estimand, objective, is_exact, se_method, missing_method, tie_method
export approximations
export TIE_POLICIES, check_tie_policy

# Utilities
export network_size, network_density, network_edgecount
export is_two_mode
export permute_vertices, get_neighborhood, get_induced_subgraph

# ----------------------------------------------------------------------------
# ERGM.jl's own API
# ----------------------------------------------------------------------------

# Core types
export AbstractERGMTerm, ERGMFormula, ERGMModel, ERGMResult
export TermSet
export compute, compute_all, name

# Built-in terms
export Edges, Mutual, Triangle, Kstar, OStar, IStar, TwoPath
export Degree, IDegree, ODegree
export NodeFactor, NodeCov, NodeMatch, NodeMismatch, NodeMix, AbsDiff
export EdgeCov, GWESP, GWDSP, GWDegree, GWIDegree, GWODegree
export GWNSP, ESP, Concurrent, DegRange, IDegRange, ODegRange, MeanDeg, Density
export Sender, Receiver
export TransitiveTies, CyclicalTies
export Offset
export CurvedGWESP, CurvedGWDegree
# `TriadCensus` (statnet `triadcensus`) is public but NOT exported: Siena.jl
# exports a GOF statistic of the same name, and two packages exporting one
# name leave it undefined for `using ERGM, Siena`. Write `ERGM.TriadCensus()`
# or `using ERGM: TriadCensus`.
public TriadCensus

# Model fitting (`ergm` is a `const` alias of `fit_ergm`)
export ergm, fit_ergm
export mple, mcmle
export has_dyad_dependent, mcmc_se
# The shared optimizer and logistic kernel, hosted in NetworkCore.jl (`public`
# there) and re-exported here for the ERGM variants and for `using ERGM`
export newton_fit, logistic_derivatives

# Simulation (`mh_toggle!` is the Metropolis kernel the variants adopt)
export simulate_ergm, sample_networks, mh_sample, mh_toggle!

# Diagnostics (`gof` is NetworkCore.jl's shared generic, extended with a method
# for ERGMResult; the explicit export keeps `using ERGM: gof` working)
export gof, mcmc_diagnostics, MCMCDiagnostics

# Utilities
export change_stat, change_stat_all, summary_stats

# The public term-trait protocol (`src/terms/traits.jl`): what a term declares
# about the data it needs and the models it belongs in. Third-party terms add
# methods to these and thereby take part in the same formula validation as the
# built-ins. (`supports_missing`, the missing-data half, is a NetworkCore.jl
# generic already re-exported above.)
export is_dyad_dependent
export required_vertex_attributes, required_edge_attributes
export requires_directed, requires_undirected

# StatsAPI methods (re-exported so `coef(fit)` etc. work with just `using ERGM`;
# `coeftable` and `coefnames` are the NetworkCore/StatsAPI bindings, exported
# here exactly once)
export coef, stderror, vcov, confint, loglikelihood, aic, bic, nobs, dof, coeftable,
       coefnames

# ----------------------------------------------------------------------------
# Public API that is deliberately NOT exported (Julia ≥ 1.11 `public`).
#
# `resolve_method` is R's default-estimator rule (`method=:auto`), shared with
# the variants so that `:auto` means one thing across the family; the MCMLE
# convergence tests (`mcmc_convergence`, `MCMLEConvergence`) are the rule a
# variant's own moment-matching estimator reports.
#
# `Extension` is the stable API for packages that build estimators or term
# families on ERGM.jl (TERGM, ERGMCount, ERGMRank, ERGMEgo, ERGMMulti,
# ERGMUserterms, or a third party): the pseudo-likelihood fitter and its
# boundary rules, the MCMLE drivers, the formula pipeline and the attainable
# range a term declares. Its names carry no underscore and are covered by
# semver; `using ERGM.Extension` brings them in. No underscore name of ERGM
# is `public`: everything with a leading underscore is internal. The
# "Extension API surface" testset pins both.
# ----------------------------------------------------------------------------
public resolve_method
public mcmc_convergence, MCMLEConvergence
public Extension

# The extension API's generic functions are owned by the `Extension`
# submodule; ERGM imports them by name and adds its methods in the source
# files below, exactly as a variant adds methods for its own types.
include("extension.jl")
import .Extension: validate_formula, materialize, expand_terms, collect_terms,
                   n_observed_dyads, require_supported_network,
                   attainable_range, extreme_statistics,
                   boundary_columns, warn_boundary, mple_fit_design,
                   mcmc_defaults, confidence_test, mcmle_solve, mcmle_covariance,
                   bridge_integrate, ess_sample, mcmle_sampler

# Include source files
include("missing.jl")
include("terms/base.jl")
include("terms/structural.jl")
include("terms/nodal.jl")
include("terms/dyadic.jl")
include("terms/traits.jl")
include("terms/materialize.jl")
include("terms/extended.jl")
include("estimation/mple.jl")
include("estimation/mcmle.jl")
include("mcmc/simulation.jl")
include("mcmc/diagnostics.jl")
include("estimation/curved.jl")

# ----------------------------------------------------------------------------
# Precompile workload. Time-to-first-fit was ~5 s for
# the first `mple` on the Florentine data:
# the whole estimation path — term dispatch through the tuple-backed TermSet,
# the compressed design, the Newton kernel, the MH kernel and its closures,
# the convergence tests, GOF — is compiled lazily on first use. Running one
# tiny fit of each kind here at precompile time caches those native-code
# specializations in the package image, for BOTH directedness type
# parameters (`ERGMModel{Int,false}` and `ERGMModel{Int,true}` are different
# specializations). The networks are built inline — no dataset I/O at
# precompile time — everything is seeded and small, and the log output the
# tiny fits emit (a one-iteration MCMLE cannot pass its convergence tests) is
# silenced: a precompile-time warning is never about the user's data.
# ----------------------------------------------------------------------------
@setup_workload begin
    _pc_u = network(7; directed=false)
    for (i, j) in ((1, 2), (2, 3), (1, 3), (3, 4), (4, 5), (2, 5), (5, 6), (1, 5))
        add_edge!(_pc_u, i, j)
    end
    set_vertex_attribute!(_pc_u, :grp,
        Dict(1 => "A", 2 => "A", 3 => "B", 4 => "B", 5 => "A", 6 => "B", 7 => "A"))
    _pc_d = network(6; directed=true)
    for (i, j) in ((1, 2), (2, 1), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3), (1, 5),
                   (5, 6), (6, 2))
        add_edge!(_pc_d, i, j)
    end
    set_vertex_attribute!(_pc_d, :grp,
        Dict(1 => "A", 2 => "B", 3 => "A", 4 => "B", 5 => "A", 6 => "B"))
    # A console logger writing to devnull: the tiny one-iteration MCMLE below
    # cannot pass its convergence tests, and routing its warning through the
    # same logger type a user's REPL has caches the (surprisingly expensive)
    # logging path without printing anything
    _pc_null = Base.CoreLogging.ConsoleLogger(devnull, Base.CoreLogging.Warn)
    @compile_workload begin
        Base.CoreLogging.with_logger(_pc_null) do
            for _pc_net in (_pc_u, _pc_d)
                _pc_rng = Random.Xoshiro(20260909)
                _pc_fit = fit_ergm(_pc_net, [Edges(), NodeMatch(:grp), GWESP(0.5)];
                                   method=:mple)
                coeftable(_pc_fit)
                confint(fit_ergm(_pc_net, [Edges(), NodeMatch(:grp)]))
                show(devnull, _pc_fit); sprint(show, _pc_fit)
                # The most common dyad-dependent specification on its own
                # (`edges + gwesp`, the docs' and fixtures' MCMLE model) is a
                # different TermSet specialization from the three-term one
                _pc_model = ERGMModel(ERGMFormula([Edges(), GWESP(0.5)]), _pc_net)
                _pc_mc = mcmle(_pc_model; n_samples=20, burnin=20, interval=1,
                               maxiter=1, bridge_rungs=1, bridge_samples=10,
                               rng=_pc_rng)
                sprint(show, mcmc_diagnostics(_pc_mc)); sprint(show, _pc_mc); coeftable(_pc_mc)
                mcmle(_pc_model; n_samples=20, burnin=20, interval=1, maxiter=1,
                      bridge_rungs=0, n_chains=2, rng=_pc_rng)
                gof(_pc_fit; n_sim=2, burnin=10, interval=1, rng=_pc_rng)
                simulate_ergm(_pc_fit; n_sim=1, burnin=10, interval=1, rng=_pc_rng)
            end
        end
    end
end

end # module
