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
using Random
using Statistics
using Printf: @sprintf
using PrecompileTools: @setup_workload, @compile_workload

import StatsAPI
# The ecosystem StatsAPI surface (Networks.jl `src/statsapi.jl`): every verb
# is defined on `ERGMResult` and re-exported below. `coeftable` is the SAME
# binding `using Networks` already provides (`Networks.coeftable ===
# StatsAPI.coeftable`), so it is exported exactly once.
import StatsAPI: coef, stderror, vcov, confint, loglikelihood, aic, bic, nobs,
                 dof, coeftable

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
# masked dyads from the pseudo-likelihood), `missing_policies` is extended for
# every MCMC entry point (they take `:condition_on_face`, NOT the generic
# `:face`), and `require_observed` is the shared guard the MCMC routines call.
# Imported by name because we add methods to the traits.
import Networks: supports_missing, require_observed, missing_policies

# The shared result-metadata protocol (Networks.jl `src/results.jl`): seven
# generic accessors that say what a fit actually did (which estimand, which
# objective, whether that objective is exact FOR THIS FIT, how the standard
# errors were computed, how masked dyads and tied events were treated, plus
# free-text caveats). Imported by name because ERGM adds methods for
# `ERGMResult`; `fit_metadata(fit)` then collects them.
import Networks: estimand, objective, is_exact, se_method, missing_method,
                 tie_method, approximations

# Shared numerics, hosted in Networks.jl and imported by name (panel 2026-09,
# items 13, 14, 28): the ONE Newton–Raphson optimizer and logistic-likelihood
# kernel behind the MPLE (`newton_fit`, `logistic_derivatives` — `public` in
# Networks, re-exported here so `using ERGM` is unchanged and
# `ERGM.newton_fit === Networks.newton_fit`), the ONE z → p helper behind every
# Pr(>|z|) column (`z_pvalues`), and the ONE `se=` validator (`check_se`).
import Networks: newton_fit, logistic_derivatives, z_pvalues, check_se

# ----------------------------------------------------------------------------
# Curated re-export of the Networks.jl user-facing API, so that `using ERGM`
# alone provides the network constructors and accessors — mirroring R's
# `library(ergm)` attaching the `network` package (panel 2026-09, item 3).
#
# This is an EXPLICIT list mirroring Networks.jl's own export blocks, not a
# runtime loop over `names(Networks)`: the loop used to skip `Network` itself
# (on a comment that stopped being true when the module was renamed from
# `Network` to `Networks`, so `using ERGM; Network(5)` was an UndefVarError)
# and leaked developer tooling into every user namespace. Deliberately NOT
# re-exported: the golden-fixture harness (`GoldenFixture`, `load_golden`,
# `check_golden`, `golden_report`, `golden_tolerance`), the shared numerics
# a user never types (`bootstrap_cov`, `check_se`, `check_statsapi`), the
# conversion-report mutator `record_drop!`, and the module name `Networks`.
# The "Re-exports" testset pins this list against Networks' frozen inventory
# so it cannot silently fall behind.
# ----------------------------------------------------------------------------

# Core types
export AbstractNetwork, Network, BipartiteNetwork

# Graph interface (Graphs.jl bindings re-exported by Networks.jl)
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

# Model fitting (`ergm` is a `const` alias of `fit_ergm`)
export ergm, fit_ergm
export mple, mcmle
export has_dyad_dependent, mcmc_se
# The shared optimizer and logistic kernel, hosted in Networks.jl (`public`
# there) and re-exported here for the ERGM variants and for `using ERGM`
export newton_fit, logistic_derivatives

# Simulation (`mh_toggle!` is the Metropolis kernel the variants adopt)
export simulate_ergm, sample_networks, mh_sample, mh_toggle!

# Diagnostics (`gof` is Networks.jl's shared generic, extended with a method
# for ERGMResult; the explicit export keeps `using ERGM: gof` working)
export gof, mcmc_diagnostics, MCMCDiagnostics

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

# StatsAPI methods (re-exported so `coef(fit)` etc. work with just `using ERGM`;
# `coeftable` is the Networks/StatsAPI binding, exported here exactly once)
export coef, stderror, vcov, confint, loglikelihood, aic, bic, nobs, dof, coeftable

# ----------------------------------------------------------------------------
# Public API that is deliberately NOT exported (Julia ≥ 1.11 `public`).
#
# Underscore names that downstream packages reach into and may keep calling
# qualified (`ERGM._requires_directed`): they are supported, semver-covered
# bindings — mostly `const` aliases of the public generics, or building blocks
# the variants (TERGM, ERGMMulti, ERGMUserterms, ...) build on — but they are
# not pulled into a user's namespace. New code should prefer the public names
# (`requires_directed`, `required_vertex_attributes`, `has_dyad_dependent`,
# `Networks.z_pvalues`). The "Public private helpers" testset pins the `===`
# identities and the `public` declarations.
# ----------------------------------------------------------------------------
public _requires_directed, _requires_undirected, _vertex_attribute
public _validate_formula, _materialize, _expand_terms, _collect_terms
public _copy_network, _n_dyads, _refuse_two_mode, _refuse_self_loops
public _has_dyad_dependent, _z_pvalues
# The MCMLE convergence machinery and the one dyad-scaled sampler rule, for
# the variants (ERGMEgo's moment matching; every variant's sampler defaults)
public mcmc_convergence, _mcmc_defaults, MCMLEConvergence
# The pseudo-likelihood building blocks the variants' MPLEs run on: R ergm's
# boundary-statistic test and drop semantics, the separation verdict, their
# two R sentences, and the compressed-row design fitter that combines them
# (TERGM's CMPLE, ERGMMulti's and ERGMRank's MPLEs; panel 2026-09, item 13)
public _boundary_columns, _boundary_columns_iterated, _separated
public _warn_boundary, _warn_separated, _mple_fit_design
# The MCMLE's covariance (Fisher plus the Monte-Carlo component), its
# degenerate-sample warning and the bridge estimator of the log-normalizer,
# for a variant with its own MCMC MLE (ERGMRank's `method=:mcmle`)
public _mcmle_covariance, _warn_degenerate_stats, _bridge_logZ

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
include("mcmc/simulation.jl")
include("mcmc/diagnostics.jl")

# ----------------------------------------------------------------------------
# Precompile workload (panel 2026-09, item 18). Time-to-first-fit was ~5 s for
# the first `mple` on the Florentine data (8 s in the panel's measurement):
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
                _pc_fit = fit_ergm(_pc_net, [Edges(), NodeMatch(:grp), GWESP(0.5)])
                coeftable(_pc_fit); confint(_pc_fit)
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
